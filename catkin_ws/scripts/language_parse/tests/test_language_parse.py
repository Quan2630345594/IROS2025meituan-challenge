"""Offline contract and transport regressions; these do not measure model semantics."""

import copy
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_dataset import read_jsonl
from check_dataset import check
from contract import ParseValidationError, get_schema, load_result, validate_result
from evaluate import score
from parse_goal_direction import (ClarificationRequired, InferenceError, build_messages, from_legacy,
                                  ollama_chat, openai_chat, parse_goal_direction, parse_instruction,
                                  standardize_output, to_legacy, validate_direction)
from train_lora import build_command


def good():
    return {"status": "complete", "steps": [
        {"direction": "front", "object": "tree", "target_gap_m": None},
        {"direction": "right", "object": "fire hydrant", "target_gap_m": 0.5}], "issues": []}


def fake_response(result=None, **changes):
    payload = {"done": True, "done_reason": "stop", "message": {"content": json.dumps(result or good())}}
    payload.update(changes)
    return io.BytesIO(json.dumps(payload).encode("utf-8"))


class ContractTests(unittest.TestCase):
    def test_old_round_trip_keeps_multiword_classes(self):
        old = to_legacy(good())
        self.assertEqual(old, '"front", "tree", "right", "fire hydrant"')
        self.assertTrue(validate_direction(old))
        self.assertEqual(from_legacy(old)["steps"][1]["object"], "fire hydrant")
        self.assertIsNone(from_legacy(old)["steps"][1]["target_gap_m"])

    def test_single_front_empty_odd_unquoted_or_wrong_order_rejected(self):
        for value in ('"front"', '', '"front", "tree", "right"', 'front, tree', '"tree", "front"',
                      '"front", "bucket"', '"left front", "tree"', None):
            with self.subTest(value=value):
                self.assertFalse(validate_direction(value))

    def test_standardize_handles_spacing_without_guessing_words(self):
        self.assertEqual(standardize_output(' "front","tractor trailer" '), '"front", "tractor trailer"')
        with self.assertRaises(ValueError):
            standardize_output('"front", "tractortrailer"')

    def test_gaps_reject_bool_string_negative_nonfinite(self):
        for value in (True, "0.5", -1, float("nan"), float("inf")):
            result = good()
            result["steps"][0]["target_gap_m"] = value
            with self.subTest(value=value), self.assertRaises(ParseValidationError):
                validate_result(result)
        for value in (None, 0, 0.5):
            result = good()
            result["steps"][0]["target_gap_m"] = value
            validate_result(result)

    def test_status_consistency_and_extra_fields(self):
        variants = []
        r = good(); r["steps"] = []; variants.append(r)
        r = good(); r["status"] = "needs_clarification"; variants.append(r)
        r = good(); r["issues"] = [{"code": "missing_object", "message": "Specify target."}]; variants.append(r)
        r = good(); r["steps"][0]["surprise"] = 1; variants.append(r)
        r = good(); del r["steps"][0]["target_gap_m"]; variants.append(r)
        r = good(); r["status"] = "partial"; variants.append(r)
        for result in variants:
            with self.subTest(result=result), self.assertRaises(ParseValidationError):
                validate_result(result)

    def test_partial_route_cannot_be_executed_through_legacy(self):
        result = good()
        result["status"] = "needs_clarification"
        result["issues"] = [{"code": "missing_object", "message": "Specify final target."}]
        validate_result(result)
        with self.assertRaises(ClarificationRequired) as caught:
            to_legacy(result)
        self.assertIs(caught.exception.result, result)

    def test_duplicate_json_keys_codefences_and_nan_rejected(self):
        for text in ('{"status":"complete","status":"needs_clarification","steps":[],"issues":[]}',
                     '```json\n' + json.dumps(good()) + '\n```', json.dumps(good()).replace('0.5', 'NaN')):
            with self.subTest(text=text), self.assertRaises(ParseValidationError):
                load_result(text)

    def test_unknown_issue_and_empty_message(self):
        for code, message in (("other", "clarify"), ("no_action", " ")):
            with self.assertRaises(ParseValidationError):
                validate_result({"status": "needs_clarification", "steps": [], "issues": [{"code": code, "message": message}]})


class TransportTests(unittest.TestCase):
    @patch("parse_goal_direction.urlopen")
    def test_one_schema_call_low_temperature_and_separate_user_message(self, request):
        request.return_value = fake_response()
        result = parse_instruction("向前走到树，再向右到消防栓", base_url="http://localhost:11434/v1/", few_shot=False)
        self.assertEqual(result, good())
        request.assert_called_once()
        req = request.call_args.args[0]
        body = json.loads(req.data)
        self.assertEqual(req.full_url, "http://localhost:11434/api/chat")
        self.assertEqual(body["format"], get_schema())
        self.assertEqual(body["options"]["temperature"], 0)
        self.assertFalse(body["stream"])
        self.assertEqual(body["messages"][-1]["role"], "user")

    @patch("parse_goal_direction.urlopen")
    def test_original_api(self, request):
        request.return_value = fake_response()
        self.assertEqual(parse_goal_direction("front to tree right to hydrant", few_shot=False),
                         '"front", "tree", "right", "fire hydrant"')

    @patch("parse_goal_direction.urlopen")
    def test_openai_json_schema_deployment(self, request):
        envelope = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(good())}}]}
        request.return_value = io.BytesIO(json.dumps(envelope).encode("utf-8"))
        self.assertEqual(parse_instruction("go forward to tree", backend="openai", model="navigation-lora", few_shot=False), good())
        body = json.loads(request.call_args.args[0].data)
        self.assertEqual(body["response_format"]["json_schema"]["schema"], get_schema())
        self.assertEqual(body["model"], "navigation-lora")

    @patch("parse_goal_direction.urlopen")
    def test_truncation_error_and_invalid_envelope(self, request):
        for changes in ({"done": False}, {"done_reason": "length"}, {"error": "bad model"}, {"message": {}}):
            request.return_value = fake_response(**changes)
            with self.subTest(changes=changes), self.assertRaises(InferenceError):
                ollama_chat([])

    @patch("parse_goal_direction.urlopen", side_effect=URLError("offline"))
    def test_network_failure_is_actionable(self, request):
        with self.assertRaisesRegex(InferenceError, "Ollama request failed"):
            ollama_chat([])
        request.assert_called_once()

    @patch("parse_goal_direction.urlopen")
    def test_http_failure_includes_server_detail(self, request):
        request.side_effect = HTTPError("http://localhost", 400, "bad", {}, io.BytesIO(b"unsupported schema"))
        with self.assertRaisesRegex(InferenceError, "unsupported schema"):
            ollama_chat([])

    @patch("parse_goal_direction.urlopen")
    def test_openai_truncation_rejected(self, request):
        request.return_value = io.BytesIO(b'{"choices":[{"finish_reason":"length","message":{"content":"{}"}}]}')
        with self.assertRaises(InferenceError):
            openai_chat([])

    def test_empty_input_bad_backend_or_url(self):
        for text in ("", " ", None):
            with self.assertRaises(ValueError):
                build_messages(text)
        with self.assertRaises(ValueError):
            parse_instruction("go forward to tree", backend="bad")
        with self.assertRaises(ValueError):
            ollama_chat([], base_url="file:///tmp")


class DataAndEvaluationTests(unittest.TestCase):
    def test_full_data_checks(self):
        self.assertEqual(check()["checks"], "passed")

    def test_perfect_and_reordered_predictions(self):
        row = {"id": "one", "output": good()}
        metrics = score([row], [{"id": "one", "output": good()}])
        self.assertEqual(metrics["exact_accuracy"], 1)
        self.assertEqual(metrics["explicit_gap_accuracy"], 1)
        reordered = good()
        reordered["steps"].reverse()
        metrics = score([row], [{"id": "one", "output": reordered}])
        self.assertEqual(metrics["ordered_route_accuracy"], 0)
        self.assertEqual(metrics["direction_accuracy"], 0)

    def test_missing_extra_steps_and_errors_penalized(self):
        row = {"id": "one", "output": good()}
        missing = good(); missing["steps"].pop()
        metrics = score([row], [{"id": "one", "output": missing}])
        self.assertEqual(metrics["direction_accuracy"], 0.5)
        extra = good(); extra["steps"].append(copy.deepcopy(extra["steps"][0]))
        metrics = score([row], [{"id": "one", "output": extra}])
        self.assertAlmostEqual(metrics["direction_accuracy"], 2 / 3)
        metrics = score([row], [{"id": "one", "error": "offline"}])
        self.assertEqual(metrics["errors"], 1)
        self.assertEqual(metrics["exact_accuracy"], 0)
        self.assertEqual(metrics["direction_accuracy"], 0)

    def test_issue_message_wording_ignored_but_codes_checked(self):
        result = {"status": "needs_clarification", "steps": [], "issues": [{"code": "missing_direction", "message": "Clarify."}]}
        pred = copy.deepcopy(result); pred["issues"][0]["message"] = "请说明方向。"
        metrics = score([{"id": "one", "output": result}], [{"id": "one", "output": pred}])
        self.assertEqual(metrics["exact_accuracy"], 1)
        self.assertEqual(metrics["clarification_recall"], 1)

    def test_prediction_id_integrity(self):
        with self.assertRaises(ValueError):
            score([{"id": "one", "output": good()}], [{"id": "different", "output": good()}])

    def test_all_test_labels_score_perfectly_as_scorer_sanity_check(self):
        rows = read_jsonl(ROOT / "data" / "test.records.jsonl")
        metrics = score(rows, [{"id": r["id"], "output": r["output"]} for r in rows])
        self.assertEqual(metrics["exact_accuracy"], 1)

    def test_training_command_uses_explicit_validation_no_test(self):
        config = json.loads((ROOT / "training" / "lora_config.json").read_text(encoding="utf-8"))
        command = build_command(config)
        self.assertIn("--val_dataset", command)
        self.assertIn("--external_plugins", command)
        self.assertNotIn("--callbacks", command)
        self.assertFalse(any("test.jsonl" in arg for arg in command))
        self.assertEqual(command[command.index("--loss_scale") + 1], "last_round")


if __name__ == "__main__":
    unittest.main()
