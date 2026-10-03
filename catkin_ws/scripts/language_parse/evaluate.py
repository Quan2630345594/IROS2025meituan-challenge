#!/usr/bin/env python3
"""Run schema/LoRA-schema baselines or score previously saved predictions."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

if __package__:
    from .build_dataset import read_jsonl, write_jsonl
    from .contract import ROOT, load_result, validate_result
    from .parse_goal_direction import parse_instruction
else:
    from build_dataset import read_jsonl, write_jsonl
    from contract import ROOT, load_result, validate_result
    from parse_goal_direction import parse_instruction


def issue_codes(result):
    return sorted(i["code"] for i in result["issues"])


def same_gap(left, right):
    if left is None or right is None:
        return left is right
    return abs(left - right) < 1e-6


def score(records, predictions):
    """Score aligned fields; missing/extra steps and failed requests count as errors."""
    ids = [r["id"] for r in records]
    pred_ids = [p["id"] for p in predictions]
    if len(set(ids)) != len(ids) or len(set(pred_ids)) != len(pred_ids):
        raise ValueError("duplicate record or prediction IDs")
    if set(ids) != set(pred_ids):
        raise ValueError("prediction IDs must match dataset IDs exactly")
    by_id = {p["id"]: p for p in predictions}
    counts = Counter()
    total = len(records)
    if not total:
        raise ValueError("evaluation dataset cannot be empty")
    for row in records:
        gold = validate_result(row["output"])
        prediction = by_id[row["id"]]
        pred = prediction.get("output")
        try:
            validate_result(pred)
        except ValueError:
            pred = None
        counts["valid_outputs"] += pred is not None
        counts["errors"] += pred is None
        counts["gold_clarification"] += gold["status"] == "needs_clarification"
        counts["gold_gap_slots"] += sum(s["target_gap_m"] is not None for s in gold["steps"])
        counts["status_correct"] += pred is not None and pred["status"] == gold["status"]
        counts["issues_correct"] += pred is not None and issue_codes(pred) == issue_codes(gold)
        if pred and pred["status"] == "needs_clarification":
            counts["pred_clarification"] += 1
            counts["clarification_tp"] += gold["status"] == "needs_clarification"
        pred_steps = pred["steps"] if pred else []
        gold_steps = gold["steps"]
        count_match = pred is not None and len(pred_steps) == len(gold_steps)
        counts["step_count_correct"] += count_match
        route_match = pred is not None and [(s["direction"], s["object"]) for s in pred_steps] == [
            (s["direction"], s["object"]) for s in gold_steps]
        counts["ordered_route_correct"] += route_match
        full_match = route_match and pred["status"] == gold["status"] and issue_codes(pred) == issue_codes(gold)
        slots = max(len(gold_steps), len(pred_steps))
        counts["field_slots"] += slots
        for i in range(slots):
            gs = gold_steps[i] if i < len(gold_steps) else None
            ps = pred_steps[i] if i < len(pred_steps) else None
            if gs is None or ps is None:
                full_match = False
                continue
            counts["direction_correct"] += gs["direction"] == ps["direction"]
            counts["object_correct"] += gs["object"] == ps["object"]
            gap_match = same_gap(gs["target_gap_m"], ps["target_gap_m"])
            counts["gap_correct"] += gap_match
            if gs["target_gap_m"] is not None:
                counts["explicit_gap_correct"] += gap_match
            full_match = full_match and gap_match
        counts["exact_correct"] += bool(full_match)
    def ratio(numerator, denominator):
        return counts[numerator] / denominator if denominator else None
    return {"samples": total, "valid_rate": ratio("valid_outputs", total), "errors": counts["errors"],
            "exact_accuracy": ratio("exact_correct", total), "status_accuracy": ratio("status_correct", total),
            "issue_codes_accuracy": ratio("issues_correct", total), "step_count_accuracy": ratio("step_count_correct", total),
            "ordered_route_accuracy": ratio("ordered_route_correct", total),
            "direction_accuracy": ratio("direction_correct", counts["field_slots"]),
            "object_accuracy": ratio("object_correct", counts["field_slots"]),
            "gap_accuracy": ratio("gap_correct", counts["field_slots"]),
            "explicit_gap_accuracy": ratio("explicit_gap_correct", counts["gold_gap_slots"]),
            "explicit_gap_slots": counts["gold_gap_slots"],
            "clarification_precision": ratio("clarification_tp", counts["pred_clarification"]),
            "clarification_recall": ratio("clarification_tp", counts["gold_clarification"])}


def run(records, mode, model, base_url, timeout, few_shot, backend="ollama"):
    predictions = []
    for index, row in enumerate(records, 1):
        prediction = {"id": row["id"], "model": model, "mode": mode, "output": None}
        try:
            prediction["output"] = parse_instruction(row["instruction"], model=model, base_url=base_url,
                                                     timeout=timeout, few_shot=few_shot, backend=backend)
        except (ValueError, RuntimeError) as exc:
            prediction["error"] = str(exc)
        predictions.append(prediction)
        print("%s %d/%d" % (mode, index, len(records)), file=sys.stderr)
    return predictions


def report(records, predictions):
    result = {"all": score(records, predictions)}
    for name, subset in (("unseen_expression", [r for r in records if "unseen_expression" in r["tags"]]),
                         ("english", [r for r in records if "en" in r["tags"]]),
                         ("chinese", [r for r in records if "zh" in r["tags"]]),
                         ("clarification", [r for r in records if r["output"]["status"] == "needs_clarification"]),
                         ("gap", [r for r in records if any(s["target_gap_m"] is not None for s in r["output"]["steps"])])):
        if subset:
            ids = {r["id"] for r in subset}
            result[name] = score(subset, [p for p in predictions if p["id"] in ids])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "test.records.jsonl")
    parser.add_argument("--mode", choices=["schema", "lora-schema"], default="schema")
    parser.add_argument("--model", default="qwen2.5vl:7b", help="For lora-schema, supply the deployed LoRA model name")
    parser.add_argument("--base-url")
    parser.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--no-few-shot", action="store_true")
    parser.add_argument("--predictions", type=Path, help="Score an existing predictions JSONL without inference")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    if args.mode == "lora-schema" and args.model == "qwen2.5vl:7b" and not args.predictions:
        parser.error("supply the deployed LoRA model with --model")
    rows = read_jsonl(args.dataset)
    predictions = read_jsonl(args.predictions) if args.predictions else run(
        rows, args.mode, args.model, args.base_url, args.timeout, not args.no_few_shot, args.backend)
    result = {"mode": args.mode, "model": args.model, "backend": args.backend,
              "few_shot": not args.no_few_shot,
              "dataset": os.path.relpath(args.dataset, Path.cwd()), "metrics": report(rows, predictions)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.predictions:
        write_jsonl(args.output_dir / (args.mode + ".predictions.jsonl"), predictions)
    (args.output_dir / (args.mode + ".metrics.json")).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["metrics"]["all"]["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
