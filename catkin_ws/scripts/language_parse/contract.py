"""Shared vocabulary, JSON schema and strict semantic validation (stdlib only)."""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VOCABULARY = {
    "trash": ["trash", "rubbish", "garbage", "trash can", "rubbish bin", "garbage bin", "垃圾", "垃圾桶"],
    "bench": ["bench", "park bench", "rest bench", "长椅", "公园长椅"],
    "billboard": ["billboard", "advertising board", "advertisement board", "advertising panel", "ad board", "billboard sign", "广告牌"],
    "tree": ["tree", "sapling", "树", "树木", "小树"],
    "tractor trailer": ["tractor trailer", "tractor-trailer", "big rig", "semi trailer", "semi-trailer", "18 wheeler", "牵引挂车", "半挂车"],
    "barrel": ["barrel", "cask", "keg", "木桶", "酒桶"],
    "fire hydrant": ["fire hydrant", "fire plug", "water hydrant", "street hydrant", "hydrant", "消防栓", "消火栓"],
    "traffic cone": ["traffic cone", "road cone", "safety cone", "交通锥", "路锥", "锥桶"],
}
DIRECTIONS = ("front", "back", "left", "right")
OBJECTS = tuple(VOCABULARY)
ISSUE_CODES = ("missing_direction", "missing_object", "unsupported_direction", "unsupported_object",
               "ambiguous_direction", "ambiguous_object", "ambiguous_gap", "invalid_gap", "no_action")


def get_schema():
    return json.loads((ROOT / "schema.json").read_text(encoding="utf-8"))


class ParseValidationError(ValueError):
    """The generated result fails the navigation contract."""


def validate_result(result):
    """Reject malformed fields and inconsistent status; preserve the original order.

    Semantic fidelity to the input is measured by evaluation, not provable by schema.
    Incomplete steps are represented by issues, never by fabricated targets.
    """
    def require(condition, message):
        if not condition:
            raise ParseValidationError(message)

    require(isinstance(result, dict), "result must be an object")
    require(set(result) == {"status", "steps", "issues"}, "unexpected or missing root fields")
    require(result["status"] in ("complete", "needs_clarification"), "invalid status")
    require(isinstance(result["steps"], list), "steps must be an array")
    require(isinstance(result["issues"], list), "issues must be an array")
    for i, step in enumerate(result["steps"]):
        require(isinstance(step, dict), "step %d must be an object" % i)
        require(set(step) == {"direction", "object", "target_gap_m"}, "invalid step fields")
        require(isinstance(step["direction"], str) and step["direction"] in DIRECTIONS, "invalid direction")
        require(isinstance(step["object"], str) and step["object"] in OBJECTS, "invalid object")
        gap = step["target_gap_m"]
        require(gap is None or (type(gap) in (int, float) and gap >= 0 and (type(gap) is int or math.isfinite(gap))),
                "target_gap_m must be a finite nonnegative number or null")
    for issue in result["issues"]:
        require(isinstance(issue, dict), "issue must be an object")
        require(set(issue) == {"code", "message"}, "invalid issue fields")
        require(isinstance(issue["code"], str) and issue["code"] in ISSUE_CODES, "invalid issue code")
        require(isinstance(issue["message"], str) and bool(issue["message"].strip()), "empty issue message")
    if result["status"] == "complete":
        require(bool(result["steps"]), "complete result requires at least one step")
        require(not result["issues"], "complete result cannot contain issues")
    else:
        require(bool(result["issues"]), "needs_clarification requires issues")
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ParseValidationError("duplicate JSON field: " + key)
        result[key] = value
    return result


def load_result(text):
    try:
        result = json.loads(text, object_pairs_hook=_unique_object, parse_constant=lambda value: (_ for _ in ()).throw(
            ParseValidationError("invalid JSON constant: " + value)))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ParseValidationError("model output is not a JSON object: %s" % exc) from exc
    return validate_result(result)


def system_prompt():
    synonyms = json.dumps(VOCABULARY, ensure_ascii=False)
    return (ROOT / "system_prompt.txt").read_text(encoding="utf-8").strip() + "\nObject aliases: " + synonyms
