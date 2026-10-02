#!/usr/bin/env python3
"""Import the old examples without executing their code; build grouped SFT splits."""

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import random
import re
from pathlib import Path

if __package__:
    from .contract import ROOT, DIRECTIONS, OBJECTS, system_prompt, validate_result
    from .parse_goal_direction import from_legacy
else:
    from contract import ROOT, DIRECTIONS, OBJECTS, system_prompt, validate_result
    from parse_goal_direction import from_legacy


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def route_family(output):
    # All bilingual rewrites of the same route, including different gaps, stay together.
    return "route-" + digest(json.dumps([(s["direction"], s["object"]) for s in output["steps"]]))


def extract_prompt(source):
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_goal_direction")
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "prompt" for t in node.targets):
            if not isinstance(node.value, ast.JoinedStr):
                raise ValueError("expected original f-string prompt")
            parts = []
            for value in node.value.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    parts.append(value.value)
                elif isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name) and value.value.id == "language_instr":
                    parts.append("{language_instr}")
                else:
                    raise ValueError("unrecognized original prompt interpolation")
            return "".join(parts)
    raise ValueError("original prompt not found")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def import_original(source):
    prompt = extract_prompt(source)
    (ROOT / "reference").mkdir(exist_ok=True)
    snapshot = "\n".join(line.rstrip() for line in prompt.splitlines()).rstrip() + "\n"
    (ROOT / "reference" / "legacy_prompt.txt").write_text(snapshot, encoding="utf-8")
    accepted, rejected, seen = [], [], set()
    section = ""
    # These aliases are intentionally NOT silently re-labelled under the new taxonomy.
    broad = re.compile(r"\b(timber|shrub|bush|bucket|vat|drum|seat|lorry|truck|signboard|pylon|cone|traffic pyramid)\b", re.I)
    for line in prompt.splitlines():
        if line.strip().startswith("##"):
            section = line.strip()
        match = re.match(r'^\s*(\d+)\.\s*(.+?)\s*(?:→|->)\s*(".+)\s*$', line)
        if not match:
            continue
        number, instruction, labels = match.groups()
        labels = re.sub(r'"\s+([^"\n]+)"', r'"\1"', labels)
        row = {"instruction": instruction.strip(), "section": section, "number": int(number), "legacy_label": labels.strip()}
        reason = None
        # Match conservative exclusions after removing the allowed road/safety/traffic cone phrases.
        checked = re.sub(r"\b(?:traffic|road|safety) cone\b", "", instruction, flags=re.I)
        if broad.search(checked):
            reason = "旧示例使用了已移除的过宽物体映射，需人工重标"
        if re.search(r"\b(?:walk|go|move|reach)(?: to| the)\b", instruction, re.I):
            reason = "旧示例把无方向移动补成 front；新规则需明确方向或待澄清"
        normalized = " ".join(instruction.casefold().split())
        if normalized in seen:
            reason = "原句大小写/空白重复，已去重"
        seen.add(normalized)
        try:
            output = from_legacy(row["legacy_label"])
        except ValueError as exc:
            reason = "无效旧标签: " + str(exc)
        if reason:
            rejected.append({**row, "reason": reason})
        else:
            accepted.append({"id": "old-" + digest(normalized), "family_id": route_family(output),
                             "instruction": row["instruction"], "output": output,
                             "tags": ["legacy_seed", "en", "at_to" if '"at"' in section else "route"],
                             "source": {"section": section, "number": int(number)}})
    if not accepted:
        raise ValueError("no usable original examples found")
    write_jsonl(ROOT / "data" / "seed_records.jsonl", accepted)
    write_jsonl(ROOT / "data" / "seed_quarantine.jsonl", rejected)
    return {"original_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "accepted_seeds": len(accepted), "quarantined_seeds": len(rejected)}


def generated_records():
    records = []
    english = {"front": "Go straight", "back": "Go back", "left": "Turn left", "right": "Turn right"}
    chinese = {"front": "向前走", "back": "向后走", "left": "向左走", "right": "向右走"}
    objects_zh = ["垃圾桶", "长椅", "广告牌", "树", "半挂车", "木桶", "消防栓", "交通锥"]
    for direction in DIRECTIONS:
        for obj, zh_obj in zip(OBJECTS, objects_zh):
            output = {"status": "complete", "steps": [{"direction": direction, "object": obj, "target_gap_m": None}], "issues": []}
            for language, instruction in (("en", english[direction] + " to the " + obj), ("zh", chinese[direction] + "到" + zh_obj + "那里")):
                records.append({"id": "basic-" + digest(instruction), "family_id": route_family(output),
                                "instruction": instruction, "output": output, "tags": ["basic", language]})
    edges = json.loads((ROOT / "data" / "edge_cases.json").read_text(encoding="utf-8"))
    for edge in edges:
        validate_result(edge["output"])
        # Valid route variants join the seed route family; issue rewrites share explicit family.
        family = route_family(edge["output"]) if edge["output"]["status"] == "complete" else "edge-" + edge["family"]
        for instruction in edge["instructions"]:
            records.append({"id": "edge-" + digest(instruction), "family_id": family,
                            "instruction": instruction, "output": edge["output"],
                            "tags": [t for t in edge["tags"] if t not in ("en", "zh")] +
                                    ["zh" if re.search(r"[\u4e00-\u9fff]", instruction) else "en"],
                            "holdout": edge.get("holdout", False)})
    return records


def build(source=None, seed=42):
    metadata = import_original(source.read_text(encoding="utf-8-sig")) if source else json.loads(
        (ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))["import"]
    candidates = read_jsonl(ROOT / "data" / "seed_records.jsonl") + generated_records()
    groups, seen = defaultdict(list), {}
    for record in candidates:
        validate_result(record["output"])
        key = " ".join(record["instruction"].casefold().split())
        if key in seen:
            if record["output"] != seen[key]["output"]:
                raise ValueError("conflicting labels for " + record["instruction"])
            continue
        seen[key] = record
        groups[record["family_id"]].append(record)
    # Hold out entire families if any member uses deliberately unseen phrasing.
    held = {family for family, rows in groups.items() if any(row.get("holdout") for row in rows)}
    families = sorted(set(groups) - held)
    random.Random(seed).shuffle(families)
    n = len(families)
    val_count, test_count = max(1, round(n * 0.1)), max(1, round(n * 0.1))
    split_families = {"validation": set(families[:val_count]),
                      "test": set(families[val_count:val_count + test_count]) | held,
                      "train": set(families[val_count + test_count:])}
    splits = {name: sorted([r for family in members for r in groups[family]], key=lambda r: r["id"])
              for name, members in split_families.items()}
    # Keep rare issues in training by deterministic family moves; holdouts never move.
    train_codes = {i["code"] for r in splits["train"] for i in r["output"]["issues"]}
    all_codes = {i["code"] for rows in groups.values() for r in rows for i in r["output"]["issues"]}
    for code in sorted(all_codes - train_codes):
        family = next((f for f in families if any(i["code"] == code for r in groups[f] for i in r["output"]["issues"])), None)
        if family:
            for members in split_families.values():
                members.discard(family)
            split_families["train"].add(family)
    splits = {name: sorted([r for family in members for r in groups[family]], key=lambda r: r["id"])
              for name, members in split_families.items()}
    prompt = system_prompt()
    for name, rows in splits.items():
        write_jsonl(ROOT / "data" / (name + ".records.jsonl"), rows)
        write_jsonl(ROOT / "data" / (name + ".jsonl"), [{"messages": [
            {"role": "system", "content": prompt}, {"role": "user", "content": r["instruction"]},
            {"role": "assistant", "content": json.dumps(r["output"], ensure_ascii=False)}]} for r in rows])
    # Runtime examples are selected ONLY from the training split.
    examples, categories = [], defaultdict(list)
    for row in sorted(splits["train"], key=lambda r: ("basic" in r["tags"], r["id"])):
        result = row["output"]
        category = "at_to" if "at_to" in row["tags"] else (
            result["issues"][0]["code"] if result["issues"] else
            "gap" if any(s["target_gap_m"] is not None for s in result["steps"]) else
            "multi" if len(result["steps"]) > 1 else "single")
        categories[category].append(row)
    for category in ("single", "multi", "at_to", "gap", "missing_direction", "missing_object", "ambiguous_direction", "unsupported_object"):
        if categories[category]:
            row = categories[category][0]
            examples.append({"id": row["id"], "instruction": row["instruction"], "output": row["output"]})
    (ROOT / "data" / "few_shot.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"seed": seed, "grouping": "same ordered canonical direction-object route; explicit issue rewrite families",
                "label_review": "Rule-screened legacy seeds and deterministic authored labels; human review required before production training",
                "import": metadata, "split_counts": {name: len(rows) for name, rows in splits.items()},
                "family_counts": {name: len(members) for name, members in split_families.items()},
                "unseen_test_records": sum(bool(r.get("holdout")) for r in splits["test"]),
                "status_counts": {name: dict(Counter(r["output"]["status"] for r in rows)) for name, rows in splits.items()}}
    (ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Original script, read as AST without importing/executing it")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.seed), ensure_ascii=False, indent=2))
