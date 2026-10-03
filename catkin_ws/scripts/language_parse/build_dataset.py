#!/usr/bin/env python3
"""Build grouped SFT splits from saved structured annotations."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import random
import re
from pathlib import Path

if __package__:
    from .contract import ROOT, DIRECTIONS, OBJECTS, system_prompt, validate_result
else:
    from contract import ROOT, DIRECTIONS, OBJECTS, system_prompt, validate_result


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def route_family(output):
    # All bilingual rewrites of the same route, including different gaps, stay together.
    return "route-" + digest(json.dumps([(s["direction"], s["object"]) for s in output["steps"]]))


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def build(seed=42):
    metadata = json.loads(
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(args.seed), ensure_ascii=False, indent=2))
