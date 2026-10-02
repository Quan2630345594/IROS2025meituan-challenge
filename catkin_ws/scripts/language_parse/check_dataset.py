#!/usr/bin/env python3
"""Validate data contracts, SFT labels, vocabulary coverage and group isolation."""

import json

if __package__:
    from .build_dataset import read_jsonl
    from .contract import ROOT, DIRECTIONS, OBJECTS, ISSUE_CODES, get_schema, load_result, system_prompt, validate_result
else:
    from build_dataset import read_jsonl
    from contract import ROOT, DIRECTIONS, OBJECTS, ISSUE_CODES, get_schema, load_result, system_prompt, validate_result


def check():
    schema = get_schema()
    props = schema["properties"]["steps"]["items"]["properties"]
    assert tuple(props["direction"]["enum"]) == DIRECTIONS, "schema/vocabulary drift"
    assert tuple(props["object"]["enum"]) == OBJECTS, "schema/vocabulary drift"
    assert tuple(schema["properties"]["issues"]["items"]["properties"]["code"]["enum"]) == ISSUE_CODES
    splits = {name: read_jsonl(ROOT / "data" / (name + ".records.jsonl")) for name in ("train", "validation", "test")}
    global_ids, global_inputs, family_owner = set(), set(), {}
    for name, rows in splits.items():
        assert rows, "empty split: " + name
        sft = read_jsonl(ROOT / "data" / (name + ".jsonl"))
        assert len(rows) == len(sft), "record/SFT length mismatch"
        for row, chat in zip(rows, sft):
            validate_result(row["output"])
            assert row["id"] not in global_ids, "duplicate record ID"
            global_ids.add(row["id"])
            key = " ".join(row["instruction"].casefold().split())
            assert key not in global_inputs, "duplicate instruction across splits"
            global_inputs.add(key)
            owner = family_owner.setdefault(row["family_id"], name)
            assert owner == name, "rewrite family leakage"
            assert not row.get("holdout") or name == "test", "holdout leakage"
            assert [m["role"] for m in chat["messages"]] == ["system", "user", "assistant"]
            assert chat["messages"][0]["content"] == system_prompt(), "stale training system prompt"
            assert chat["messages"][1]["content"] == row["instruction"]
            assert load_result(chat["messages"][2]["content"]) == row["output"]
    examples = json.loads((ROOT / "data" / "few_shot.json").read_text(encoding="utf-8"))
    train = {r["id"]: r for r in splits["train"]}
    for example in examples:
        assert example["id"] in train, "few-shot test contamination"
        assert example["output"] == train[example["id"]]["output"]
    train_steps = [s for r in splits["train"] for s in r["output"]["steps"]]
    assert set(s["direction"] for s in train_steps) == set(DIRECTIONS)
    assert set(s["object"] for s in train_steps) == set(OBJECTS)
    assert any("zh" in r["tags"] for r in splits["train"]), "missing Chinese training samples"
    assert any(r.get("holdout") for r in splits["test"]), "missing unseen expression test"
    return {"samples": {name: len(rows) for name, rows in splits.items()}, "families": len(family_owner),
            "few_shot_examples": len(examples), "checks": "passed"}


if __name__ == "__main__":
    print(json.dumps(check(), ensure_ascii=False, indent=2))
