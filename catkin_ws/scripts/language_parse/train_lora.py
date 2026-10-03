#!/usr/bin/env python3
"""Print or launch a checked ms-swift LoRA-SFT command, with no shell interpolation."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

if __package__:
    from .check_dataset import check
    from .contract import ROOT
else:
    from check_dataset import check
    from contract import ROOT


def build_command(config, model=None, output_dir=None):
    config = dict(config)
    if model:
        config["model"] = Path(os.path.relpath(model, ROOT)).as_posix() if Path(model).exists() else model
    command = ["swift", "sft"]
    for name, value in config.items():
        command.append("--" + name)
        command.append(str(value).lower() if isinstance(value, bool) else str(value))
    command.extend(["--dataset", "data/train.jsonl",
                    "--val_dataset", "data/validation.jsonl",
                    "--output_dir", "output/qwen2.5vl-navigation" if output_dir is None else Path(os.path.relpath(output_dir, ROOT)).as_posix(),
                    "--external_plugins", "training/early_stop.py"])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "training" / "lora_config.json")
    parser.add_argument("--model", help="Override with exact matching trainable base weights or local model path")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run", action="store_true", help="Actually train; default only validates and prints argv")
    args = parser.parse_args()
    check()
    command = build_command(json.loads(args.config.read_text(encoding="utf-8")), args.model, args.output_dir)
    if not args.run:
        print(json.dumps(command, ensure_ascii=False, indent=2))
        return 0
    executable = shutil.which("swift")
    if not executable:
        parser.error("ms-swift is missing; install requirements-training.txt in your CUDA environment")
    command[0] = executable
    return subprocess.run(command, cwd=str(ROOT), check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
