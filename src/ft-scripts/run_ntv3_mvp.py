"""Run one seed with three-rank DDP, then lock and evaluate its best checkpoint."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from train_ntv3 import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tiny50_result", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--finetune_mode", choices=["frozen", "full"], default="frozen")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--accumulate_grad_batches", type=int, required=True)
    args = parser.parse_args()
    gate = json.loads(args.tiny50_result.read_text())
    if (gate["status"] != "training_passed" or gate["train_windows"] != 225
            or gate["revision"] != args.revision or gate["checkpoint_prediction_max_abs_diff"] != 0
            or gate.get("world_size") != 3 or gate.get("batch_size") != args.batch_size
            or gate.get("finetune_mode", "frozen") != args.finetune_mode
            or gate.get("gradient_checkpointing", False) != args.gradient_checkpointing
            or gate.get("accumulate_grad_batches") != args.accumulate_grad_batches):
        raise ValueError("Selected three-GPU tiny50 configuration has not passed")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    status = dict(status="training", seeds=[args.seed], world_size=3,
                  batch_size=args.batch_size, accumulate_grad_batches=args.accumulate_grad_batches,
                  effective_batch_size=3 * args.batch_size * args.accumulate_grad_batches,
                  max_epochs=args.max_epochs, split="split_chr10_chr11",
                  revision=args.revision, finetune_mode=args.finetune_mode,
                  lr=args.lr, gradient_checkpointing=args.gradient_checkpointing,
                  resumed_from=str(Path(args.checkpoint_path).resolve()) if args.checkpoint_path else None)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2", HF_HUB_OFFLINE="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    write_json(root / "pipeline_status.json", status)
    try:
        with (root / "training.log").open("x") as log:
            cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=3",
                   str(scripts / "train_ntv3.py"), "--revision", args.revision,
                   "--split_name", "split_chr10_chr11", "--devices", "0,1,2",
                   "--seed", str(args.seed), "--max_epochs", str(args.max_epochs),
                   "--batch_size", str(args.batch_size), "--accumulate_grad_batches", str(args.accumulate_grad_batches),
                   "--finetune_mode", args.finetune_mode,
                   "--out", str(root / f"training_seed{args.seed}.json")]
            if args.checkpoint_path:
                cmd.extend(["--checkpoint_path", str(Path(args.checkpoint_path).resolve())])
            if args.gradient_checkpointing:
                cmd.append("--gradient_checkpointing")
            if args.lr is not None:
                cmd.extend(["--lr", str(args.lr)])
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        status["status"] = "locked_evaluation"
        write_json(root / "pipeline_status.json", status)
        with (root / "evaluation.log").open("x") as log:
            subprocess.run([sys.executable, str(scripts / "finish_ntv3.py"),
                            "--root", str(root), "--device", "0", "--seed", str(args.seed)],
                           check=True, stdout=log, stderr=subprocess.STDOUT,
                           env=dict(env, CUDA_VISIBLE_DEVICES="0"))
        status["status"] = "complete"
        write_json(root / "pipeline_status.json", status)
    except Exception as exc:
        status.update(status="failed", error=str(exc))
        write_json(root / "pipeline_status.json", status)
        raise


if __name__ == "__main__":
    main()
