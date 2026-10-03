"""Measure real full-window forward/backward throughput; never update weights."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from grelu.lightning import LightningModel
from borzoi_mmap_dataset import build_dataset_pair
from saijou_tasks import TASK_NAMES
from train_borzoi import DEFAULT_GENOME, DEFAULT_BIGWIG_DIR
from train_ntv3 import write_json, state_checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_float32_matmul_precision("medium")
    model = LightningModel.load_from_checkpoint(args.checkpoint, map_location="cpu").cuda().train()
    datasets = build_dataset_pair(
        split_name="split_chr10_chr11_tiny50", target_mode="poisson_multinomial",
        bw_files=[str(Path(DEFAULT_BIGWIG_DIR) / f"{t}.CPM.mapq10.bw") for t in TASK_NAMES],
        task_names=TASK_NAMES, genome=DEFAULT_GENOME, split_dir=Path("dataset_store/splits"),
        cache_dir=Path("dataset_store/mmap_caches"),
    )
    before = state_checksum(model.model)
    rows = []
    try:
        for batch_size in (1, 2, 4, 8, 16):
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            fetch_start = time.perf_counter()
            x, y = next(iter(DataLoader(datasets[0], batch_size=batch_size, num_workers=0)))
            fetch_seconds = time.perf_counter() - fetch_start
            x, y = x.cuda(), y.cuda()
            durations = []
            torch.cuda.reset_peak_memory_stats()
            try:
                for repeat in range(4):
                    model.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(x, logits=True)
                        loss = model.loss(logits, y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite benchmark loss")
                    loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all()
                           for p in model.model.head.parameters()):
                        raise FloatingPointError("Invalid head gradients")
                    torch.cuda.synchronize()
                    if repeat:
                        durations.append(time.perf_counter() - start)
                    del logits, loss
                peak = torch.cuda.max_memory_allocated()
                row = dict(batch_size=batch_size, median_batch_seconds=float(np.median(durations)),
                           samples_per_second=batch_size / float(np.median(durations)),
                           peak_cuda_bytes=peak, cpu_fetch_seconds=fetch_seconds, status="passed")
            except torch.cuda.OutOfMemoryError:
                row = dict(batch_size=batch_size, status="oom")
            rows.append(row)
            print(row, flush=True)
            write_json(args.out, dict(checkpoint=args.checkpoint, measurements=rows))
            del x, y
            if row["status"] == "oom" or peak > .75 * torch.cuda.get_device_properties(0).total_memory:
                break
        if state_checksum(model.model) != before:
            raise RuntimeError("Benchmark changed model weights")
    finally:
        for dataset in datasets:
            dataset.close_handles()


if __name__ == "__main__":
    main()
