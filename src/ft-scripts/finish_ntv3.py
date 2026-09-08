"""Lock one validation-selected checkpoint, then evaluate the existing split.

All test-label materialization is experiment-local, after the immutable lock.
No model selection, retraining, or configuration changes use test metrics.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from borzoi_mmap_dataset import BorzoiMmapSeqDataset, read_intervals, build_label_mmap
from grelu.lightning import LightningModel
from grelu.lightning.losses import PoissonMultinomialLoss
from grelu.lightning.metrics import MSE, PearsonCorrCoef
from grelu.model.trunks.ntv3 import CHECKPOINT, CODE_REVISION
from saijou_tasks import TASK_NAMES
from train_borzoi import DEFAULT_GENOME, DEFAULT_BIGWIG_DIR
from train_ntv3 import write_json, state_checksum


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_set(device):
    return {name: cls(num_outputs=4, average=False, log_transform=log).to(device)
            for name, cls, log in (("mse", MSE, False), ("log1p_mse", MSE, True),
                                   ("pearson", PearsonCorrCoef, False),
                                   ("log1p_pearson", PearsonCorrCoef, True))}


def validate_metric_rows(rows, seeds=(17,29,43)):
    expected = {(seed, split, predictor, task) for seed in seeds
                for split in ("val", "test") for predictor in ("ntv3", "bias_only")
                for task in TASK_NAMES}
    keys = [(r["seed"], r["split"], r["predictor"], r["task"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Metric table keys are duplicated, missing, or unexpected")
    for row in rows:
        if not all(np.isfinite(row[key]) for key in ("loss", "mse", "log1p_mse")):
            raise ValueError("Nonfinite loss or MSE in metric table")
        for key in ("pearson", "log1p_pearson"):
            if not (np.isnan(row[key]) or -1 <= row[key] <= 1):
                raise ValueError("Invalid Pearson in metric table")


@torch.no_grad()
def evaluate(model, dataset, means, seed, split):
    device = model.device
    learned, constant = metric_set(device), metric_set(device)
    objective = PoissonMultinomialLoss(total_weight=.2, log_input=True, reduction="none")
    losses = {"ntv3": torch.zeros(4, device=device), "bias_only": torch.zeros(4, device=device)}
    constant_logits = torch.as_tensor(np.log(np.maximum(means, 1e-4)),
                                      dtype=torch.float32, device=device)[None, :, None]
    n = 0
    model.eval()
    for x, y in DataLoader(dataset, batch_size=1, num_workers=2):
        x, y = x.to(device), y.to(device)
        # Match the training precision contract for the learned head.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            logits = model(x, logits=True)
        bias_logits = constant_logits.expand_as(logits)
        if not torch.isfinite(y).all() or (y < 0).any():
            raise FloatingPointError("Invalid evaluation labels")
        for name, raw, metrics in (("ntv3", logits, learned), ("bias_only", bias_logits, constant)):
            pred = raw.float().exp()
            loss = objective(raw, y).reshape(1, 4, -1).mean(dim=(0, 2))
            if not torch.isfinite(pred).all() or not torch.isfinite(loss).all():
                raise FloatingPointError(f"Nonfinite {name} evaluation")
            losses[name] += loss
            for metric in metrics.values():
                metric.update(pred, y)
        n += 1
    rows = []
    for name, metrics in (("ntv3", learned), ("bias_only", constant)):
        values = {key: metric.compute().cpu().tolist() for key, metric in metrics.items()}
        values["loss"] = (losses[name] / n).cpu().tolist()
        if name == "bias_only":
            # Constant predictions have undefined Pearson, including roundoff cases.
            values["pearson"] = values["log1p_pearson"] = [float("nan")] * 4
        for i, task in enumerate(TASK_NAMES):
            rows.append(dict(seed=seed, split=split, predictor=name, task=task,
                             **{key: value[i] for key, value in values.items()}))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / "final_validation_summary.json").exists():
        raise FileExistsError("Evaluation already complete; do not reopen the test")
    seeds = (args.seed,)
    records = [json.loads((root / f"training_seed{seed}.json").read_text()) for seed in seeds]
    for seed, record in zip(seeds, records):
        if (record["status"] != "training_passed" or record["seed"] != seed
                or record["split"] != "split_chr10_chr11"
                or record["train_windows"] != 11201 or record["val_windows"] != 662
                or record["code_revision"] != CODE_REVISION):
            raise ValueError("Incomplete or incompatible full training results")
    if len({r["revision"] for r in records}) != 1:
        raise ValueError("Checkpoint revisions differ")
    split_dir = Path("dataset_store/splits/split_chr10_chr11")
    lock = dict(checkpoint=CHECKPOINT, code_revision=CODE_REVISION,
                revision=records[0]["revision"], task_names=TASK_NAMES,
                geometry=[524288, 196608, 32],
                representation="final_post_skip_deconv_7",
                head="NTv3LocalProfileHead_86148_parameters",
                training=dict(loss="poisson_multinomial", total_weight=.2,
                              optimizer="adam", lr=3e-4, batch_size=records[0].get("batch_size", 1),
                              accumulate_grad_batches=records[0].get("accumulate_grad_batches", 8), max_epochs=40,
                              world_size=records[0].get("world_size", 1),
                              early_stopping_patience=3, selection="minimum_val_loss"),
                split_hashes={split: sha256(split_dir / f"{split}_intervals.bed")
                              for split in ("train", "val", "test")},
                test_intervals_sha256=sha256(split_dir / "test_intervals.bed"),
                selected=[dict(seed=r["seed"], checkpoint=r["best_checkpoint"],
                               sha256=sha256(r["best_checkpoint"])) for r in records])
    lock_path = root / "test_lock.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != lock:
            raise ValueError("Existing immutable test lock differs")
    else:
        with lock_path.open("x") as handle:
            json.dump(lock, handle, indent=2)
    test_intervals = read_intervals(split_dir / "test_intervals.bed")
    if len(test_intervals) != 619:
        raise ValueError("Expected 619 locked test windows")
    test_labels = root / "test_labels.npy"
    if not test_labels.exists():
        pending = root / "test_labels.pending.npy"
        build_label_mmap(test_intervals,
                         [str(Path(DEFAULT_BIGWIG_DIR) / f"{task}.CPM.mapq10.bw") for task in TASK_NAMES],
                         pending, "poisson_multinomial", 196608, 32)
        pending.replace(test_labels)
    labels = np.load(test_labels, mmap_mode="r")
    if labels.shape != (619, 4, 6144) or not np.isfinite(labels).all() or (labels < 0).any():
        raise ValueError("Invalid locked test labels")
    write_json(root / "test_label_validation.json", dict(shape=list(labels.shape),
               sha256=sha256(test_labels), intervals_sha256=lock["test_intervals_sha256"]))
    rows = []
    torch.cuda.set_device(args.device)
    for record in records:
        selected = next(item for item in lock["selected"] if item["seed"] == record["seed"])
        if sha256(selected["checkpoint"]) != selected["sha256"]:
            raise ValueError("Locked checkpoint changed")
        model = LightningModel.load_from_checkpoint(selected["checkpoint"], map_location="cpu").to(f"cuda:{args.device}")
        if (model.model_params["revision"] != lock["revision"]
                or model.train_params["max_epochs"] != 40
                or model.train_params["lr"] != 3e-4
                or model.train_params["loss"] != "poisson_multinomial"):
            raise ValueError("Locked checkpoint configuration differs from MVP contract")
        for split in ("val", "test"):
            dataset = BorzoiMmapSeqDataset(
                intervals=read_intervals(split_dir / f"{split}_intervals.bed"),
                labels_path=test_labels if split == "test" else Path(record["label_cache"]) / "val_labels.npy",
                genome=DEFAULT_GENOME, tasks=TASK_NAMES, seq_len=524288, label_len=196608,
                bin_size=32, rc=False,
            )
            try:
                rows.extend(evaluate(model, dataset, record["train_track_means"], record["seed"], split))
            finally:
                dataset.close_handles()
        if state_checksum(model.model.embedding) != record["trunk_checksum_before"]:
            raise RuntimeError("Frozen trunk changed in locked evaluation")
        del model
        torch.cuda.empty_cache()
    validate_metric_rows(rows, seeds)
    with (root / "metrics_by_seed.tsv").open("x") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    aggregates = []
    for split in ("val", "test"):
        for predictor in ("ntv3", "bias_only"):
            for task in TASK_NAMES:
                group = [r for r in rows if (r["split"], r["predictor"], r["task"]) == (split, predictor, task)]
                for metric in ("loss", "mse", "log1p_mse", "pearson", "log1p_pearson"):
                    values = np.asarray([r[metric] for r in group])
                    aggregates.append(dict(split=split, predictor=predictor, task=task, metric=metric,
                                           median=float(np.median(values)), minimum=float(np.min(values)),
                                           maximum=float(np.max(values))))
    with (root / "aggregate_metrics.tsv").open("x") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(aggregates)
    val_learned = [r for r in rows if r["split"] == "val" and r["predictor"] == "ntv3"]
    val_bias = [r for r in rows if r["split"] == "val" and r["predictor"] == "bias_only"]
    criteria = dict(
        all_seeds_val_loss_better_than_bias=all(
            np.mean([r["loss"] for r in val_learned if r["seed"] == seed]) <
            np.mean([r["loss"] for r in val_bias if r["seed"] == seed]) for seed in seeds),
        all_seeds_hsc_val_log1p_pearson_positive=all(
            r["log1p_pearson"] > 0 for r in val_learned if r["task"] == "hsc"),
        all_seeds_median_track_val_log1p_pearson_positive=all(
            np.median([r["log1p_pearson"] for r in val_learned if r["seed"] == seed]) > 0
            for seed in seeds),
    )
    write_json(root / "final_validation_summary.json", dict(status="complete", seeds=list(seeds),
               test_rows=619, metric_rows=len(rows), test_lock_sha256=sha256(lock_path),
               metrics_sha256=sha256(root / "metrics_by_seed.tsv"),
               validation_go_no_go={key: bool(value) for key,value in criteria.items()},
               caveat="Constant-predictor Pearson is undefined (NaN in TSV). No test-driven retraining."))


if __name__ == "__main__":
    main()
