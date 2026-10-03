import torch
import time
import pandas as pd
import numpy as np
from grelu.lightning import LightningModel
from scripts.smoke_tests.config import WEIGHTS_PATH, AG_META_PATH, AG_INPUT_LEN, AG_BIN_SIZE

def benchmark_ag(n_variants=50, devices=[0]):
    print(f"Benchmarking AlphaGenome with {n_variants} variants on device {devices}...")
    
    # 1. Load Model
    model = LightningModel(
        model_params={
            "model_type": "AlphaGenomeModel",
            "output_key": "rna_seq",
            "weights_path": WEIGHTS_PATH,
            "resolution": 128,
        },
        train_params={"task": "regression", "loss": "mse"},
    )
    model.data_params["train"] = {"seq_len": AG_INPUT_LEN, "bin_size": AG_BIN_SIZE}
    model.model_params["crop_len"] = 0
    
    # Create synthetic variants
    variants = pd.DataFrame({
        "chrom": ["chr21"] * n_variants,
        "pos": np.linspace(10_000_000, 40_000_000, n_variants).astype(int),
        "ref": ["A"] * n_variants,
        "alt": ["G"] * n_variants
    })
    
    # 2. Warmup
    print("Warming up...")
    _ = model.predict_on_seqs(torch.randn(1, 4, AG_INPUT_LEN), devices=devices)
    
    # 3. Benchmark standard (Auto-selected kernels)
    print("Running standard inference...")
    start = time.time()
    # Batch size 4 as per earlier test
    _ = model.predict_on_dataset(variants, genome="hg38", devices=devices, batch_size=4)
    end = time.time()
    standard_time = end - start
    print(f"Standard time: {standard_time:.2f}s ({n_variants/standard_time:.2f} vars/s)")

    # 4. Benchmark with FlashAttention backend forced (if possible via SDPA)
    # Note: Since AG doesn't use F.scaled_dot_product_attention yet, this 
    # might not show direct improvement unless we modify the code.
    # But we can test if the environment supports it.
    
    print("\nAttempting to force FlashAttention via context manager (testing compatibility)...")
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            start = time.time()
            _ = model.predict_on_dataset(variants, genome="hg38", devices=devices, batch_size=4)
            end = time.time()
            flash_time = end - start
            print(f"Forced Flash time: {flash_time:.2f}s ({n_variants/flash_time:.2f} vars/s)")
    except Exception as e:
        print(f"FlashAttention context failed or not applicable: {e}")

if __name__ == "__main__":
    benchmark_ag(n_variants=50, devices=[0])
