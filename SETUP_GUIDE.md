# AlphaGenome Inference — 从零开始配置指南

> 适用于：在同一集群上想跑 AlphaGenome 推理的组员。  
> 前置条件：有集群账号、可访问 `/work/miniconda3`、有 GPU 节点。

---

## 1. 获取代码

```bash
git clone --recurse-submodules https://github.com/<your-org>/gReLU.git
cd gReLU
```

> `--recurse-submodules` 会同时拉取 `src/alphagenome_pytorch` 子模块。  
> 如果忘了加，事后补：`git submodule update --init --recursive`

---

## 2. 创建 Conda 环境

```bash
source /work/miniconda3/etc/profile.d/conda.sh

# 创建隔离环境（Python 3.12，名字随意）
conda create -n grelu_dev python=3.12 -y
conda activate grelu_dev
```

---

## 3. 安装依赖

```bash
# 安装 AlphaGenome PyTorch 子模块（editable）
pip install -e ./src/alphagenome_pytorch

# 安装 gReLU 主包（editable）
pip install -e .

# 安装推理所需的额外依赖
pip install huggingface_hub pyfaidx pandas pyranges
```

---

## 4. 下载模型权重

模型权重托管在 HuggingFace，约 **6 GB**。

**方式 A（推荐）：让 demo 脚本自动下载**

第一次运行 demo 时会自动下载到 `~/.cache/alphagenome/`，无需额外操作。

**方式 B：手动下载**

```bash
python - <<'EOF'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="gtca/alphagenome_pytorch",
    filename="model_fold_0.safetensors",
    local_dir="~/.cache/alphagenome",
)
print("Done")
EOF
```

**方式 C：直接拷贝（集群内部，最快）**

```bash
# 从 fqijun 的缓存拷贝（需要对方同意/开放权限）
cp /home/fqijun/.cache/huggingface/hub/models--gtca--alphagenome_pytorch/snapshots/*/model_fold_0.safetensors \
   ~/.cache/alphagenome/model_fold_0.safetensors
```

---

## 5. 运行 Demo

```bash
conda activate grelu_dev
cd gReLU

# 用随机序列快速验证环境（~2 分钟，含权重下载）
python demo_inference.py --random

# 查看 CAGE 所有可用 track 的元数据
python demo_inference.py --list_tracks --output_type cage

# 提取特定 track（CAGE track 0，128bp 分辨率）
python demo_inference.py --random --output_type cage --track_index 0

# 用 FASTA + 基因组坐标（需要 hg38.fa）
python demo_inference.py \
    --fasta /path/to/hg38.fa \
    --region chr1:1000000-2048576 \
    --output_type cage \
    --track_index 0
```

### 可用的输出类型（--output_type）

| 参数值        | 说明                   | 分辨率      | Track 数 |
|--------------|----------------------|------------|---------|
| `atac`       | 染色质可及性 (ATAC-seq)  | 1bp, 128bp | 256     |
| `dnase`      | DNase 超敏感位点         | 1bp, 128bp | 384     |
| `cage`       | 转录起始位点 (CAGE)      | 1bp, 128bp | 640     |
| `rna_seq`    | RNA 表达量              | 1bp, 128bp | 768     |
| `procap`     | PRO-cap               | 1bp, 128bp | 128     |
| `chip_tf`    | 转录因子 ChIP-seq        | 128bp 仅   | 1664    |
| `chip_histone` | 组蛋白修饰 ChIP-seq    | 128bp 仅   | 1152    |

---

## 6. 在自己的脚本中调用

```python
import numpy as np
import torch
from alphagenome_pytorch import AlphaGenome

SEQUENCE_LENGTH = 1_048_576  # 固定 1M bp

# 准备输入：one-hot 编码 (1, L, 4)
np.random.seed(42)
seq_idx = np.random.randint(0, 4, size=(1, SEQUENCE_LENGTH))
dna = torch.from_numpy(np.eye(4, dtype=np.float32)[seq_idx])  # (1, L, 4)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlphaGenome.from_pretrained(
    "~/.cache/alphagenome/model_fold_0.safetensors",
    device=device,
)
model.eval()

# 推理
organism = torch.tensor([0], dtype=torch.long, device=device)  # 0=human, 1=mouse
with torch.no_grad():
    outputs = model(dna.to(device), organism)

# 取 CAGE 在 128bp 分辨率下的第 0 个 track
cage_128 = outputs["cage"][128][0, :, 0].cpu().numpy()  # shape: (8192,)
print(f"CAGE track 0 (128bp): shape={cage_128.shape}, max={cage_128.max():.4f}")
```

---

## 常见问题

**Q: CUDA out of memory**  
A: 模型约需 16GB 显存。申请有足够显存的 GPU 节点（如 A100 40G）。

**Q: `ModuleNotFoundError: No module named 'alphagenome_pytorch'`**  
A: 确认 conda 环境已激活且 `pip install -e ./src/alphagenome_pytorch` 已执行。

**Q: 权重文件下载太慢**  
A: 用方式 C 从 fqijun 的缓存目录直接拷贝。

**Q: 输入序列长度不对**  
A: AlphaGenome 要求精确 1,048,576 bp (= 2^20)。用 `--fasta` 模式时，`end - start` 必须等于这个值。
