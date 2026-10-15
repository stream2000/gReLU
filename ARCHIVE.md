# gReLU-replication 归档说明

归档日期：2026-10-15。本仓库已停止开发，只用于查阅。

- 后续工作在 `/work2/Users/qijun/gReLU-clustering`，GitHub 分支 `dev`
  （`git@github.com:stream2000/gReLU.git`）。新仓库是从本仓库
  `alphagenome @ 8409dfe` clone 后精简出来的，只保留 ISM 核心，Git 历史连续。
- 本仓库的归档点是 tag `replication-final`，对应 `alphagenome` 分支上提交本文件的那个 commit。
- 已关闭的 Saijou HSC / WT1 课题有更细的索引：
  `docs/history/saijou_20261013/HISTORY_INDEX.md`。本文件不重复那里的内容。

## 这个仓库做过什么

| 时间 | 内容 |
| --- | --- |
| 2026-04 至 05 | fork Genentech gReLU，接入 AlphaGenome（PyTorch 移植，子模块 `src/alphagenome_pytorch`）；AlphaGenome 与 Borzoi 推理对比；eQTL AUROC 基准（`scripts/eqtl/`） |
| 2026-06 下旬 至 07 上旬 | 用 Saijou 小鼠肝脏 scRNA-seq 伪 bulk（HSC / Mac / LSEC / Chol 四个细胞类型）微调 Borzoi 与 AlphaGenome（head-only 与 LoRA），另有 NTv3 MVP；搭建 `grelu_dev` 环境；positive control 检查 |
| 2026-07 | 建立通用 saturation ISM 核心（`src/grelu/interpret/ism/`）；Mdk / Col1a1 / Acta2 及九基因 TSS 扫描；四模型（AG/Borzoi × original/fine-tuned）对比；positive control 审计 |
| 2026-08 至 09 | 九基因 TSS ±3 kb strict-shuffle 基线；Mdk motif ISM；1-bp 分辨率重打分；WT1 exact counterfactual |
| 2026-10-03 至 13 | HSC 与 AlphaGenome 训练 track 相似性原型；WT1 全 track 复跑与 TPM 过滤符号检验；课题关闭 |

最终结论的权威来源是各实验目录里的 `result_summary.md`（见下文）。其中最后两项：

- 九基因基线：`experiments/ism/saijou_nine_gene_tss_3kb_strict_shuffle_pdf_transcripts/result_summary.md`
- WT1：`experiments/ism/20260918_1349_wt1_exact_counterfactual/result_summary.md`
  （WT1 motif 身份不能解释观察到的响应。`wt1_0002` 保留为末端外显子 / 3′ 加工候选，机制未证实。）

## 东西在哪里

### 代码（Git 跟踪）

| 内容 | 位置 |
| --- | --- |
| ISM 核心（新仓库已带走） | `src/grelu/interpret/ism/` |
| 微调 checkpoint 的 ISM adapter（新仓库未带走） | `src/grelu/interpret/ism/model_adapters/{alphagenome,borzoi}_finetuned.py` |
| 微调训练脚本、LoRA 辅助函数、mmap 数据集 | `src/ft-scripts/`（`train_alphagenome.py`、`train_borzoi.py`、`train_ntv3.py`、`saijou_tasks.py`、`borzoi_mmap_dataset.py`） |
| Saijou ISM 流程（冻结） | `scripts/ism/experiments/saijou_hsc/`，入口顺序见 `WORKFLOW_INDEX.md`，分层约定见 `ANALYSIS_HARNESS.md` |
| 实验性分析（MDK motif、WT1、track 相似性、原始多 track 打分等） | `scripts/ism/experiments/saijou_hsc/experimental/` |
| eQTL / 推理 / 评估脚本 | `scripts/eqtl/`、`scripts/run_*.py` |
| 训练与运行手册 | `experiments/RUNBOOK.md`、`SETUP_GUIDE.md` |
| 框架设计说明 | `docs/saturation_ism_framework_design.md`、`docs/saijou_ism_engineering.md` |
| 计划与阶段报告（中文） | `docs/*_zh.md` |

### 只在本地 checkout 里的文件（被 `.gitignore` 忽略，不在 GitHub 上）

| 内容 | 位置 |
| --- | --- |
| 进度交接与滚动日志 | `experiments/Progress.md`；更早的在 `experiments/archive/progress/` |
| 早期探针与审计记录 | `experiments/*.md`、`experiments/*.py`、`experiments/*.tsv` |
| 被替换掉的旧实现 | `scripts/ism/experiments/saijou_hsc/archive/` |
| AlphaGenome 论文原文与补充材料 | `docs/papers/alphagenome/`（未跟踪） |
| 本地笔记 | `MEMORY.md`、`case_learning/`、`plan/`、`referrence/` |

### 数据与产物

| 内容 | 位置 | 大小 |
| --- | --- | --- |
| ISM 运行产物（75 个目录） | `/work3/Users/qijun/gReLU-history/saijou-20261013/ism/<name>/`（权威副本，2026-10-15 校验一致） | 175 GB |
| 同一批产物的原件 | `experiments/ism/<name>/`（`/work2`，尚未替换成符号链接） | 176 GB |
| 新旧路径对照 | `docs/history/saijou_20261013/artifact_locations.tsv` | |
| 微调 checkpoint | `runs/<run>/checkpoints/`。九基因基线用的是 AlphaGenome-FT epoch 19（`alphagenome_..._res1_..._lora_active_h512x1`、`..._res128_...`）和 Borzoi-FT epoch 39（`borzoi_split_chr10_chr11_poisson_multinomial_lora`） | 44 GB |
| 训练数据缓存与划分 | `dataset_store/`（`splits/`、`mmap_caches/` 等） | 4.2 GB |
| Saijou bigWig | `bigwig` → `/work2/Projects/Project_DL_pillar/Finetune_Borzoi_Saijou_scRNAseq/bigwig` | |
| 交付的 PDF 报告副本 | `/home/fqijun/report/` | |

`experiments/ism/20261003_hsc_ag_training_data_similarity/` 没有迁移，WT1 的 TPM 符号检验会读它的 `prepared/expr_cpm.parquet`。

## 分支与工作树

| 分支 | 说明 |
| --- | --- |
| `alphagenome` | 本仓库主线，归档点打了 tag `replication-final` |
| `main` | 跟随上游 Genentech gReLU |
| `origin/ism` | 2026-06 的早期 ISM / MREG 工作 |
| `origin/codex/share-saijou-lora-20260831` | 2026-08-31 共享 Saijou LoRA 冒烟交付 |
| `agent/*` | Raft agent 工作树，位于 `/work2/Users/qijun/grelu-wt/`。`builder-11` 的 `71a289b` 是收尾时的平行版本，内容已被 `a1e4d46` 和 `8409dfe` 覆盖 |

## 怎么查

```bash
cd /work2/Users/qijun/gReLU-replication
source activate.sh          # grelu_dev 环境，import 指向本 checkout 的 src/
git log --oneline -- <path> # 查某个文件的来历
```

- 只读复现：直接在本 checkout 跑，冻结脚本依赖 `experiments/ism/...` 相对路径和 `parents[n]` 定位仓库根，所以不要搬目录。
- 想在新仓库复用这里的代码：从本仓库 cherry-pick 或复制，并按新仓库 `AGENTS.md` 的分层规则进入 `experimental/` 或核心。
- 已知问题（resume 不校验来源、`compare_checkpoints.py` 缺少空输入保护、两套 motif family 标签）见 `HISTORY_INDEX.md` 的 Known debts。
