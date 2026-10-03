# track_similarity (experimental)

Which AlphaGenome mouse training samples (ENCODE RNA-seq, FANTOM5 CAGE) look most
like Saijou HSC / Mac / LSEC / Chol? Plan:
`docs/hsc_alphagenome_track_similarity_plan_20261003_zh.md` (v2). Fast/rough
prototype; not imported by canonical workflows.

Deviations from the plan:
- No single-cell count matrix is readable (`/work2/Users/saijou` is not
  accessible), so Saijou enters as the 4 pseudobulk CPM bigWigs summed over the
  union of exons per gene (protein-coding, mm10 GTF). No HSC_1/HSC_2 split.
- Bulk and Saijou both become multinomial pseudocells (30 per sample, 5,000
  UMI), so the pseudocell step is identical on both sides.
- FANTOM5: all 538 description groups (replicates pooled) are scored, and
  `ag_cage_name_hit` flags groups whose description contains an AlphaGenome CAGE
  biosample name (a substring match, not an ontology mapping).
- Rank intervals come from a pseudocell bootstrap (100x), not 20 redraws.

## Run (bagpipe)

```bash
O=experiments/ism/20261003_hsc_ag_training_data_similarity   # raw/ already downloaded
source activate.sh
python scripts/ism/experiments/saijou_hsc/experimental/track_similarity/prepare_inputs.py --out $O
python scripts/ism/experiments/saijou_hsc/experimental/track_similarity/make_pseudocells.py --out $O
# UCE: ~/tools/UCE (snap-stanford), weights from HF mirrors lza1/uce_hf + minwoosun/uce-misc
# (figshare blocks scripted downloads); venv ~/.venvs/uce layered on grelu_dev.
source ~/.venvs/uce/activate_uce.sh; cd ~/tools/UCE
for s in 0 1; do CUDA_VISIBLE_DEVICES=$s python eval_single_anndata.py --adata_path $O/uce_input/shard$s.h5ad \
  --dir $O/uce_output/ --species mouse --model_loc model_files/33l_8ep_1024t_1280.torch --nlayers 33 --batch_size 16 & done; wait
cd -; source activate.sh; PYTHONPATH=$HOME/.venvs/uce/lib/python3.12/site-packages \
  python scripts/ism/experiments/saijou_hsc/experimental/track_similarity/analyze.py --out $O
```

UCE on two 48 GB GPUs takes about 12 minutes. The whole pipeline takes about 40 minutes, including downloads.
