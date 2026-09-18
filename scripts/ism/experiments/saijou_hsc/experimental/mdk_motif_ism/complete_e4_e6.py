#!/usr/bin/env python3
"""Add E4, explicitly defer E5, and validate an Mdk motif--ISM package."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

REPO=Path(__file__).resolve().parents[6]; sys.path.insert(0,str(REPO/'src'))
from grelu.io.motifs import get_jaspar
from grelu.interpret.ism.fasta import FastaReference
PRIMARY=REPO/'experiments/ism/20260903_nine_gene_tss6kb_strict_shuffle'
OUT=REPO/'experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5'
FASTA=Path('/work/Database/Database_fromDocker/Referencedata_mm10/genome.fa')
BACKENDS=('alphagenome_finetuned','alphagenome_original','borzoi_finetuned','borzoi_original')
BASE={'A':0,'C':1,'G':2,'T':3}

def parse_args():
 p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,default=OUT); p.add_argument('--fasta',type=Path,default=FASTA); return p.parse_args()
def pwm_score(seq,pwm): return float(sum(np.log2((pwm[BASE[b],i]+1e-4)/.25) for i,b in enumerate(seq)))
def changed_bases(ref,alt): return sum(a!=b for a,b in zip(ref,alt))
def build_pair_definitions(evidence):
 rows=[]
 for (iid,center,n),g in evidence[evidence.loss_class.isin(['loss','preserved'])].groupby(['instance_id','edit_center_position','actual_changed_bases'],sort=False):
  losses=g[g.loss_class.eq('loss')]; preserved=g[g.loss_class.eq('preserved')]
  for left in losses.itertuples(index=False):
   for right in preserved.itertuples(index=False): rows.append(dict(pair_id=f'{iid}_{center}_{left.mutation_id}_{right.mutation_id}',instance_id=iid,edit_center_position=int(center),actual_changed_bases=int(n),loss_mutation_id=left.mutation_id,preserved_mutation_id=right.mutation_id,loss_pwm_delta=float(left.continuous_delta),preserved_pwm_delta=float(right.continuous_delta)))
 return pd.DataFrame(rows)
def load_pair_effects(mutation_ids):
 parts=[]
 cols=['model_backend','mutation_id','track_id','readout_id','readout_role','log2fc_ratio_of_sums']
 for backend in BACKENDS:
  path=PRIMARY/'runs'/backend/'features/combined_mutation_features.parquet'; filt=(ds.field('gene')=='Mdk') & ds.field('mutation_id').isin(list(mutation_ids)); parts.append(ds.dataset(path,format='parquet').to_table(columns=cols,filter=filt).to_pandas())
 return pd.concat(parts,ignore_index=True)
def expand_pairs(defs,effects):
 if defs.empty: return pd.DataFrame(columns=['pair_id','instance_id','backend','track_id','readout_id','readout_role','edit_center_position','actual_changed_bases','loss_mutation_id','preserved_mutation_id','loss_pwm_delta','preserved_pwm_delta','loss_signed_effect','preserved_signed_effect','loss_absolute_effect','preserved_absolute_effect','absolute_effect_difference','signed_effect_difference','status','reason'])
 left=effects.rename(columns={'model_backend':'backend','mutation_id':'loss_mutation_id','log2fc_ratio_of_sums':'loss_signed_effect'}); right=effects.rename(columns={'model_backend':'backend','mutation_id':'preserved_mutation_id','log2fc_ratio_of_sums':'preserved_signed_effect'})
 x=defs.merge(left,on='loss_mutation_id',validate='one_to_many').merge(right,on=['preserved_mutation_id','backend','track_id','readout_id','readout_role'],validate='many_to_one'); x['loss_absolute_effect']=x.loss_signed_effect.abs(); x['preserved_absolute_effect']=x.preserved_signed_effect.abs(); x['absolute_effect_difference']=x.loss_absolute_effect-x.preserved_absolute_effect; x['signed_effect_difference']=x.loss_signed_effect-x.preserved_signed_effect; x['status']='matched_same_center_same_changed_bases'; x['reason']='existing edits only; descriptive paired contrast, not independent replicates'
 return x[['pair_id','instance_id','backend','track_id','readout_id','readout_role','edit_center_position','actual_changed_bases','loss_mutation_id','preserved_mutation_id','loss_pwm_delta','preserved_pwm_delta','loss_signed_effect','preserved_signed_effect','loss_absolute_effect','preserved_absolute_effect','absolute_effect_difference','signed_effect_difference','status','reason']]
def response_summary(pairs):
 if pairs.empty: return pd.DataFrame(columns=['instance_id','backend','track_id','readout_id','readout_role','matched_pairs','independent_centers','median_loss_absolute_effect','median_preserved_absolute_effect','median_absolute_effect_difference','median_signed_effect_difference','status','limitation'])
 out=pairs.groupby(['instance_id','backend','track_id','readout_id','readout_role'],sort=False).agg(matched_pairs=('pair_id','nunique'),independent_centers=('edit_center_position','nunique'),median_loss_absolute_effect=('loss_absolute_effect','median'),median_preserved_absolute_effect=('preserved_absolute_effect','median'),median_absolute_effect_difference=('absolute_effect_difference','median'),median_signed_effect_difference=('signed_effect_difference','median')).reset_index(); out['status']='descriptive_matched_control_available'; out['limitation']='overlapping edit windows and shared sequence background; no p-value/FDR or causal attribution'; return out

TABLE_KEYS={
 'hypothesis_registry.tsv':['hypothesis_id'],'candidate_instances.tsv':['instance_id'],'edit_effects.tsv':['model_backend','mutation_id','track_id','readout_id'],'center_effects.tsv':['model_backend','edit_center_position','track_id','readout_id'],'background_windows.tsv':['instance_id','background_id','backend','score_id','track_id','readout_role'],'region_scores.tsv':['instance_id','backend','score_id','track_id','readout_role'],'original_view_scores.tsv':['instance_id','backend','score_view'],'motif_edit_evidence.tsv':['instance_id','pwm_id','mutation_id'],'motif_control_pairs.tsv':['pair_id','backend','track_id','readout_id'],'motif_response_evidence.tsv':['instance_id','backend','track_id','readout_id'],'snv_bridge.tsv':['instance_id','backend'],'blocked_or_deferred.tsv':['item_id']}
def dtype_name(dtype):
 if pd.api.types.is_bool_dtype(dtype): return 'boolean'
 if pd.api.types.is_integer_dtype(dtype): return 'integer'
 if pd.api.types.is_float_dtype(dtype): return 'number'
 return 'string'
def unit_for(name):
 if name in {'edit_start','edit_end','edit_center_position','genomic_start','genomic_end'}: return 'mm10 genomic bp, 0-based half-open where interval'
 if name.startswith('tx_') or 'tss_transcription_bp' in name: return 'transcript-direction bp from TSS'
 if 'percentile' in name: return 'scan-background percentile 0-100'
 if 'effect' in name or 'log2fc' in name: return 'log2 fold-change unless named absolute delta'
 if name.endswith('_n') or name in {'matched_pairs','independent_centers','covered_centers'}: return 'count'
 return 'identifier/text/dimensionless as named'
def schema_for(out):
 tables={}
 for name,key in TABLE_KEYS.items():
  path=out/name; frame=pd.read_csv(path,sep='\t',nrows=5000); tables[name]={'primary_key':key,'columns':{col:{'type':dtype_name(frame[col].dtype),'unit':unit_for(col),'nullable':bool(frame[col].isna().any()),'null_rule':'allowed only when status/reason marks not applicable, not ranked, or deferred; never encode missing numeric values as zero'} for col in frame.columns}}
 return {'version':'mdk-motif-ism-v1-r5','tables':tables,'null_convention':'Empty TSV fields represent unavailable/not-applicable values and require an explicit row status/reason. Numeric missing values are never zero-filled.'}
def validate(out):
 region=pd.read_csv(out/'region_scores.tsv',sep='\t'); bg=pd.read_csv(out/'background_windows.tsv',sep='\t'); views=pd.read_csv(out/'original_view_scores.tsv',sep='\t'); cand=pd.read_csv(out/'candidate_instances.tsv',sep='\t'); pairs=pd.read_csv(out/'motif_control_pairs.tsv',sep='\t'); response=pd.read_csv(out/'motif_response_evidence.tsv',sep='\t'); snv=pd.read_csv(out/'snv_bridge.tsv',sep='\t'); blocked=pd.read_csv(out/'blocked_or_deferred.tsv',sep='\t')
 checks={}
 def add(name,passed,observed,expected): checks[name]={'passed':bool(passed),'observed':observed,'expected':expected}
 for name,key in TABLE_KEYS.items():
  f=pd.read_csv(out/name,sep='\t',usecols=key); add(f'{name}_primary_key_unique',not f.duplicated(key).any(),int(f.duplicated(key).sum()),0)
 ok=region.status.eq('ok'); add('finite_ranked_region_statistics',np.isfinite(region.loc[ok,['raw_statistic','percentile_score','background_n']].to_numpy(float)).all(),int(ok.sum()),'all ranked cells finite')
 add('finite_background_statistics',np.isfinite(bg.raw_statistic.to_numpy(float)).all(),int((~np.isfinite(bg.raw_statistic.to_numpy(float))).sum()),0)
 originals=region[region.score_id.eq('S_MULTI')]; add('original_percentiles_recalculated',len(originals)==16 and originals.percentile_score.nunique()>1 and not originals.percentile_score.eq(0).all(),originals.percentile_score.round(6).tolist(),'16 finite, non-degenerate percentiles')
 rar=cand[cand.instance_id.eq('rar_rxr_0001')].pwm_id.astype(str); valid_rar=len(rar)==1 and 'RAR' in rar.iloc[0].upper() and 'RXR' in rar.iloc[0].upper() and 'PPAR' not in rar.iloc[0].upper(); add('rar_rxr_pwm_identity',valid_rar,rar.tolist(),'RAR*/RXR* heterodimer, not PPAR/RXR')
 h=region[region.score_id.eq('S_HSC')].copy(); rna=region[region.score_id.eq('S_RNA')][['instance_id','backend','percentile_score']].rename(columns={'percentile_score':'rna_percentile'}); h=h.merge(rna,on=['instance_id','backend'],validate='one_to_one'); expected=(h.raw_statistic>0)&(h.percentile_score>95)&(h.rna_percentile>95); add('hsc_joint_rank_contract',h.rank_pass.astype(bool).eq(expected).all(),int(h.rank_pass.astype(bool).ne(expected).sum()),0)
 add('matched_preserving_controls_searched',len(pairs)>0 and pairs.status.eq('matched_same_center_same_changed_bases').all(),len(pairs),'>0 actual matched pair rows'); add('motif_response_summary_present',len(response)>0,len(response),'>0 rows'); add('e5_truthfully_deferred',snv.status.eq('deferred_not_run').all(),sorted(snv.status.unique()),['deferred_not_run']); add('e4_not_blocked',not blocked.item_id.str.startswith('E4').any(),blocked.item_id.tolist(),'no E4 blocker')
 failures=[k for k,v in checks.items() if not v['passed']]; return checks,failures

def main():
 a=parse_args(); out=a.out.resolve(); candidates=pd.read_csv(out/'candidate_instances.tsv',sep='\t'); selected=candidates[candidates.selected_for_scoring.fillna(False)].copy(); manifest=pd.read_csv(PRIMARY/'prepared/mutation_manifest.tsv',sep='\t'); manifest=manifest[manifest.gene.eq('Mdk')]; motifs=get_jaspar(release='JASPAR2024',tax_group='vertebrates'); rows=[]
 with FastaReference(a.fasta) as fa:
  for r in selected.itertuples(index=False):
   sub=manifest[(manifest.edit_start<r.genomic_end)&(manifest.edit_end>r.genomic_start)]
   if not r.pwm_id or r.pwm_id not in motifs:
    for x in sub.itertuples(index=False): rows.append(dict(instance_id=r.instance_id,pwm_id=str(r.pwm_id) if pd.notna(r.pwm_id) else '',mutation_id=x.mutation_id,edit_center_position=x.edit_center_position,replacement_replicate=x.replacement_replicate,continuous_ref_score=np.nan,continuous_alt_score=np.nan,continuous_delta=np.nan,loss_class='not_applicable_structural',actual_changed_bases=changed_bases(x.ref_sequence,x.alt_sequence),overlap_note=r.category,gain_status='not_evaluated_structural'))
    continue
   gs,ge=int(r.genomic_start),int(r.genomic_end); ref=fa.extract('chr2',gs,ge).upper(); pwm=motifs[r.pwm_id]
   if r.pwm_strand=='+': pwm=pwm[[3,2,1,0],::-1]
   for x in sub.itertuples(index=False):
    alt=list(ref)
    for pos in range(max(int(x.edit_start),gs),min(int(x.edit_end),ge)): alt[pos-gs]=x.alt_sequence[pos-int(x.edit_start)]
    rs,al=pwm_score(ref,pwm),pwm_score(''.join(alt),pwm); delta=rs-al; rows.append(dict(instance_id=r.instance_id,pwm_id=r.pwm_id,mutation_id=x.mutation_id,edit_center_position=x.edit_center_position,replacement_replicate=x.replacement_replicate,continuous_ref_score=rs,continuous_alt_score=al,continuous_delta=delta,loss_class='loss' if delta>=5 else ('preserved' if abs(delta)<1 else 'ambiguous'),actual_changed_bases=changed_bases(x.ref_sequence,x.alt_sequence),overlap_note='fixed_reference_instance',gain_status='not_evaluated_other_motifs'))
 evidence=pd.DataFrame(rows); evidence.to_csv(out/'motif_edit_evidence.tsv',sep='\t',index=False); defs=build_pair_definitions(evidence); effects=load_pair_effects(set(defs.loss_mutation_id)|set(defs.preserved_mutation_id)); pairs=expand_pairs(defs,effects); pairs.to_csv(out/'motif_control_pairs.tsv',sep='\t',index=False); response_summary(pairs).to_csv(out/'motif_response_evidence.tsv',sep='\t',index=False)
 pd.DataFrame([dict(instance_id=r.instance_id,backend=b,status='deferred_not_run',reason='optional E5 per-instance SNV coverage/readout/metadata audit was not executed; no comparability conclusion claimed') for r in selected.itertuples(index=False) for b in ('alphagenome','borzoi')]).to_csv(out/'snv_bridge.tsv',sep='\t',index=False); pd.DataFrame([dict(item_id='E5_snv_bridge',status='deferred',reason='optional bridge not run; requires explicit per-instance old-SNV coverage, checkpoint, readout, and 23-bp TES audit')]).to_csv(out/'blocked_or_deferred.tsv',sep='\t',index=False)
 (out/'schema.json').write_text(json.dumps(schema_for(out),indent=2,sort_keys=True)+'\n'); checks,failures=validate(out); summary={'engineering_status':'ok' if not failures else 'validation_failed','analysis_ready_scope':'E1-E4 scores and matched-control evidence; E5 explicitly deferred; no biological conclusion','checks':checks,'failures':failures,'warnings':['Motif-control contrasts remain model-internal and share overlapping edit windows.','Nearby gain/other-motif confounding was not evaluated.','All region ranks are scan-background ranks, not p-values/FDR.']}; (out/'validation_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); (out/'execution_handoff.md').write_text('# Mdk motif-ISM execution handoff r5\n\nExisting artifacts only; no inference or new mutations. E1-E4 were rebuilt and validated. E5 is explicitly deferred, not declared incomparable. Read `validation_summary.json` and `schema.json` before biological analysis.\n')
 if failures: raise SystemExit('validation failed: '+', '.join(failures))
 print(f'validated {out}')
if __name__=='__main__': main()
