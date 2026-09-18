#!/usr/bin/env python3
"""Rebuild E1--E3 of the frozen Mdk motif--ISM plan from existing artifacts."""
from __future__ import annotations

import argparse, json, os, shutil, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

REPO = Path(__file__).resolve().parents[6]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts/ism/experiments/saijou_hsc"), str(REPO / "scripts/ism/experiments/saijou_hsc/experimental/original_multitrack_score")]
from grelu.io.motifs import get_jaspar
from grelu.interpret.motifs import scan_sequences
import profile_features
import score_calculations as frozen_score

PRIMARY = REPO / "experiments/ism/20260903_nine_gene_tss6kb_strict_shuffle"
OUT = REPO / "experiments/ism/20260909_mdk_motif_ism_v1_e1e6_r5"
FASTA = Path("/work/Database/Database_fromDocker/Referencedata_mm10/genome.fa")
BACKENDS = ("alphagenome_finetuned", "alphagenome_original", "borzoi_finetuned", "borzoi_original")
HYPOTHESES = {
 "sp_klf_gc": ("SP/KLF GC box", "sp_or_klf", "https://pmc.ncbi.nlm.nih.gov/articles/PMC4310735/", "human direct promoter evidence; mouse/HSC identity unproven"),
 "wt1": ("WT1", "wt1", "https://pubmed.ncbi.nlm.nih.gov/8950987/", "human relation; direction/site transfer not assumed"),
 "rar_rxr": ("RAR/RXR RARE", "rar_rxr_heterodimer", "https://www.jstage.jst.go.jp/article/biochemistry1922/117/4/117_4_845/_article/-char/en", "human retinoic-acid response; mouse instance unproven"),
 "hif": ("HIF1A/ARNT HRE", "hif_or_arnt", "https://pubmed.ncbi.nlm.nih.gov/15197188/", "non-HSC promoter evidence"),
 "nfkb": ("NF-kB/RELA", "nfkb_rel", "https://link.springer.com/article/10.1186/1755-8794-1-6", "TF-gene relation is not a Mdk site"),
 "ap1": ("AP-1", "fos_or_jun", "https://pmc.ncbi.nlm.nih.gov/articles/PMC3362071/", "HSC-relevant hypothesis, not direct Mdk site"),
 "smad": ("SMAD3/4", "smad", "https://pmc.ncbi.nlm.nih.gov/articles/PMC3362071/", "TGF-beta/HSC hypothesis, not direct Mdk site"),
 "tcf21": ("TCF21-associated E-box", "tcf21", "https://onlinelibrary.wiley.com/doi/full/10.1002/hep.30965", "association only; PWM hit is not binding evidence"),
}
FEATURE_COLUMNS = ['model_backend','model_id','gene','mutation_id','edit_start','edit_end','edit_center_position','variant_offset_from_tss_transcription_bp','ref_sequence','alt_sequence','replacement_replicate','track_id','track_group','track_task_name','track_cell_type','track_modality','track_strand','readout_id','readout_role','n_bins','ref_sum','alt_sum','absolute_delta_mean','log2fc_ratio_of_sums','ref_negative_bins','alt_negative_bins']

def parse_args():
 p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,default=OUT); p.add_argument('--fasta',type=Path,default=FASTA); p.add_argument('--reuse-e2',type=Path); return p.parse_args()
def write_tsv(p,d): d.to_csv(p,sep='\t',index=False)
def write_json(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def percentile(v,b):
 b=np.asarray(b,float)
 if not np.isfinite(v): raise ValueError('candidate statistic is non-finite')
 if b.size and not np.isfinite(b).all(): raise ValueError('background contains non-finite statistics')
 if b.size<100: return np.nan,'insufficient_background'
 return 100*((b<v-1e-12).sum()+.5*np.isclose(b,v,rtol=0,atol=1e-12).sum())/len(b),'ok'
def hsc_priority_pass(c,h,r): return bool(c>0 and h>95 and r>95)
def tx_to_genomic(a,b,tss=91932297): return tss-b+1,tss-a+1

def motif_matches_rule(mid,rule):
 factors=set(mid.upper().split('_',1)[-1].replace('::',' ').split())
 if rule=='rar_rxr_heterodimer': return any(x.startswith('RAR') for x in factors) and any(x.startswith('RXR') for x in factors)
 tokens={'sp_or_klf':('SP','KLF'),'wt1':('WT1',),'hif_or_arnt':('HIF','ARNT'),'nfkb_rel':('NFKB','RELA','RELB','REL'),'fos_or_jun':('FOS','JUN'),'smad':('SMAD',),'tcf21':('TCF21',)}[rule]
 return any(any(f.startswith(t) for t in tokens) for f in factors)

def load_manifest():
 m=pd.read_csv(PRIMARY/'prepared/mutation_manifest.tsv',sep='\t'); m=m[m.gene.eq('Mdk')].copy()
 if not (len(m)==8922 and m.mutation_id.is_unique and m.gene_strand.eq('-').all()): raise ValueError('Mdk manifest identity check failed')
 return m
def reference_tx(fasta):
 from grelu.interpret.ism.fasta import FastaReference
 with FastaReference(fasta) as f: g=f.extract('chr2',91929297,91935297).upper()
 return g.translate(str.maketrans('ACGT','TGCA'))[::-1]
def make_candidates(m,fasta):
 seq=reference_tx(fasta); motifs=get_jaspar(release='JASPAR2024',tax_group='vertebrates'); hs=[]; ins=[]
 for hid,(label,rule,url,limit) in HYPOTHESES.items():
  subset={k:v for k,v in motifs.items() if motif_matches_rule(k,rule)}
  if not subset: raise ValueError(f'no PWM matches {rule}')
  hs.append(dict(hypothesis_id=hid,label=label,pwm_rule=rule,source_url=url,evidence_limit=limit,evidence_level='hypothesis_or_nonmouse',expected_direction='unknown',status='registered_before_model_scores'))
  hits=scan_sequences(seq,subset,seq_ids=['Mdk_tx'],pthresh=1e-4,rc=True).rename(columns={'fimo_p-value':'fimo_pvalue'})
  if hits.empty:
   ins.append(dict(instance_id=f'{hid}_no_reference_hit',hypothesis_id=hid,category=label,pwm_id='',aliases='',tx_start=np.nan,tx_end=np.nan,genomic_start=np.nan,genomic_end=np.nan,pwm_strand='',fimo_pvalue=np.nan,selection_rank=np.nan,selected_for_scoring=False,prior_ism_selected=False,status='no_reference_hit')); continue
  hits=hits.sort_values(['fimo_pvalue','motif','start','end','strand'])
  for rank,r in enumerate(hits.itertuples(index=False),1):
   a,b=-2999+int(r.start),-2999+int(r.end); gs,ge=tx_to_genomic(a,b)
   ins.append(dict(instance_id=f'{hid}_{rank:04d}',hypothesis_id=hid,category=label,pwm_id=r.motif,aliases=r.motif,tx_start=a,tx_end=b,genomic_start=gs,genomic_end=ge,pwm_strand=r.strand,fimo_pvalue=r.fimo_pvalue,selection_rank=rank,selected_for_scoring=rank==1,prior_ism_selected=False,status='reference_hit'))
 ins += [dict(instance_id='core_promoter_tss',hypothesis_id='core_promoter',category='core promoter/start element',pwm_id='',aliases='',tx_start=-5,tx_end=5,genomic_start=91932293,genomic_end=91932303,pwm_strand='.',fimo_pvalue=np.nan,selection_rank=1,selected_for_scoring=True,prior_ism_selected=False,status='reference_defined'),dict(instance_id='splice_prior_2175_2189',hypothesis_id='splice_structure',category='splice acceptor proximity',pwm_id='',aliases='',tx_start=2175,tx_end=2189,genomic_start=91930109,genomic_end=91930123,pwm_strand='.',fimo_pvalue=np.nan,selection_rank=1,selected_for_scoring=True,prior_ism_selected=True,status='prior_ism_selected')]
 c=pd.DataFrame(ins); s=c[c.selected_for_scoring].copy()
 for i,r in s.iterrows(): s.loc[i,'covered_centers']=m[(m.edit_start<r.genomic_end)&(m.edit_end>r.genomic_start)].edit_center_position.nunique()
 return pd.DataFrame(hs),c.merge(s[['instance_id','covered_centers']],on='instance_id',how='left'),s

def load_features(b):
 p=PRIMARY/'runs'/b/'features/combined_mutation_features.parquet'; return ds.dataset(p,format='parquet').to_table(columns=FEATURE_COLUMNS,filter=ds.field('gene')=='Mdk').to_pandas()
def edit_and_centers():
 edits=[]; centers=[]
 for b in BACKENDS:
  x=load_features(b); x['pseudocount']=1.; x['signed_effect']=x.log2fc_ratio_of_sums; x['absolute_effect']=x.signed_effect.abs(); edits.append(x)
  keys=['model_backend','model_id','gene','track_id','track_group','track_modality','track_strand','readout_id','readout_role','edit_center_position','variant_offset_from_tss_transcription_bp']
  z=x.groupby(keys,dropna=False,sort=False).agg(median_absolute_effect=('absolute_effect','median'),median_signed_effect=('signed_effect','median'),replacement_mad=('signed_effect',lambda v:float(np.median(np.abs(v-np.median(v))))),replacements=('replacement_replicate','nunique'),ref_sum=('ref_sum','median'),alt_sum=('alt_sum','median'),n_bins=('n_bins','median'),negative_ref_bins=('ref_negative_bins','sum'),negative_alt_bins=('alt_negative_bins','sum')).reset_index(); centers.append(z)
 return pd.concat(edits,ignore_index=True),pd.concat(centers,ignore_index=True)
def link_or_copy(src,dst):
 try: os.link(src,dst)
 except OSError: shutil.copy2(src,dst)
def support(m,r,col): return sorted(m[(m.edit_start<r.genomic_end)&(m.edit_end>r.genomic_start)][col].unique())
def backgrounds(selected,m,col):
 centers=sorted(m[col].unique()); cset=set(centers); ss={r.instance_id:support(m,r,col) for r in selected.itertuples(index=False)}; excluded=set().union(*(set(x) for x in ss.values())); out={}
 for iid,s in ss.items():
  rel=[x-s[0] for x in s] if s else []; out[iid]=[[start+d for d in rel] for start in centers if rel and all(start+d in cset for d in rel) and not set(start+d for d in rel)&excluded]
 return out
def score_region(series,s,bgs):
 if not series.index.is_unique: raise ValueError('score index not unique')
 if set(s)-set(series.index): raise ValueError('candidate support missing from score series')
 stat=float(series.reindex(s).median()); bg=np.asarray([series.reindex(x).median() for x in bgs],float); pct,status=percentile(stat,bg); return stat,bg,pct,status
def bg_records(iid,backend,score_id,track,readout,basis,windows,stats):
 return [dict(instance_id=iid,background_id=f'{iid}_{score_id}_{j:04d}',backend=backend,score_id=score_id,track_id=track,readout_role=readout,coordinate_basis=basis,center_positions=';'.join(map(str,w)),geometry_relative_offsets=';'.join(str(x-w[0]) for x in w),raw_statistic=float(v),status='included',exclusion_reason='') for j,(w,v) in enumerate(zip(windows,stats),1)]

def ft_scores(selected,centers,m):
 rows=[]; br=[]; bgs=backgrounds(selected,m,'edit_center_position')
 for backend in ('alphagenome_finetuned','borzoi_finetuned'):
  z=centers[centers.model_backend.eq(backend)&centers.readout_role.eq('gene_body_output_clipped')]; piv=z.pivot(index='edit_center_position',columns='track_id',values='median_absolute_effect')
  if not {'hsc','mac','lsec','chol'}<=set(piv): raise ValueError(f'{backend} lacks four cells')
  values={'S_RNA':piv.hsc,'S_HSC':piv.hsc-piv[['mac','lsec','chol']].max(axis=1)}
  for r in selected.itertuples(index=False):
   s=support(m,r,'edit_center_position'); calc={}
   for sid,ser in values.items():
    stat,bg,pct,status=score_region(ser,s,bgs[r.instance_id]); calc[sid]=(stat,pct,status); track='hsc' if sid=='S_RNA' else 'hsc_vs_max_other'; br+=bg_records(r.instance_id,backend,sid,track,'gene_body_output_clipped','genomic_0_based_edit_center_position',bgs[r.instance_id],bg)
   rs,rp,rst=calc['S_RNA']; hc,hp,hst=calc['S_HSC']
   rows += [dict(instance_id=r.instance_id,backend=backend,score_id='S_RNA',track_id='hsc',readout_role='gene_body_output_clipped',raw_statistic=rs,percentile_score=rp,background_n=len(bgs[r.instance_id]),rank_pass=rp>95 if rst=='ok' else np.nan,status=rst,reason=''),dict(instance_id=r.instance_id,backend=backend,score_id='S_HSC',track_id='hsc_vs_max_other',readout_role='gene_body_output_clipped',raw_statistic=hc,percentile_score=hp,background_n=len(bgs[r.instance_id]),rank_pass=hsc_priority_pass(hc,hp,rp) if hst==rst=='ok' else np.nan,status=hst,reason='requires C>0, S_HSC>95, and HSC S_RNA>95')]
   for cell in ('hsc','mac','lsec','chol'): rows.append(dict(instance_id=r.instance_id,backend=backend,score_id='cell_descriptive',track_id=cell,readout_role='gene_body_output_clipped',raw_statistic=float(piv[cell].reindex(s).median()),percentile_score=np.nan,background_n=np.nan,rank_pass=np.nan,status='not_ranked',reason='descriptive_cell_effect'))
 return pd.DataFrame(rows),pd.DataFrame(br)

def original_scores(selected,m):
 rows=[]; views=[]; br=[]; bgs=backgrounds(selected,m,'variant_offset_from_tss_transcription_bp')
 for backend in ('alphagenome_original','borzoi_original'):
  x=load_features(backend); tracks=pd.read_csv(PRIMARY/'runs'/backend/'track_manifest.tsv',sep='\t'); out=x[x.readout_role.eq('tss_1024bp')&x.track_id.isin(tracks[tracks.modality.isin(['rna_seq','cage'])].track_id)].copy(); out['median_absolute_log2fc']=out.log2fc_ratio_of_sums.abs(); out['median_signed_log2fc']=out.log2fc_ratio_of_sums
  out=out.groupby(['gene','variant_offset_from_tss_transcription_bp','track_id','track_group'],sort=False).agg(replacements=('replacement_replicate','nunique'),median_absolute_log2fc=('median_absolute_log2fc','median'),median_signed_log2fc=('median_signed_log2fc','median')).reset_index().merge(tracks[['track_id','modality']],on='track_id',how='left'); out['score_view']='output'; out['readout_definition']='tss_1024bp_ratio_of_sums'
  local=profile_features.summarize_local_profile_effects(PRIMARY,gene='Mdk',run_name=backend,output_modalities={'rna_seq','cage'}); _,vc,combined=frozen_score.score_modality_views(pd.concat([out,local],ignore_index=True,sort=False),95.); ser=combined.set_index('variant_offset_from_tss_transcription_bp').max_view_score_raw
  for r in selected.itertuples(index=False):
   s=support(m,r,'variant_offset_from_tss_transcription_bp'); stat,bg,pct,status=score_region(ser,s,bgs[r.instance_id]); rows.append(dict(instance_id=r.instance_id,backend=backend,score_id='S_MULTI',track_id='combined_output_local',readout_role='tss_1024bp_and_local_bin_neighbors',raw_statistic=stat,percentile_score=pct,background_n=len(bg),rank_pass=pct>95 if status=='ok' else np.nan,status=status,reason='frozen_method3_center_statistic_with_tx_offset_background')); br+=bg_records(r.instance_id,backend,'S_MULTI','combined_output_local','tss_1024bp_and_local_bin_neighbors','transcript_tss_offset_bp',bgs[r.instance_id],bg)
   for view,part in vc.groupby('score_view',sort=False):
    idx=part.set_index('variant_offset_from_tss_transcription_bp'); vs,vbg,vp,vstatus=score_region(idx.spatial_support,s,bgs[r.instance_id]); readout='tss_1024bp' if view=='output' else 'edit_bin_plus_neighbors'; views.append(dict(instance_id=r.instance_id,backend=backend,score_view=view,readout_role=readout,covered_centers=len(s),track_groups_min=int(idx.track_groups.reindex(s).min()),raw_statistic=vs,percentile_score=vp,background_n=len(vbg),median_center_view_score=float(idx.view_score.reindex(s).median()),median_group_signed_log2fc=float(idx.median_group_signed_log2fc.reindex(s).median()),loss_direction_group_fraction=float(idx.loss_direction_group_fraction.reindex(s).median()),status=vstatus,reason='instance-level same-geometry view summary')); br+=bg_records(r.instance_id,backend,'S_MULTI_VIEW',view,readout,'transcript_tss_offset_bp',bgs[r.instance_id],vbg)
 return pd.DataFrame(rows),pd.DataFrame(views),pd.DataFrame(br)

def main():
 a=parse_args(); out=a.out.resolve()
 if out.exists(): raise SystemExit(f'refusing overwrite: {out}')
 out.mkdir(parents=True); m=load_manifest(); hs,cands,selected=make_candidates(m,a.fasta); write_tsv(out/'hypothesis_registry.tsv',hs); write_tsv(out/'candidate_instances.tsv',cands)
 write_json(out/'score_contract.json',{'version':'mdk-motif-ism-v1-r5','frozen_at_utc':datetime.now(timezone.utc).isoformat(),'pwm':'JASPAR2024 vertebrates; p<=1e-4','rar_rxr_rule':'factor identities include both RAR* and RXR*; PPAR/RXR excluded','candidate_selection':'reference sequence plus pre-registered hypothesis only; selected rank 1; no model response read','effect':'log2((alt_sum+n*1)/(ref_sum+n*1))','ft_readout':'gene_body_output_clipped','thresholds':{'rank':'strictly >95','background_minimum':100},'background':'same exact center-offset geometry; explicit coordinate basis; excludes all selected supports','hsc':'C>0 and S_HSC>95 and HSC S_RNA>95','original':'frozen Method 3 center statistic with transcript-TSS-offset background','motif_control':'same instance/center/changed-base count; loss delta>=5; preserved abs(delta)<1'})
 if a.reuse_e2:
  src=a.reuse_e2.resolve()
  for name in ('edit_effects.tsv','center_effects.tsv'): link_or_copy(src/name,out/name)
  centers=pd.read_csv(out/'center_effects.tsv',sep='\t'); note=f'reused independently checked E2 tables from {src}'
 else:
  edits,centers=edit_and_centers(); write_tsv(out/'edit_effects.tsv',edits); write_tsv(out/'center_effects.tsv',centers); note='re-exported from source parquet'
 fr,fb=ft_scores(selected,centers,m); orig,views,ob=original_scores(selected,m); write_tsv(out/'region_scores.tsv',pd.concat([fr,orig],ignore_index=True)); write_tsv(out/'original_view_scores.tsv',views); write_tsv(out/'background_windows.tsv',pd.concat([fb,ob],ignore_index=True)); write_json(out/'e1_e3_status.json',{'status':'ok','e2_source':note,'selected_instances':len(selected),'region_score_rows':len(fr)+len(orig),'background_rows':len(fb)+len(ob),'original_view_rows':len(views)}); print(out)
if __name__=='__main__': main()
