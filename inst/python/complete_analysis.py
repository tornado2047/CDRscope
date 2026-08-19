#!/usr/bin/env python3
"""
CDRscope Complete Analysis Module
==================================
Single entry point for all deep analysis:
  --significance:  Domain-level RA vs Control significance (Fisher exact, OR)
  --breakthrough:  Expansion gradient, UMAP axis decode, disease scoring, sequence network
  --validation:    Frequency redistribution, citrullination axis, HLA stratification

Usage:
  python3 complete_analysis.py --output-dir DIR --ref-dir DIR [--coords-csv CSV]
                               [--significance] [--breakthrough] [--validation]
"""
import os, sys, json, argparse, re, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from scipy.stats import mannwhitneyu, fisher_exact, chi2_contingency, spearmanr
from sklearn.metrics import roc_auc_score
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec

cjk_fonts = ['Noto Sans CJK SC', 'PingFang SC', 'STHeiti', 'Heiti SC', 'Arial Unicode MS']
for f in cjk_fonts:
    matches = [m for m in fm.fontManager.ttflist if f in m.name]
    if matches:
        plt.rcParams['font.family'] = f
        break
plt.rcParams['axes.unicode_minus'] = False

C_NAVY='#1B3A5C'; C_STEEL='#3D6A8C'; C_ACCENT='#D97742'; C_GREEN='#5B9279'; C_GRAY='#9A9A9A'
KD = {'I':4.5,'V':4.2,'L':3.8,'F':2.8,'C':2.5,'M':1.9,'A':1.8,'G':-0.4,
      'T':-0.7,'S':-0.8,'W':-0.9,'Y':-1.3,'P':-1.6,'H':-3.2,'E':-3.5,
      'Q':-3.5,'D':-3.5,'N':-3.5,'K':-3.9,'R':-4.5,'X':0,'U':0,'Z':0}
CHARGE = {'K':+1,'R':+1,'H':+0.5,'D':-1,'E':-1,'X':0,'U':0,'Z':0}
AROMATIC = set('FWY')

def net_charge(s): return sum(CHARGE.get(a,0) for a in s)
def hydro(s): return np.mean([KD.get(a,0) for a in s]) if s else 0
def arom(s): return sum(1 for a in s if a in AROMATIC)/len(s) if s else 0

def load_reference(ref_dir):
    meta_path = os.path.join(ref_dir, "ref_metadata.csv")
    if not os.path.exists(meta_path):
        alt = os.path.join(ref_dir, "ref_metadata.csv.gz")
        if os.path.exists(alt):
            meta_path = alt
        else:
            alt2 = os.path.join(ref_dir, "refmap_metadata_augmented.csv")
            if os.path.exists(alt2): meta_path = alt2
    meta = pd.read_csv(meta_path)
    return meta

def run_significance(meta, out_dir):
    """Domain-level significance analysis."""
    results = []
    domains = {
        'CDR3 Length': lambda r: 'Short' if r['length']<=12 else 'Medium' if r['length']<=16 else 'Long',
        'Net Charge': lambda r: 'Negative' if r['net_charge']<=-1 else 'Neutral' if r['net_charge']<=1 else 'Positive',
        'Hydrophobicity': lambda r: 'Hydrophilic' if r['hydrophobicity']<-0.5 else 'Neutral' if r['hydrophobicity']<0.3 else 'Hydrophobic',
        'Aromatic': lambda r: 'Low' if r['aromatic_frac']<0.10 else 'Medium' if r['aromatic_frac']<0.20 else 'High',
    }

    ra = meta[meta['group']=='RA-specific']
    ctrl = meta[meta['group']=='Control-specific']
    total_ra, total_ctrl = len(ra), len(ctrl)

    for domain, classifier in domains.items():
        cats = sorted(meta.apply(classifier, axis=1).unique())
        for cat in cats:
            mask = meta.apply(classifier, axis=1) == cat
            n_ra = len(meta[mask & (meta['group']=='RA-specific')])
            n_ctrl = len(meta[mask & (meta['group']=='Control-specific')])
            table = np.array([[n_ra, n_ctrl], [total_ra-n_ra, total_ctrl-n_ctrl]])
            or_val, p_val = fisher_exact(table, alternative='two-sided')
            results.append({
                'domain': domain, 'category': cat,
                'n_control': n_ctrl, 'n_ra': n_ra,
                'odds_ratio': round(or_val, 4), 'p_value': p_val,
                'significant': '***' if p_val<0.001 else '**' if p_val<0.01 else '*' if p_val<0.05 else 'ns'
            })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "domain_significance.csv"), index=False)
    print(f"  Significance: {len(df)} tests, {(df['p_value']<0.05).sum()} significant")
    return df

def run_breakthrough(meta, out_dir):
    """Breakthrough analyses: expansion, axis decode, disease scoring, network."""
    summary = {}

    # 1. Expansion gradient
    if 'duplicate_count' in meta.columns or 'dup_count' in meta.columns:
        dc_col = 'duplicate_count' if 'duplicate_count' in meta.columns else 'dup_count'
        ra_dc = np.log1p(meta[meta['group']=='RA-specific'][dc_col])
        ctrl_dc = np.log1p(meta[meta['group']=='Control-specific'][dc_col])
        summary['expansion'] = {
            'ra_mean': round(float(ra_dc.mean()), 3),
            'ctrl_mean': round(float(ctrl_dc.mean()), 3),
            'p_value': float(mannwhitneyu(ra_dc, ctrl_dc).pvalue)
        }

    # 2. UMAP axis decoding
    for prop in ['net_charge', 'hydrophobicity', 'aromatic_frac', 'length']:
        if prop in meta.columns:
            rho1, p1 = spearmanr(meta['umap1'], meta[prop])
            rho2, p2 = spearmanr(meta['umap2'], meta[prop])
            summary[f'umap_axis_{prop}'] = {
                'umap1_rho': round(float(rho1), 3), 'umap1_p': float(p1),
                'umap2_rho': round(float(rho2), 3), 'umap2_p': float(p2)
            }

    # 3. Disease scoring (sequence-level)
    if 'umap1' in meta.columns and 'umap2' in meta.columns:
        y = (meta['group'] == 'RA-specific').astype(int)
        auc1 = roc_auc_score(y, meta['umap1']) if y.nunique() > 1 else 0.5
        auc2 = roc_auc_score(y, meta['umap2']) if y.nunique() > 1 else 0.5
        summary['disease_scoring'] = {
            'umap1_auc': round(float(auc1), 4),
            'umap2_auc': round(float(auc2), 4),
            'conclusion': 'sequence-level AUC ≈ 0.5 → disease is population-level' if max(auc1, auc2) < 0.55 else 'some sequence-level signal exists'
        }

    # 4. Centroid shift
    if 'umap2' in meta.columns:
        ra_u2 = meta[meta['group']=='RA-specific']['umap2']
        ctrl_u2 = meta[meta['group']=='Control-specific']['umap2']
        shift = abs(ra_u2.mean() - ctrl_u2.mean())
        overlap = np.minimum(
            np.interp(ctrl_u2, np.sort(ra_u2), np.linspace(0, 1, len(ra_u2))),
            1 - np.interp(ctrl_u2, np.sort(ra_u2), np.linspace(0, 1, len(ra_u2)))
        ).mean()
        summary['centroid_shift'] = {
            'umap2_shift': round(float(shift), 4),
            'spatial_overlap': round(float(overlap), 4),
            'mwu_p': float(mannwhitneyu(ra_u2, ctrl_u2).pvalue)
        }

    with open(os.path.join(out_dir, "breakthrough_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Breakthrough: {len(summary)} metrics computed")
    return summary

def run_validation(meta, out_dir):
    """Biological validation: frequency redistribution, citrullination, HLA."""
    summary = {}

    # 1. Frequency redistribution (per-domain RA/Control ratio)
    ra = meta[meta['group']=='RA-specific']
    ctrl = meta[meta['group']=='Control-specific']
    total_ra, total_ctrl = len(ra), len(ctrl)

    domain_freq = {}
    for prop, cats in [('hydrophobicity', ['Hydrophobic','Neutral','Hydrophilic']),
                       ('length', ['Short','Medium','Long']),
                       ('net_charge', ['Negative','Neutral','Positive'])]:
        if prop not in meta.columns: continue
        for cat in cats:
            mask = meta[prop] == cat
            n_ra = len(meta[mask & (meta['group']=='RA-specific')])
            n_ctrl = len(meta[mask & (meta['group']=='Control-specific')])
            ra_freq = n_ra / total_ra if total_ra > 0 else 0
            ctrl_freq = n_ctrl / total_ctrl if total_ctrl > 0 else 0
            ratio = ra_freq / (ctrl_freq + 1e-6)
            domain_freq[f'{prop}_{cat}'] = {
                'ra_freq': round(float(ra_freq), 4),
                'ctrl_freq': round(float(ctrl_freq), 4),
                'ratio': round(float(ratio), 3)
            }
    summary['frequency_redistribution'] = domain_freq

    # 2. Citrullination axis (chemical complementarity)
    cit_KD = 0.5; R_KD = -4.5
    cit_charge = 0; R_charge = +1

    autoantigens = {
        'Vimentin': {'native': 'GVYATRSSAVRLRSS', 'cit': 'GVYATXSSAVXLSS'},
        'alpha-Enolase': {'native': 'KIHALEEELGEEYSVK', 'cit': 'KIHAXIEALLYEGK'},
        'Fibrinogen': {'native': 'GSEDTGEGDFRAEMK', 'cit': 'GSEDTGEGDFXAEMK'},
    }

    cit_results = {}
    for name, seqs in autoantigens.items():
        n_seq = seqs['native']; c_seq = seqs['cit']
        n_charge = sum(R_charge if a=='R' else CHARGE.get(a,0) for a in n_seq)
        c_charge = sum(cit_charge if a=='X' else CHARGE.get(a,0) for a in c_seq)
        n_hydro = np.mean([R_KD if a=='R' else KD.get(a,0) for a in n_seq])
        c_hydro = np.mean([cit_KD if a=='X' else KD.get(a,0) for a in c_seq])
        cit_results[name] = {
            'native_charge': float(n_charge), 'cit_charge': float(c_charge),
            'native_hydro': round(float(n_hydro), 3), 'cit_hydro': round(float(c_hydro), 3),
            'delta_charge': float(c_charge - n_charge),
            'delta_hydro': round(float(c_hydro - n_hydro), 3)
        }
    summary['citrullination_axis'] = cit_results

    # 3. HLA stratification (V gene proxy + QDFA motif)
    if 'v_family' in meta.columns:
        ra_vf = ra['v_family'].value_counts()
        ctrl_vf = ctrl['v_family'].value_counts()
        v_genes = {}
        for vf in ['TRAV20', 'TRBV25', 'TRAV7', 'TRBV20']:
            ra_f = ra_vf.get(vf, 0) / total_ra
            ctrl_f = ctrl_vf.get(vf, 0) / total_ctrl
            ratio = ra_f / (ctrl_f + 1e-6)
            v_genes[vf] = {
                'ra_freq': round(float(ra_f), 5), 'ctrl_freq': round(float(ctrl_f), 5),
                'log2_ratio': round(float(np.log2(ratio)), 3)
            }
        summary['hla_stratification'] = {'v_gene_proxy': v_genes}

    # QDFA motif
    if 'sequence' in meta.columns:
        qdfa_mask = meta['sequence'].str.contains('QDFA', na=False)
        n_ra_qdfa = len(meta[qdfa_mask & (meta['group']=='RA-specific')])
        n_ctrl_qdfa = len(meta[qdfa_mask & (meta['group']=='Control-specific')])
        summary['hla_stratification']['qdfa_motif'] = {
            'n_ra': int(n_ra_qdfa), 'n_ctrl': int(n_ctrl_qdfa),
            'ra_ctrl_ratio': round(float(n_ra_qdfa / max(n_ctrl_qdfa, 1)), 2),
            'expected_hla_ratio': 2.55
        }

    with open(os.path.join(out_dir, "validation_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Validation: {len(summary)} modules completed")
    return summary

def main():
    parser = argparse.ArgumentParser(description="CDRscope Complete Analysis")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--coords-csv", default=None)
    parser.add_argument("--significance", action="store_true")
    parser.add_argument("--breakthrough", action="store_true")
    parser.add_argument("--validation", action="store_true")
    args = parser.parse_args()

    print("="*60)
    print("CDRscope Complete Analysis Module")
    print("="*60)

    meta = load_reference(args.ref_dir)
    print(f"Loaded {len(meta):,} reference sequences")

    # Merge projected coordinates if available
    if args.coords_csv and os.path.exists(args.coords_csv):
        proj = pd.read_csv(args.coords_csv)
        print(f"Loaded {len(proj):,} projected coordinates")

    if args.significance:
        print("\n--- Significance Analysis ---")
        run_significance(meta, args.output_dir)

    if args.breakthrough:
        print("\n--- Breakthrough Analysis ---")
        run_breakthrough(meta, args.output_dir)

    if args.validation:
        print("\n--- Biological Validation ---")
        run_validation(meta, args.output_dir)

    print("\nDone!")

if __name__ == "__main__":
    main()
