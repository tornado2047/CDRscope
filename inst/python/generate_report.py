#!/usr/bin/env python3
"""
CDRscope Automated Report Generator
====================================
Generates a self-contained HTML report from analysis outputs.

Usage:
  python3 generate_report.py --output-dir DIR --chain CHAIN
"""
import os, sys, json, argparse, base64, glob
from datetime import datetime

def img_to_base64(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
    return ""

def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def load_csv(path):
    if os.path.exists(path):
        import pandas as pd
        return pd.read_csv(path)
    return None

def generate_report(output_dir, chain):
    bt = load_json(os.path.join(output_dir, "breakthrough_summary.json"))
    val = load_json(os.path.join(output_dir, "validation_summary.json"))
    cv_df = load_csv(os.path.join(output_dir, "cv_results.csv"))
    fi_df = load_csv(os.path.join(output_dir, "feature_importance.csv"))
    sig_df = load_csv(os.path.join(output_dir, "domain_significance.csv"))

    # Find images
    img_pattern = os.path.join(output_dir, "*.png")
    images = sorted(glob.glob(img_pattern))

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CDRscope v2.0 Analysis Report</title>
<style>
:root {{
  --navy: #1B3A5C; --steel: #3D6A8C; --accent: #D97742;
  --green: #5B9279; --gray: #9A9A9A; --light: #F7F9FB;
  --border: #E0E4E8; --ink: #1A1A1A;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Georgia', 'PingFang SC', serif; color: var(--ink); background: #fff; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
h1 {{ color: var(--navy); font-size: 28px; margin-bottom: 8px; }}
h2 {{ color: var(--navy); font-size: 22px; margin: 30px 0 12px; padding-bottom: 6px; border-bottom: 2px solid var(--navy); }}
h3 {{ color: var(--steel); font-size: 18px; margin: 20px 0 8px; }}
.metric {{ display: inline-block; background: var(--light); border: 1px solid var(--border); border-radius: 8px; padding: 12px 20px; margin: 8px; text-align: center; }}
.metric .value {{ font-size: 28px; font-weight: bold; color: var(--accent); }}
.metric .label {{ font-size: 13px; color: var(--steel); }}
table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 14px; }}
th {{ background: var(--navy); color: white; padding: 8px 12px; text-align: left; }}
td {{ padding: 6px 12px; border-bottom: 1px solid var(--border); }}
tr:nth-child(even) {{ background: var(--light); }}
img {{ max-width: 100%; border: 1px solid var(--border); border-radius: 8px; margin: 12px 0; }}
.summary-box {{ background: var(--light); border-left: 4px solid var(--accent); padding: 16px; margin: 12px 0; border-radius: 0 8px 8px 0; }}
.footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--border); color: var(--gray); font-size: 12px; text-align: center; }}
</style>
</head>
<body>

<h1>CDRscope v2.0 Complete Analysis Report</h1>
<p style="color:var(--steel); font-size:14px;">Chain mode: <strong>{chain}</strong> | Generated: {now}</p>

<div class="summary-box">
<h3>Core Finding</h3>
<p>TCR repertoire disease signal operates at the <strong>population frequency level</strong>,
not at the individual sequence level. The 65-feature sample-level classifier captures
this population-level frequency redistribution, achieving high accuracy.</p>
</div>
"""

    # CV Results
    if cv_df is not None and len(cv_df) > 0:
        html += "<h2>Cross-Validation Results</h2>\n"
        if 'mean_accuracy' in cv_df.columns:
            acc = cv_df['mean_accuracy'].iloc[0]
            html += f'<div class="metric"><div class="value">{acc*100:.1f}%</div><div class="label">Accuracy</div></div>\n'
        if 'auc_mean' in cv_df.columns:
            auc = cv_df['auc_mean'].iloc[0]
            html += f'<div class="metric"><div class="value">{auc:.3f}</div><div class="label">AUC-ROC</div></div>\n'
        if 'auc_pr_mean' in cv_df.columns:
            auc_pr = cv_df['auc_pr_mean'].iloc[0]
            html += f'<div class="metric"><div class="value">{auc_pr:.3f}</div><div class="label">AUC-PR</div></div>\n'
        html += cv_df.to_html(index=False, classes='cv-table', border=0)

    # CV Details (per-fold)
    cv_details_path = os.path.join(output_dir, "cv_details.csv")
    if os.path.exists(cv_details_path):
        import pandas as pd
        cv_details = pd.read_csv(cv_details_path)
        if len(cv_details) > 0:
            html += "<h3>Per-Fold Details</h3>\n"
            html += cv_details.to_html(index=False, border=0)

    # ROC Curve image
    roc_img = os.path.join(output_dir, "roc_curve.png")
    roc_b64 = img_to_base64(roc_img)
    if roc_b64:
        html += f'<h3>ROC Curve</h3>\n<img src="{roc_b64}" alt="ROC Curve" style="max-width:600px;">\n'

    # Feature Importance
    if fi_df is not None and len(fi_df) > 0:
        html += "<h2>Feature Importance (Top 15)</h2>\n"
        html += fi_df.head(15).to_html(index=False, border=0)

    # Breakthrough
    if bt:
        html += "<h2>Breakthrough Analysis</h2>\n"
        if 'disease_scoring' in bt:
            ds = bt['disease_scoring']
            html += f"""<div class="summary-box">
<h3>Disease Scoring (Key Finding)</h3>
<p>Sequence-level UMAP AUC: UMAP1={ds.get('umap1_auc',0):.3f}, UMAP2={ds.get('umap2_auc',0):.3f}</p>
<p><strong>{ds.get('conclusion', '')}</strong></p>
</div>\n"""
        if 'centroid_shift' in bt:
            cs = bt['centroid_shift']
            html += f"""<div class="summary-box">
<h3>Centroid Shift</h3>
<p>UMAP2 shift: {cs.get('umap2_shift',0):.4f} | Spatial overlap: {cs.get('spatial_overlap',0):.1%} | MWU p={cs.get('mwu_p',0):.2e}</p>
</div>\n"""
        if 'expansion' in bt:
            ex = bt['expansion']
            html += f"""<div class="summary-box">
<h3>Clonal Expansion</h3>
<p>RA mean log(dup): {ex.get('ra_mean',0):.2f} | Control: {ex.get('ctrl_mean',0):.2f} | p={ex.get('p_value',0):.2e}</p>
</div>\n"""

    # Significance
    if sig_df is not None and len(sig_df) > 0:
        html += "<h2>Domain-Level Significance Analysis</h2>\n"
        sig_sig = sig_df[sig_df['p_value'] < 0.05] if 'p_value' in sig_df.columns else sig_df
        html += f"<p>{len(sig_sig)} significant domain-category tests (p<0.05)</p>\n"
        html += sig_sig.to_html(index=False, border=0)

    # Validation
    if val:
        html += "<h2>Biological Validation</h2>\n"
        if 'frequency_redistribution' in val:
            html += "<h3>Frequency Redistribution</h3>\n<table><tr><th>Domain</th><th>RA Freq</th><th>Ctrl Freq</th><th>Ratio</th></tr>\n"
            for k, v in val['frequency_redistribution'].items():
                html += f"<tr><td>{k}</td><td>{v['ra_freq']:.4f}</td><td>{v['ctrl_freq']:.4f}</td><td>{v['ratio']:.2f}</td></tr>\n"
            html += "</table>\n"
        if 'citrullination_axis' in val:
            html += "<h3>Citrullination-Hydrophobicity Axis</h3>\n<table><tr><th>Antigen</th><th>ΔCharge</th><th>ΔHydrophobicity</th></tr>\n"
            for name, props in val['citrullination_axis'].items():
                html += f"<tr><td>{name}</td><td>{props['delta_charge']:+.1f}</td><td>{props['delta_hydro']:+.3f}</td></tr>\n"
            html += "</table>\n"
        if 'hla_stratification' in val:
            hla = val['hla_stratification']
            html += "<h3>HLA Stratification</h3>\n"
            if 'v_gene_proxy' in hla:
                html += "<table><tr><th>V Gene</th><th>RA Freq</th><th>Ctrl Freq</th><th>log2(RA/Ctrl)</th></tr>\n"
                for gene, props in hla['v_gene_proxy'].items():
                    html += f"<tr><td>{gene}</td><td>{props['ra_freq']:.5f}</td><td>{props['ctrl_freq']:.5f}</td><td>{props['log2_ratio']:+.3f}</td></tr>\n"
                html += "</table>\n"
            if 'qdfa_motif' in hla:
                q = hla['qdfa_motif']
                html += f"<p>QDFA motif: RA={q['n_ra']}, Ctrl={q['n_ctrl']}, ratio={q['ra_ctrl_ratio']:.2f} (expected HLA-DRB1*15 ratio: {q['expected_hla_ratio']:.2f})</p>\n"

    # Images
    if images:
        html += "<h2>Visualizations</h2>\n"
        for img_path in images:
            img_name = os.path.basename(img_path)
            img_b64 = img_to_base64(img_path)
            if img_b64:
                html += f'<h3>{img_name}</h3>\n<img src="{img_b64}" alt="{img_name}">\n'

    # Images from docs
    docs_dir = os.path.join(os.path.dirname(output_dir), "docs")
    if os.path.isdir(docs_dir):
        doc_imgs = sorted(glob.glob(os.path.join(docs_dir, "*.png")))
        if doc_imgs:
            html += "<h2>Reference Map Visualizations</h2>\n"
            for img_path in doc_imgs[:10]:
                img_name = os.path.basename(img_path)
                img_b64 = img_to_base64(img_path)
                if img_b64:
                    html += f'<h3>{img_name}</h3>\n<img src="{img_b64}" alt="{img_name}">\n'

    html += f"""
<div class="footer">
<p>CDRscope v2.0 — Complete Closed-Loop Analysis Pipeline</p>
<p>Generated: {now} | Chain: {chain}</p>
<p>10-Module Pipeline: Input → Features → ESM-2 → Reference Map → Classification → UMAP → Significance → Breakthrough → Validation → Report</p>
</div>

</body>
</html>"""

    report_path = os.path.join(output_dir, "CDRscope_Analysis_Report.html")
    with open(report_path, 'w') as f:
        f.write(html)
    print(f"Report saved: {report_path}")
    return report_path

def main():
    parser = argparse.ArgumentParser(description="CDRscope Report Generator")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chain", default="single")
    args = parser.parse_args()
    generate_report(args.output_dir, args.chain)

if __name__ == "__main__":
    main()
