"""
Fig7: Theory-vs-Empirics Comparison Summary
- Panel A: H1-H5 prediction vs actual sign table
- Panel B: Cross-partial H slope (simulation) vs interaction β3 (empirical)
- Panel C: Coefficient comparison (standardized β)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 9,
})

# ============================================================
# Data
# ============================================================

hypotheses = ['H1\nForward\nLinkage', 'H2\nSubsidy\nMain', 'H3\nSubstitution\n(Link×Sub)',
              'H4\nSpatial\nHeterog.', 'H5\nSpatial\nSpillover']
theory_sign = [1, 1, -1, 1, -1]   # predicted direction
empirical_sign = [1, 1, 1, 0.5, -0.5]  # actual (0=null, ±1=confirmed, ±0.5=mixed)
empirical_sig = [True, True, True, False, False]  # statistically significant?

# OLS standardized coefficients (Spec A: DV=log POI, standardized X and Y)
ols_coef_labels = ['Link\n(SA)', 'Sub\n(Policy)', 'Link×Sub', 'Market\nAccess', 'ln(Pop)', 'ln(GDPpc)']
ols_coefs = [2.151, 19.037, 0.051, -0.092, -18.831, 1.057]  # from standardized OLS
ols_ses = [0.115, 4.259, 0.130, 0.125, 4.259, 0.145]
ols_sig = [True, True, False, False, True, True]

# Simulation: M(s_F) slopes for different (γ, α) — from compute_nev_params.py output
M_slopes = [0.02797, 0.02509, 0.02230, 0.01962]  # 4 panels in FigS3
M_mean = np.mean(M_slopes)
M_std = np.std(M_slopes)

# ============================================================
# Figure
# ============================================================
fig = plt.figure(figsize=(16, 10))

# ---- Panel A: Hypothesis Test Summary Table ----
ax_a = fig.add_axes([0.05, 0.52, 0.42, 0.42])
ax_a.axis('off')

col_labels = ['Hypothesis', 'Theory\nPrediction', 'Empirical\nResult', 'Consistency']
table_data = [
    ['H1: Forward Linkage', 'b1 > 0 (+)', 'b1 > 0 (+) ***', '[OK] Consistent'],
    ['H2: Subsidy Main Effect', 'b2 > 0 (+)', 'b2 > 0 (+) ***', '[OK] Consistent'],
    ['H3: Link x Sub Interaction', 'b3 < 0 (-)', 'b3 ~ 0 or > 0 (+) ***', '[!!] Contradicted'],
    ['H4: Spatial Heterogeneity', 'East: H1 dom.\nWest: H2 dom.', 'Not yet tested', '-- Pending'],
    ['H5: Spatial Competition', 'theta_Sub < 0 (-)', 'SDM rho unstable', '-- Pending'],
]

table = ax_a.table(cellText=table_data, colLabels=col_labels, loc='center',
                   cellLoc='center', colColours=['#E3F2FD']*4)
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 2.0)

# Color-code consistency column
for i in range(5):
    cell = table[i+1, 3]
    text = cell.get_text().get_text()
    if 'Consistent' in text:
        cell.set_facecolor('#C8E6C9')
    elif 'Contradicted' in text:
        cell.set_facecolor('#FFCDD2')
    elif 'Pending' in text:
        cell.set_facecolor('#FFF9C4')

ax_a.set_title('A: Hypothesis Test Summary', fontsize=12, fontweight='bold', loc='left', pad=8)

# ---- Panel B: Cross-partial H vs Empirical Interaction ----
ax_b = fig.add_axes([0.55, 0.52, 0.42, 0.42])

# Left y-axis: simulation M(s_F) slopes
x_sim = [1, 2, 3, 4]
colors_sim = ['#1565C0', '#1976D2', '#1E88E5', '#42A5F5']
for i, (slope, c) in enumerate(zip(M_slopes, colors_sim)):
    ax_b.bar(i+1, slope, color=c, alpha=0.8, width=0.6,
             label=f'Panel {i+1}' if i == 0 else '')

ax_b.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
ax_b.axhline(y=M_mean, color='#1565C0', linewidth=1.5, linestyle='--', alpha=0.5)

ax_b.set_xlabel('Parameter grid panel', fontsize=9)
ax_b.set_ylabel('M(s_F) slope (simulation)', fontsize=9, color='#1565C0')
ax_b.tick_params(axis='y', labelcolor='#1565C0')
ax_b.set_xticks([1, 2, 3, 4])
ax_b.set_xticklabels([f'(γ,α)={i+1}' for i in range(4)], fontsize=7)

# Right y-axis: empirical interaction β3
ax_b2 = ax_b.twinx()
ax_b2.bar(6, 0.051, color='#E53935', alpha=0.7, width=0.8, label='Spec A (POI)')
ax_b2.bar(7, 2.588, color='#FF7043', alpha=0.7, width=0.8, label='Spec B (GWR)')
ax_b2.set_ylabel('Empirical β3 (Link×Sub)', fontsize=9, color='#E53935')
ax_b2.tick_params(axis='y', labelcolor='#E53935')

# Add annotation
ax_b.annotate('Simulation:\nH > 0 confirmed\n(36 param combos)',
              xy=(2.5, M_mean), fontsize=7.5, ha='center', va='bottom',
              bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))

ax_b.annotate('Empirical:\nb3 >= 0 (complementarity)\nTheory predicted b3 < 0',
              xy=(6.5, 1.5), fontsize=7.5, ha='center', va='bottom',
              bbox=dict(boxstyle='round', facecolor='#FFEBEE', alpha=0.8))

ax_b.set_title('B: Cross-Partial H (Simulation) vs β3 (Empirical)', fontsize=12, fontweight='bold', loc='left', pad=8)

# ---- Panel C: OLS Coefficient Forest Plot ----
ax_c = fig.add_axes([0.05, 0.06, 0.42, 0.38])

y_pos = range(len(ols_coef_labels))
colors_c = ['#4CAF50' if (label != 'Link×Sub') else '#E53935' for label in ols_coef_labels]

# Normalize coefficients for display (some are huge due to different scales)
# Use standardized betas from OLS
coefs_display = np.array(ols_coefs)
ses_display = np.array(ols_ses)

# For very large coefs, cap the display
coefs_plot = np.clip(coefs_display, -25, 25)
ses_plot = np.clip(ses_display, 0, 10)

ax_c.barh(y_pos, coefs_plot, xerr=ses_plot, color=colors_c, alpha=0.8,
          capsize=3, height=0.6, edgecolor='white', linewidth=0.5)
ax_c.axvline(x=0, color='black', linewidth=0.8, linestyle='-')

# Mark significance
for i, (coef, sig) in enumerate(zip(coefs_plot, ols_sig)):
    label = '***' if sig else ''
    x_pos = coef + (0.3 if coef >= 0 else -0.3)
    ax_c.text(x_pos, i, label, va='center', fontsize=9, fontweight='bold')

ax_c.set_yticks(list(y_pos))
ax_c.set_yticklabels(ols_coef_labels, fontsize=8)
ax_c.set_xlabel('Standardized β (OLS, Spec A: DV=log POI)', fontsize=9)
ax_c.set_title('C: OLS Coefficients — NEV Agglomeration (2023)', fontsize=12, fontweight='bold', loc='left', pad=8)

# Add note about Link×Sub
ax_c.annotate('Key: Link×Sub coef ≈ 0\n(wrong sign vs theory)',
              xy=(0.05, y_pos[2]), fontsize=7, ha='left',
              xytext=(10, 5), textcoords='offset points',
              bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.6),
              arrowprops=dict(arrowstyle='->', color='#E53935'))

# ---- Panel D: Mechanism Diagram ----
ax_d = fig.add_axes([0.55, 0.06, 0.42, 0.38])
ax_d.axis('off')

mechanism_text = (
    "THEORY-EMPIRICS MECHANISM MAPPING\n"
    "══════════════════════════════════\n\n"
    "Structural Model (NEG)              Empirical SDM\n"
    "─────────────────────────────────────────────────\n"
    "λ_M (M-firm share)        ←──→    log(1+NEV_POI)\n"
    "s_r (subsidy rate)        ←──→    GWR policy mentions\n"
    "P_{B,r} (battery price)   ←──→    1 / Supplier Access\n"
    "n_{B,r} (B-firm count)    ←──→    Supplier Access (SA)\n"
    "τ_B (trade cost)          ←──→    1/W (spatial weights)\n\n"
    "KEY FINDINGS:\n"
    "[OK] H1 Forward Linkage: Strongly confirmed (p<0.001)\n"
    "[OK] H2 Subsidy Main: Confirmed for firm agglomeration\n"
    "[!!] H3 Substitution: EMPIRICALLY REVERSED\n"
    "   Model: d2Y/(dLink*dSub) < 0 (substitutes)\n"
    "   Data:  d2Y/(dLink*dSub) >= 0 (complements)\n\n"
    "-> Implication: In China's NEV industry,\n"
    "  subsidies and supply chains are COMPLEMENTS\n"
    "  not substitutes -- governments support cities\n"
    "  that already have industrial foundations."
)

ax_d.text(0, 1, mechanism_text, transform=ax_d.transAxes,
          fontsize=7.5, fontfamily='monospace', verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='#FAFAFA', alpha=0.9, edgecolor='#BDBDBD'))

ax_d.set_title('D: Mechanism Summary', fontsize=12, fontweight='bold', loc='left', pad=8)

# ---- Overall Title ----
fig.suptitle('Theory-vs-Empirics Qualitative Comparison: NEV Spatial Agglomeration',
             fontsize=14, fontweight='bold', y=1.01)

# Save
fig.savefig(os.path.join(OUTPUT_DIR, "Fig7_Theory_Empirics_Comparison.png"), dpi=300)
fig.savefig(os.path.join(OUTPUT_DIR, "Fig7_Theory_Empirics_Comparison.pdf"))
plt.close(fig)
print("Saved Fig7_Theory_Empirics_Comparison.png/.pdf")
print("Done.")
