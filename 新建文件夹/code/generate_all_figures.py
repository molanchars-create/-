"""
Generate all publication-quality figures for NEV Spatial Agglomeration paper.

Figures:
  Fig 1: NEV POI spatial distribution (choropleth + top-city labels)
  Fig 2: GWR NEV attention spatial distribution + LISA cluster map
  Fig 3: Moran scatter plot + coefficient forest plot
  Fig 4: NEV mentions trend 2019-2024 + keyword category breakdown
  Fig 5: SDM effects decomposition (direct/indirect/total)
  Fig 6: Robustness: coefficient comparison across models/W matrices

Requires: geopandas, matplotlib, seaborn, pandas, numpy
"""
import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import seaborn as sns

# ============================================================
# Paths
# ============================================================
DATA_DIR = r"C:\Users\牧原\Desktop\A\工作区\学术研究工作区\06_NEV_产业空间集聚\数据文件"
FIG_DIR  = r"C:\Users\牧原\Desktop\A\工作区\学术研究工作区\06_NEV_产业空间集聚\figures"
SHP_PATH = r"C:\Users\牧原\Desktop\A\工作区\学术研究工作区\撰写时遗留的杂糅文件\grid\ChinaAdminDivisonSHP-master\3. City\city.shp"

os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# Global style
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 9,
})

COLORS = {
    'nev_primary':   '#2196F3',
    'nev_secondary': '#FF9800',
    'nev_tertiary':  '#4CAF50',
    'nev_quaternary':'#E91E63',
    'nev_quinary':   '#9C27B0',
    'nev_senary':    '#00BCD4',
    'grey_light':    '#F5F5F5',
    'grey_medium':   '#BDBDBD',
    'grey_dark':     '#424242',
    'white':         '#FFFFFF',
}

# ============================================================
# Load data
# ============================================================
print("=" * 60)
print("  Loading data...")
print("=" * 60)

panel = pd.read_csv(os.path.join(DATA_DIR, "city_panel_wide.csv"), encoding='utf-8-sig')
gwr_kw = pd.read_csv(os.path.join(DATA_DIR, "gwr_nev_keywords.csv"), encoding='utf-8-sig')
poi_clean = pd.read_csv(os.path.join(DATA_DIR, "nev_poi_clean.csv"), encoding='utf-8-sig')
moran_df = pd.read_csv(os.path.join(DATA_DIR, "moran_I_results.csv"), encoding='utf-8-sig')
lisa_df = pd.read_csv(os.path.join(DATA_DIR, "lisa_clusters.csv"), encoding='utf-8-sig')
reg_summary = pd.read_csv(os.path.join(DATA_DIR, "spatial_regression_summary.csv"), encoding='utf-8-sig')
sdm_effects = pd.read_csv(os.path.join(DATA_DIR, "sdm_effects_decomposition.csv"), encoding='utf-8-sig')

# Load shapefile
import geopandas as gpd
gdf_cities = gpd.read_file(SHP_PATH, encoding='gbk')

# Simplify geometry to reduce PDF file size (tolerance in degrees, ~1km at 30°N)
gdf_cities['geometry'] = gdf_cities['geometry'].simplify(0.01, preserve_topology=True)

# ============================================================
# City name harmonization
# ============================================================
CITY_MAP_PANEL_TO_SHP = {
    '北京': '北京市', '上海': '上海市', '天津': '天津市',
}
# Reverse: strip 市 from shapefile names to match panel
gdf_cities['city_panel'] = gdf_cities['ct_name'].str.replace('市$', '', regex=True)
# Also handle special cases
for panel_name, shp_name in CITY_MAP_PANEL_TO_SHP.items():
    gdf_cities.loc[gdf_cities['ct_name'] == shp_name, 'city_panel'] = panel_name

# ============================================================
# Figure 1: NEV POI Spatial Distribution
# ============================================================
print("\n[Fig 1] NEV POI Spatial Distribution Map")

fig1, axes = plt.subplots(1, 2, figsize=(16, 8))

# --- Panel A: Total NEV POI count by city ---
ax = axes[0]

# Aggregate POI by city
poi_city_counts = poi_clean.groupby('city').size().reset_index(name='nev_poi_total')
poi_city_counts['log_poi'] = np.log1p(poi_city_counts['nev_poi_total'])

# Merge with shapefile
gdf_poi = gdf_cities.merge(poi_city_counts, left_on='city_panel', right_on='city', how='left')
gdf_poi['nev_poi_total'] = gdf_poi['nev_poi_total'].fillna(0)
gdf_poi['log_poi'] = gdf_poi['log_poi'].fillna(0)

# Custom colormap
poi_cmap = LinearSegmentedColormap.from_list('nev_poi', ['#F5F5F5', '#E3F2FD', '#90CAF9', '#42A5F5', '#1E88E5', '#0D47A1'])

gdf_poi.plot(column='log_poi', ax=ax, cmap=poi_cmap, edgecolor='#E0E0E0', linewidth=0.15,
             legend=True, legend_kwds={'shrink': 0.5, 'label': 'log(1 + NEV POI count)'})

# Label top cities
top_n = 12
top_cities_poi = poi_city_counts.nlargest(top_n, 'nev_poi_total')
for _, row in top_cities_poi.iterrows():
    city_name = row['city']
    match = gdf_poi[gdf_poi['city_panel'] == city_name]
    if len(match) > 0:
        centroid = match.geometry.centroid.iloc[0]
        ax.annotate(city_name, (centroid.x, centroid.y), fontsize=5.5, ha='center',
                    color='#212121', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.75, edgecolor='none'))

ax.set_title('A. NEV Industry POI Distribution by City', fontsize=11, fontweight='bold')
ax.axis('off')

# --- Panel B: POI by chain segment (top 20 cities) ---
ax = axes[1]

chain_counts = poi_clean.groupby(['city', 'chain_position']).size().unstack(fill_value=0)
chain_total = chain_counts.sum(axis=1).sort_values(ascending=False)
top20_chain = chain_total.head(20)
chain_plot = chain_counts.loc[top20_chain.index]

chain_colors_map = {
    '充电基础设施': COLORS['nev_primary'],
    '汽车零部件':   COLORS['nev_secondary'],
    '整车制造':     COLORS['nev_tertiary'],
    '电池及上游材料': COLORS['nev_quaternary'],
    '电机电控':     COLORS['nev_quinary'],
    '清洁能源装备': COLORS['nev_senary'],
    '销售与服务':   '#607D8B',
}

chain_order = [c for c in chain_colors_map if c in chain_plot.columns]
chain_plot = chain_plot[chain_order]
colors_bar = [chain_colors_map[c] for c in chain_order]

ax2 = chain_plot.plot(kind='barh', stacked=True, ax=ax, color=colors_bar, width=0.75, edgecolor='white', linewidth=0.3)
ax2.set_xlabel('NEV POI Count', fontsize=9)
ax2.set_title('B. Top 20 Cities by NEV Industry Chain Position', fontsize=11, fontweight='bold')
ax2.legend(loc='lower right', fontsize=7, framealpha=0.9, ncol=2)
ax2.invert_yaxis()
ax2.tick_params(axis='y', labelsize=7)
ax2.set_xlim(0, chain_plot.sum(axis=1).max() * 1.12)

fig1.tight_layout(pad=2)
fig1.savefig(os.path.join(FIG_DIR, "Fig1_NEV_POI_Distribution.png"), dpi=300)
fig1.savefig(os.path.join(FIG_DIR, "Fig1_NEV_POI_Distribution.pdf"))
plt.close(fig1)
print("  Saved Fig1_NEV_POI_Distribution.png/.pdf")

# ============================================================
# Figure 2: GWR NEV Attention + LISA Clusters
# ============================================================
print("\n[Fig 2] GWR NEV Attention & LISA Cluster Map")

fig2, axes = plt.subplots(1, 2, figsize=(16, 8))

# --- Panel A: GWR NEV total mentions (2023) ---
ax = axes[0]

gwr_2023 = panel[panel['year'] == 2023][['city', 'gwr_nev_total_mentions']].dropna()
gdf_gwr = gdf_cities.merge(gwr_2023, left_on='city_panel', right_on='city', how='left')
gdf_gwr['gwr_nev_total_mentions'] = gdf_gwr['gwr_nev_total_mentions'].fillna(0)

gwr_cmap = LinearSegmentedColormap.from_list('gwr', ['#F5F5F5', '#FFF3E0', '#FFCC80', '#FF9800', '#F57C00', '#E65100'])

gdf_gwr.plot(column='gwr_nev_total_mentions', ax=ax, cmap=gwr_cmap, edgecolor='#E0E0E0', linewidth=0.15,
             legend=True, legend_kwds={'shrink': 0.5, 'label': 'NEV Mentions in GWR (2023)'})

top_gwr = gwr_2023.nlargest(12, 'gwr_nev_total_mentions')
for _, row in top_gwr.iterrows():
    match = gdf_gwr[gdf_gwr['city_panel'] == row['city']]
    if len(match) > 0:
        centroid = match.geometry.centroid.iloc[0]
        ax.annotate(row['city'], (centroid.x, centroid.y), fontsize=5.5, ha='center',
                    color='#212121', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.75, edgecolor='none'))

ax.set_title('A. NEV Mentions in Government Work Reports (2023)', fontsize=11, fontweight='bold')
ax.axis('off')

# --- Panel B: LISA Cluster Map ---
ax = axes[1]

lisa_cluster_colors = {
    'HH': '#E53935',  # High-High (hotspot)
    'HL': '#FFB74D',  # High-Low (outlier)
    'LH': '#64B5F6',  # Low-High (outlier)
    'LL': '#1E88E5',  # Low-Low (coldspot)
    'NS': '#E0E0E0',  # Not significant
}

gdf_lisa = gdf_cities.merge(lisa_df[['city', 'cluster']], left_on='city_panel', right_on='city', how='left')
gdf_lisa['cluster'] = gdf_lisa['cluster'].fillna('NS')
gdf_lisa['cluster_color'] = gdf_lisa['cluster'].map(lisa_cluster_colors)

# Base layer — light grey
gdf_cities.plot(ax=ax, color='#FAFAFA', edgecolor='#D0D0D0', linewidth=0.1)

# Plot each cluster
for cluster_type, color in lisa_cluster_colors.items():
    subset = gdf_lisa[gdf_lisa['cluster'] == cluster_type]
    if len(subset) > 0:
        subset.plot(ax=ax, color=color, edgecolor='#666666', linewidth=0.2)

legend_patches = [
    Patch(facecolor='#E53935', edgecolor='#666', label='High-High (Hotspot)'),
    Patch(facecolor='#FFB74D', edgecolor='#666', label='High-Low (Outlier)'),
    Patch(facecolor='#64B5F6', edgecolor='#666', label='Low-High (Outlier)'),
    Patch(facecolor='#1E88E5', edgecolor='#666', label='Low-Low (Coldspot)'),
    Patch(facecolor='#E0E0E0', edgecolor='#666', label='Not Significant'),
]
ax.legend(handles=legend_patches, loc='lower left', fontsize=7, framealpha=0.9, ncol=2)

cluster_counts = gdf_lisa['cluster'].value_counts()
title_text = f'B. LISA Cluster Map (NEV Mentions)\n'
title_text += f'HH={cluster_counts.get("HH",0)}  HL={cluster_counts.get("HL",0)}  '
title_text += f'LH={cluster_counts.get("LH",0)}  LL={cluster_counts.get("LL",0)}  NS={cluster_counts.get("NS",0)}'
ax.set_title(title_text, fontsize=11, fontweight='bold')
ax.axis('off')

fig2.tight_layout(pad=2)
fig2.savefig(os.path.join(FIG_DIR, "Fig2_GWR_Attention_LISA.png"), dpi=300)
fig2.savefig(os.path.join(FIG_DIR, "Fig2_GWR_Attention_LISA.pdf"))
plt.close(fig2)
print("  Saved Fig2_GWR_Attention_LISA.png/.pdf")

# ============================================================
# Figure 3: Moran Scatter Plot + Coefficient Forest Plot
# ============================================================
print("\n[Fig 3] Moran Scatter + Coefficient Forest Plot")

from libpysal.weights import full2W
from esda.moran import Moran

fig3 = plt.figure(figsize=(16, 7))

# --- Panel A: Moran Scatter Plot ---
ax = fig3.add_subplot(1, 2, 1)

W_df = pd.read_csv(os.path.join(DATA_DIR, "spatial_weights_W.csv"), index_col=0, encoding='utf-8-sig')

df23 = panel[panel['year'] == 2023].copy()
avail_cities = sorted(set(W_df.columns) & set(df23['city'].values))
df23_idx = df23.set_index('city').loc[avail_cities]
dv_vals = df23_idx['gwr_nev_total_mentions'].dropna()
reg_cities = list(dv_vals.index)

W_sub = W_df.loc[reg_cities, reg_cities].values
w = full2W(W_sub, ids=reg_cities)

y = dv_vals.values
y_std = (y - y.mean()) / y.std()
Wy_std = (W_sub @ y_std) / W_sub.sum(axis=1) * W_sub.shape[0]

mi = Moran(y, w)

ax.scatter(y_std, Wy_std, c=COLORS['nev_primary'], alpha=0.55, s=40, edgecolors='white', linewidth=0.3, zorder=3)

# Regression line
from numpy.polynomial.polynomial import polyfit
b, m = polyfit(y_std, Wy_std, 1)
x_line = np.linspace(y_std.min() - 0.5, y_std.max() + 0.5, 100)
ax.plot(x_line, b + m * x_line, '--', color=COLORS['nev_quaternary'], linewidth=1.5, zorder=4,
        label=f'Slope = {m:.3f}')

# Reference lines
ax.axhline(0, color=COLORS['grey_medium'], linewidth=0.6, linestyle='-', zorder=1)
ax.axvline(0, color=COLORS['grey_medium'], linewidth=0.6, linestyle='-', zorder=1)

# Annotate quadrants
ax.text(y_std.max() * 0.7, Wy_std.max() * 0.7, 'HH', fontsize=9, fontweight='bold', color='#E53935', alpha=0.6)
ax.text(y_std.min() * 0.7, Wy_std.max() * 0.7, 'LH', fontsize=9, fontweight='bold', color='#64B5F6', alpha=0.6)
ax.text(y_std.max() * 0.7, Wy_std.min() * 0.7, 'HL', fontsize=9, fontweight='bold', color='#FFB74D', alpha=0.6)
ax.text(y_std.min() * 0.7, Wy_std.min() * 0.7, 'LL', fontsize=9, fontweight='bold', color='#1E88E5', alpha=0.6)

ax.set_xlabel('Standardized NEV Mentions (z-score)', fontsize=9)
ax.set_ylabel('Spatial Lag (W × z-score)', fontsize=9)
ax.set_title(f'A. Moran Scatter Plot (I = {mi.I:.4f}, p = {mi.p_sim:.4f})', fontsize=11, fontweight='bold')
ax.legend(fontsize=8)
ax.set_xlim(y_std.min() - 0.5, y_std.max() + 1.5)
ax.set_ylim(Wy_std.min() - 0.5, Wy_std.max() + 1.5)

# --- Panel B: Coefficient Forest Plot ---
ax = fig3.add_subplot(1, 2, 2)

# OLS coefficients from regression summary
ols_rows = reg_summary[reg_summary['model'] == 'OLS (2023)'].copy()

var_labels = {
    'terrain_ruggedness':       'Terrain Ruggedness (TRI)',
    'market_access_log':        'Market Access (log)',
    'supplier_access_log':      'Supplier Access (log)',
    'VL_transport_eq_interp':   'VL Transport Equipment',
    '常住人口（万人）':          'Population (10k)',
    '人均地区生产总值（元）':     'GDP per Capita (yuan)',
}

ols_rows['label'] = ols_rows['variable'].map(var_labels)
ols_rows = ols_rows.dropna(subset=['label'])
ols_rows = ols_rows.sort_values('coef', ascending=True)

y_pos = range(len(ols_rows))

# Compute CIs
from scipy import stats as scipy_stats
n_ols = 295
ols_rows['ci_low'] = ols_rows['coef'] - 1.96 * ols_rows['se']
ols_rows['ci_high'] = ols_rows['coef'] + 1.96 * ols_rows['se']

colors_coef = [COLORS['nev_primary'] if c > 0 else COLORS['nev_quaternary'] for c in ols_rows['coef']]

ax.barh(y_pos, ols_rows['coef'], color=colors_coef, edgecolor='white', height=0.55, zorder=3)
ax.errorbar(ols_rows['coef'], y_pos, xerr=[ols_rows['coef'] - ols_rows['ci_low'],
             ols_rows['ci_high'] - ols_rows['coef']],
            fmt='none', ecolor=COLORS['grey_dark'], capsize=3, linewidth=1.2, zorder=4)

ax.axvline(0, color=COLORS['grey_dark'], linewidth=0.8, linestyle='-', zorder=1)

# Significance stars
for i, (_, row) in enumerate(ols_rows.iterrows()):
    p = row['p_value']
    stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))
    if stars:
        x_pos = row['ci_high'] + 0.5 if row['coef'] > 0 else row['ci_low'] - 2.5
        ax.text(x_pos, i, stars, fontsize=8, va='center', fontweight='bold', color=COLORS['grey_dark'])

ax.set_yticks(y_pos)
ax.set_yticklabels(ols_rows['label'], fontsize=8)
ax.set_xlabel('Standardized Coefficient (β)', fontsize=9)
ax.set_title('B. OLS Coefficient Estimates (2023 Cross-Section)', fontsize=11, fontweight='bold')
ax.invert_yaxis()

fig3.tight_layout(pad=2)
fig3.savefig(os.path.join(FIG_DIR, "Fig3_Moran_Coef_Forest.png"), dpi=300)
fig3.savefig(os.path.join(FIG_DIR, "Fig3_Moran_Coef_Forest.pdf"))
plt.close(fig3)
print("  Saved Fig3_Moran_Coef_Forest.png/.pdf")

# ============================================================
# Figure 4: NEV Mentions Trend + Keyword Category Heatmap
# ============================================================
print("\n[Fig 4] NEV Mentions Trend & Keyword Breakdown")

fig4, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Panel A: NEV Mentions Trend (2019-2024) ---
ax = axes[0]

yearly = gwr_kw.groupby('year').agg(
    total_mentions=('nev_total_mentions', 'sum'),
    n_cities=('city', 'nunique'),
    n_nev_cities=('nev_total_mentions', lambda x: (x > 0).sum()),
    mean_mentions=('nev_total_mentions', 'mean'),
).reset_index()

years = yearly['year'].values

color_line = COLORS['nev_primary']
color_fill = COLORS['nev_primary']

# Dual axis
ax2_line = ax.twinx()

# Bar: total mentions
bars = ax.bar(years, yearly['total_mentions'], color=color_line, alpha=0.25, width=0.6,
              edgecolor=color_line, linewidth=0.8, label='Total NEV Mentions (sum)')

# Line: cities mentioning NEV
line1 = ax2_line.plot(years, yearly['n_nev_cities'], 'o-', color=COLORS['nev_quaternary'],
                      linewidth=2, markersize=7, markerfacecolor='white', markeredgewidth=1.5,
                      label='Cities with ≥1 NEV Mention')

ax.set_xlabel('Year', fontsize=9)
ax.set_ylabel('Total NEV Mentions', fontsize=9, color=color_line)
ax2_line.set_ylabel('Number of Cities', fontsize=9, color=COLORS['nev_quaternary'])
ax.tick_params(axis='y', labelcolor=color_line)
ax2_line.tick_params(axis='y', labelcolor=COLORS['nev_quaternary'])

ax.set_xticks(years)
ax.set_xticklabels([str(int(y)) for y in years])
ax.set_title('A. NEV Mentions in Government Work Reports (2019–2024)', fontsize=11, fontweight='bold')

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2_line.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8, framealpha=0.9)

# --- Panel B: Keyword Category Breakdown ---
ax = axes[1]

kw_categories = {
    'nev_mention':     'NEV Mention',
    'nev_charging':    'Charging Infra.',
    'nev_battery':     'Battery Industry',
    'nev_components':  'Components',
    'nev_policy':      'NEV Policy',
    'nev_clean_energy':'Clean Energy',
}

kw_colors_list = [
    COLORS['nev_primary'],
    COLORS['nev_secondary'],
    COLORS['nev_tertiary'],
    COLORS['nev_quaternary'],
    COLORS['nev_quinary'],
    COLORS['nev_senary'],
]

cat_yearly = gwr_kw.groupby('year')[[c for c in kw_categories]].sum()

# Stacked area chart (normalized to 100%)
cat_pct = cat_yearly.div(cat_yearly.sum(axis=1), axis=0) * 100

ax.stackplot(cat_pct.index, *[cat_pct[c] for c in kw_categories],
             labels=[kw_categories[c] for c in kw_categories],
             colors=kw_colors_list, alpha=0.8, edgecolor='white', linewidth=0.3)

ax.set_xlabel('Year', fontsize=9)
ax.set_ylabel('Share of Total Mentions (%)', fontsize=9)
ax.set_title('B. NEV Keyword Category Composition (2019–2024)', fontsize=11, fontweight='bold')
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7.5, framealpha=0.9)
ax.set_xticks(years)
ax.set_xticklabels([str(int(y)) for y in years])
ax.set_ylim(0, 105)

fig4.tight_layout(pad=2)
fig4.savefig(os.path.join(FIG_DIR, "Fig4_NEV_Mentions_Trend.png"), dpi=300)
fig4.savefig(os.path.join(FIG_DIR, "Fig4_NEV_Mentions_Trend.pdf"))
plt.close(fig4)
print("  Saved Fig4_NEV_Mentions_Trend.png/.pdf")

# ============================================================
# Figure 5: SDM Effects Decomposition
# ============================================================
print("\n[Fig 5] SDM Effects Decomposition")

fig5, ax = plt.subplots(figsize=(12, 7))

effects_vars = [var_labels.get(v, v) for v in sdm_effects['variable']]
y_positions = range(len(effects_vars))

bar_height = 0.25
y_direct  = [y + bar_height for y in y_positions]
y_indirect = [y for y in y_positions]
y_total   = [y - bar_height for y in y_positions]

ax.barh(y_direct,  sdm_effects['direct'],   height=bar_height, color=COLORS['nev_primary'],
        edgecolor='white', linewidth=0.5, label='Direct Effect', zorder=3)
ax.barh(y_indirect, sdm_effects['indirect'], height=bar_height, color=COLORS['nev_secondary'],
        edgecolor='white', linewidth=0.5, label='Indirect (Spillover)', zorder=3)
ax.barh(y_total,   sdm_effects['total'],    height=bar_height, color=COLORS['nev_quaternary'],
        edgecolor='white', linewidth=0.5, label='Total Effect', zorder=3)

ax.axvline(0, color=COLORS['grey_dark'], linewidth=0.8, zorder=1)
ax.set_yticks(y_positions)
ax.set_yticklabels(effects_vars, fontsize=9)
ax.set_xlabel('Effect Size (standardized)', fontsize=9)
ax.set_title('Spatial Durbin Model: Direct, Indirect & Total Effects (2023)', fontsize=11, fontweight='bold')
ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
ax.invert_yaxis()

# Add spillover ratio annotations
for i, (_, row) in enumerate(sdm_effects.iterrows()):
    if abs(row['direct']) > 0.01:
        ratio = abs(row['indirect'] / row['direct'])
        ax.text(max(row['direct'], row['indirect'], row['total']) + 1.5, i,
                f'|Indirect/Direct| = {ratio:.1f}', fontsize=7, va='center', color=COLORS['grey_dark'])

fig5.tight_layout()
fig5.savefig(os.path.join(FIG_DIR, "Fig5_SDM_Effects.png"), dpi=300)
fig5.savefig(os.path.join(FIG_DIR, "Fig5_SDM_Effects.pdf"))
plt.close(fig5)
print("  Saved Fig5_SDM_Effects.png/.pdf")

# ============================================================
# Figure 6: Robustness — Model Comparison & Moran's I Summary
# ============================================================
print("\n[Fig 6] Robustness & Diagnostics")

fig6 = plt.figure(figsize=(16, 12))

# --- Panel A: Moran's I Summary ---
ax = fig6.add_subplot(2, 2, 1)

moran_plot = moran_df.dropna(subset=['moran_I']).copy()
moran_plot['abs_I'] = moran_plot['moran_I'].abs()
moran_plot = moran_plot.sort_values('abs_I')

moran_labels_map = {
    'gwr_nev_total_mentions':  'GWR: NEV Total',
    'gwr_nev_mention':         'GWR: NEV Mention',
    'gwr_nev_charging':        'GWR: Charging',
    'gwr_nev_battery':         'GWR: Battery',
    'gwr_nev_components':      'GWR: Components',
    'gwr_nev_policy':          'GWR: NEV Policy',
    'VL_transport_eq_interp':  'VL Transport Eq.',
    'market_access_log':       'Market Access',
    'supplier_access_log':     'Supplier Access',
    'nev_poi_count':           'NEV POI Count',
    'terrain_ruggedness':      'Terrain Ruggedness',
}
moran_plot['label'] = moran_plot['variable'].map(moran_labels_map).fillna(moran_plot['variable'])

bar_colors = [COLORS['nev_primary'] if row['significant'] != '' and row['significant'] != '随机分布'
              else COLORS['grey_medium'] for _, row in moran_plot.iterrows()]

bars_m = ax.barh(range(len(moran_plot)), moran_plot['moran_I'], color=bar_colors, edgecolor='white', height=0.6)

# Significance stars
for i, (_, row) in enumerate(moran_plot.iterrows()):
    stars = row['significant']
    if stars:
        x_pos = row['moran_I'] + 0.005 if row['moran_I'] > 0 else row['moran_I'] - 0.03
        ax.text(x_pos, i, stars, fontsize=8, va='center', fontweight='bold')

ax.axvline(0, color=COLORS['grey_dark'], linewidth=0.8)
ax.set_yticks(range(len(moran_plot)))
ax.set_yticklabels(moran_plot['label'], fontsize=8)
ax.set_xlabel("Moran's I", fontsize=9)
ax.set_title("A. Global Moran's I — Variable Comparison", fontsize=11, fontweight='bold')
ax.invert_yaxis()

# --- Panel B: Moran's I by Year (panel spatial autocorrelation) ---
ax = fig6.add_subplot(2, 2, 2)

# Compute Moran's I for NEV mentions by year (panel spatial autocorrelation check)
mi_by_year = []
for yr in sorted(panel['year'].unique()):
    df_y = panel[panel['year'] == yr].set_index('city')
    dv_y = df_y['gwr_nev_total_mentions'].dropna()
    yr_cities = [c for c in dv_y.index if c in W_df.columns and c in W_df.index]
    if len(yr_cities) < 50:
        continue

    y_yr = dv_y[yr_cities].values
    W_yr = W_df.loc[yr_cities, yr_cities].values
    w_yr = full2W(W_yr, ids=yr_cities)

    try:
        mi_yr = Moran(y_yr, w_yr)
        mi_by_year.append({
            'year': int(yr),
            'moran_I': mi_yr.I,
            'p_value': mi_yr.p_sim,
            'significant': mi_yr.p_sim < 0.05,
        })
    except Exception:
        pass

mi_yr_df = pd.DataFrame(mi_by_year)

bars_mi_yr = ax.bar(
    mi_yr_df['year'].astype(str),
    mi_yr_df['moran_I'],
    color=[COLORS['nev_primary'] if s else COLORS['grey_medium'] for s in mi_yr_df['significant']],
    edgecolor='white', width=0.6
)

ax.axhline(0, color=COLORS['grey_dark'], linewidth=0.8)

# Add p-value annotations
for i, (_, row) in enumerate(mi_yr_df.iterrows()):
    stars = '**' if row['p_value'] < 0.05 else ('*' if row['p_value'] < 0.10 else '')
    y_pos = row['moran_I'] + 0.003 if row['moran_I'] > 0 else row['moran_I'] - 0.008
    ax.text(i, y_pos, stars, ha='center', fontsize=10, fontweight='bold', color=COLORS['grey_dark'])
    ax.text(i, row['moran_I'] / 2 if row['moran_I'] > 0 else row['moran_I'] * 1.5,
            f'{row["moran_I"]:.4f}', ha='center', fontsize=6.5, color='white' if row['significant'] else COLORS['grey_dark'])

ax.set_xlabel('Year', fontsize=9)
ax.set_ylabel("Moran's I", fontsize=9)
ax.set_title('B. Panel Moran\'s I: NEV Mentions by Year', fontsize=11, fontweight='bold')
ax.tick_params(axis='x', labelsize=8)

# Add expected I reference line
n_avg = 295
E_I = -1 / (n_avg - 1)
ax.axhline(E_I, color=COLORS['nev_quaternary'], linewidth=0.8, linestyle='--', alpha=0.6,
           label=f'E[I] = {E_I:.4f}')
ax.legend(fontsize=7.5)

# --- Panel C: Correlation Heatmap ---
ax = fig6.add_subplot(2, 2, 3)

corr_vars_map = {
    'gwr_nev_total_mentions': 'NEV Mentions',
    'nev_poi_count':          'NEV POI',
    'terrain_ruggedness':     'TRI',
    'market_access_log':      'Market Access',
    'supplier_access_log':    'Supplier Access',
    'VL_transport_eq_interp': 'VL Transport',
    '常住人口（万人）':        'Population',
    '人均地区生产总值（元）':   'GDP p.c.',
}

df23_corr = panel[panel['year'] == 2023][list(corr_vars_map.keys())].dropna()
corr_matrix = df23_corr.corr()
corr_matrix.rename(index=corr_vars_map, columns=corr_vars_map, inplace=True)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-0.5, vmax=0.8, ax=ax, linewidths=0.5,
            cbar_kws={'shrink': 0.75, 'label': 'Pearson r'},
            annot_kws={'fontsize': 8})

ax.set_title('C. Correlation Matrix (2023)', fontsize=11, fontweight='bold')
ax.tick_params(axis='both', labelsize=7.5)

# --- Panel D: NEV POI by Chain Position Treemap (Pie chart alternative) ---
ax = fig6.add_subplot(2, 2, 4)

chain_totals = poi_clean['chain_position'].value_counts()
chain_totals = chain_totals[chain_totals > 0]

pie_colors_chain = [chain_colors_map.get(c, COLORS['grey_medium']) for c in chain_totals.index]

wedges, texts, autotexts = ax.pie(
    chain_totals.values,
    labels=None,
    autopct='%1.1f%%',
    colors=pie_colors_chain,
    startangle=90,
    pctdistance=0.8,
    wedgeprops=dict(width=0.45, edgecolor='white', linewidth=1.5),
)

for at in autotexts:
    at.set_fontsize(8)
    at.set_fontweight('bold')

# Legend
ax.legend(
    [f'{label}\n(n={count:,})' for label, count in zip(chain_totals.index, chain_totals.values)],
    title='Chain Position',
    loc='center left',
    bbox_to_anchor=(1, 0, 0.5, 1),
    fontsize=8,
    title_fontsize=9,
    framealpha=0.9,
)
ax.set_title(f'D. NEV POI by Industry Chain Position\n(Total = {chain_totals.sum():,} POIs)', fontsize=11, fontweight='bold')

fig6.tight_layout(pad=2.5)
fig6.savefig(os.path.join(FIG_DIR, "Fig6_Robustness_Diagnostics.png"), dpi=300)
fig6.savefig(os.path.join(FIG_DIR, "Fig6_Robustness_Diagnostics.pdf"))
plt.close(fig6)
print("  Saved Fig6_Robustness_Diagnostics.png/.pdf")

# ============================================================
# Print summary
# ============================================================
print(f"\n{'='*60}")
print(f"  All figures generated successfully!")
print(f"{'='*60}")
print(f"  Output directory: {FIG_DIR}")
print(f"  Files:")
for f in sorted(os.listdir(FIG_DIR)):
    if f.endswith(('.png', '.pdf')):
        size_kb = os.path.getsize(os.path.join(FIG_DIR, f)) / 1024
        print(f"    {f}  ({size_kb:.0f} KB)")
print(f"{'='*60}")
