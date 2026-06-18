"""
Comprehensive Spatial Econometric Analysis for NEV Agglomeration
================================================================
A. Descriptive statistics + correlation matrix
B. LM tests for spatial model selection
C. Panel spatial models (pooled + FE SDM/SAR/SEM)
D. SDM effects decomposition (direct/indirect/total)
E. Robustness: alternative DV (NEV POI), alternative W (k-NN)

Ref: Elhorst (2014) Spatial Econometrics, LeSage & Pace (2009)
"""
import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

from libpysal.weights import full2W
from esda.moran import Moran
from scipy import stats as scipy_stats
import spreg
from spreg import OLS, GM_Lag, GM_Error, GM_Combo

OUTPUT_DIR = r"C:\Users\牧原\Desktop\A\工作区\学术研究工作区\06_NEV_产业空间集聚"
PANEL_WIDE = os.path.join(OUTPUT_DIR, "city_panel_wide.csv")
W_MATRIX = os.path.join(OUTPUT_DIR, "spatial_weights_W.csv")

# ============================================================
# Load data
# ============================================================
print("=" * 70)
print("  Full Spatial Econometric Analysis")
print("=" * 70)

panel = pd.read_csv(PANEL_WIDE, encoding='utf-8-sig')
W_df = pd.read_csv(W_MATRIX, index_col=0, encoding='utf-8-sig')
W_cities_all = list(W_df.columns)

# Variable definitions
DV = 'gwr_nev_total_mentions'
DV_ALT = 'nev_poi_count'  # alternative DV for robustness
IV_MAIN = 'terrain_ruggedness'
X_VARS = [IV_MAIN, 'market_access_log', 'supplier_access_log',
          'VL_transport_eq_interp', '常住人口（万人）', '人均地区生产总值（元）']

# ============================================================
# Part A: Descriptive Statistics
# ============================================================
print(f"\n{'='*70}")
print(f"  Part A: Descriptive Statistics (2023)")
print(f"{'='*70}")

df23 = panel[panel['year'] == 2023].copy()
desc_vars = [DV] + X_VARS
desc_labels = ['NEV总提及', '地形起伏度', 'MA(log)', 'SA(log)',
               'VL交运设备', '常住人口(万人)', '人均GDP(元)']

print(f"\n  {'Variable':<20s} {'Obs':>6s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
print(f"  {'-'*60}")
for col, label in zip(desc_vars, desc_labels):
    if col in df23.columns:
        s = df23[col].dropna()
        print(f"  {label:<20s} {len(s):>6d} {s.mean():>10.3f} {s.std():>10.3f} {s.min():>10.3f} {s.max():>10.3f}")

# Correlation matrix
corr_vars = [v for v in desc_vars if v in df23.columns]
corr_df = df23[corr_vars].dropna()
corr_matrix = corr_df.corr()
print(f"\n  Correlation Matrix:")
print(f"  {'':>20s}", end='')
for label in desc_labels[:len(corr_vars)]:
    print(f" {label[:8]:>9s}", end='')
print()
for i, (col, label) in enumerate(zip(corr_vars, desc_labels)):
    print(f"  {label:<20s}", end='')
    for j in range(len(corr_vars)):
        r = corr_matrix.iloc[i, j]
        stars = '***' if abs(r) > 0.5 else ('**' if abs(r) > 0.3 else ('*' if abs(r) > 0.1 else ''))
        print(f" {r:>9.3f}", end='')
    print()

# ============================================================
# Part B: LM Tests for Spatial Model Selection
# ============================================================
print(f"\n{'='*70}")
print(f"  Part B: LM Tests — Spatial Model Specification (2023)")
print(f"{'='*70}")

YEAR = 2023
df_yr = panel[panel['year'] == YEAR].copy()
avail_cities = sorted(set(W_cities_all) & set(df_yr['city'].values))
df_yr = df_yr.set_index('city').loc[avail_cities].reset_index()

# Build W
W_sub = W_df.loc[avail_cities, avail_cities].values
w = full2W(W_sub, ids=avail_cities)

# Prepare data
X_cols_avail = [v for v in X_VARS if v in df_yr.columns]
df_reg = df_yr[['city'] + [DV] + X_cols_avail].dropna()

reg_cities = [c for c in avail_cities if c in df_reg['city'].values]
df_reg = df_reg.set_index('city').loc[reg_cities]
n = len(reg_cities)

y = df_reg[DV].values
X = df_reg[X_cols_avail].values
# Standardize
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_std[X_std == 0] = 1
X = (X - X_mean) / X_std

W_reg = W_df.loc[reg_cities, reg_cities].values
w_reg = full2W(W_reg, ids=reg_cities)

# Step 1: OLS
ols = OLS(y, X, w=w_reg, name_y=DV, name_x=X_cols_avail, name_ds='NEV', name_w='W',
          spat_diag=True, moran=True)
residuals = ols.u
sigma2 = (residuals.T @ residuals) / n

# Step 2: LM statistics
# W as dense for computation
W_dense = W_reg.copy()
# Robust LM requires traces
W2 = W_dense @ W_dense
T1 = np.trace(W2.T @ W2 + W2)  # tr[(W'+W)W] for LM-error
# Actually use standard formulas
WW = W_dense.T @ W_dense
T1_val = np.trace(WW + W_dense @ W_dense)  # tr(W'W + W²)
M = np.eye(n) - X @ np.linalg.inv(X.T @ X) @ X.T

# LM-Error
e_We = residuals.T @ W_dense @ residuals
LM_err = float((e_We / sigma2)**2 / T1_val)
p_LM_err = float(1 - scipy_stats.chi2.cdf(LM_err, 1))

# LM-Lag
Wy = W_dense @ y
e_Wy = residuals.T @ Wy
# J = (1/sigma2) * [(WXb)' M (WXb) + T1*sigma2]
betas_flat = ols.betas.flatten()
# If betas includes intercept (first element), skip it
if len(betas_flat) == X.shape[1] + 1:
    betas_x = betas_flat[1:]
else:
    betas_x = betas_flat
WXb = W_dense @ X @ betas_x
J_lag = (WXb.T @ M @ WXb + T1_val * sigma2) / sigma2
LM_lag = float((e_Wy / sigma2)**2 / J_lag)
p_LM_lag = float(1 - scipy_stats.chi2.cdf(LM_lag, 1))

# Robust LM-Error
LM_err_robust = float(((e_We / sigma2) - (T1_val / J_lag) * (e_Wy / sigma2))**2 / (T1_val * (1 - T1_val / J_lag)))
p_LM_err_rob = float(1 - scipy_stats.chi2.cdf(max(LM_err_robust, 0), 1))

# Robust LM-Lag
LM_lag_robust = float(((e_Wy / sigma2) - (e_We / sigma2))**2 / (J_lag - T1_val))
p_LM_lag_rob = float(1 - scipy_stats.chi2.cdf(max(LM_lag_robust, 0), 1))

print(f"\n  {'Test':<35s} {'Statistic':>10s} {'p-value':>10s}")
print(f"  {'-'*57}")
print(f"  {'LM-Lag':<35s} {LM_lag:>10.4f} {p_LM_lag:>10.4f}")
print(f"  {'Robust LM-Lag':<35s} {LM_lag_robust:>10.4f} {p_LM_lag_rob:>10.4f}")
print(f"  {'LM-Error':<35s} {LM_err:>10.4f} {p_LM_err:>10.4f}")
print(f"  {'Robust LM-Error':<35s} {LM_err_robust:>10.4f} {p_LM_err_rob:>10.4f}")

if p_LM_lag < 0.05 or p_LM_err < 0.05:
    if p_LM_lag_rob < 0.05 and p_LM_err_rob >= 0.05:
        print(f"\n  >> Recommendation: SAR (Spatial Lag)")
    elif p_LM_err_rob < 0.05 and p_LM_lag_rob >= 0.05:
        print(f"\n  >> Recommendation: SEM (Spatial Error)")
    elif p_LM_lag_rob < 0.05 and p_LM_err_rob < 0.05:
        print(f"\n  >> Recommendation: SDM (both significant, prefer general model)")
    else:
        print(f"\n  >> Both robust LMs not significant — OLS sufficient")
else:
    print(f"\n  >> No spatial dependence detected — OLS sufficient")

# ============================================================
# Part C: Panel Spatial Econometrics
# ============================================================
print(f"\n{'='*70}")
print(f"  Part C: Panel Spatial Models (2019-2024)")
print(f"{'='*70}")

# Pooled OLS with year fixed effects (baseline)
all_years_data = []
for yr in sorted(panel['year'].unique()):
    df_y = panel[panel['year'] == yr].copy()
    yr_cities = [c for c in avail_cities if c in df_y['city'].values]
    df_y = df_y.set_index('city').loc[yr_cities]
    dv_vals = df_y[DV].dropna()
    yr_cities_clean = [c for c in yr_cities if c in dv_vals.index]
    if len(yr_cities_clean) < 100:
        continue

    y_clean = dv_vals[yr_cities_clean].values
    X_clean = df_y.loc[yr_cities_clean, X_cols_avail].dropna()
    final_cities = [c for c in yr_cities_clean if c in X_clean.index]
    if len(final_cities) < 100:
        continue

    all_years_data.append({
        'year': yr,
        'cities': final_cities,
        'y': dv_vals[final_cities].values,
        'X_raw': X_clean.loc[final_cities].values,
    })

# Panel FE estimation (within-transformation)
# Stack all years
y_all = np.concatenate([d['y'] for d in all_years_data])
X_all_raw = np.concatenate([d['X_raw'] for d in all_years_data])
city_ids_all = []
year_ids_all = []
for d in all_years_data:
    city_ids_all.extend(d['cities'])
    year_ids_all.extend([d['year']] * len(d['cities']))

# Create city and year dummies for FE
from sklearn.preprocessing import OneHotEncoder
city_enc = OneHotEncoder(sparse_output=False)
city_dummies = city_enc.fit_transform(np.array(city_ids_all).reshape(-1, 1))
year_enc = OneHotEncoder(sparse_output=False)
year_dummies = year_enc.fit_transform(np.array(year_ids_all).reshape(-1, 1))

# Standardize X
X_all_mean = X_all_raw.mean(axis=0)
X_all_std = X_all_raw.std(axis=0)
X_all_std[X_all_std == 0] = 1
X_all = (X_all_raw - X_all_mean) / X_all_std

# Remove first city and year dummy to avoid perfect collinearity
city_fe = city_dummies[:, 1:]
year_fe = year_dummies[:, 1:]
n_total = len(y_all)
n_cities_panel = city_fe.shape[1]
n_years_panel = year_fe.shape[1]

# Within-transformation: demean by city
# Simpler: use city dummies + year dummies in regression
# Manual FE: residualize y and X on city and year dummies
# For spatial FE, we use the stacked approach

# Panel OLS with city+year FE
X_fe = np.column_stack([X_all, city_fe, year_fe])
fe_labels = X_cols_avail + [f'city_{i}' for i in range(n_cities_panel)] + [f'yr_{i}' for i in range(n_years_panel)]

XTX = X_fe.T @ X_fe
XTy = X_fe.T @ y_all
beta_fe = np.linalg.solve(XTX, XTy)
y_pred_fe = X_fe @ beta_fe
resid_fe = y_all - y_pred_fe
rss = resid_fe.T @ resid_fe
tss = (y_all - y_all.mean()).T @ (y_all - y_all.mean())
r2_fe = 1 - rss / tss
se_fe = np.sqrt(np.diag(rss / (n_total - X_fe.shape[1]) * np.linalg.inv(XTX)))

print(f"\n  [Panel FE (city + year dummies)]")
print(f"  N = {n_total}, Cities = {n_cities_panel+1}, Years = {n_years_panel+1}")
print(f"  R² = {r2_fe:.4f}, Adj R² = {1-(1-r2_fe)*(n_total-1)/(n_total-X_fe.shape[1]-1):.4f}")
print(f"  NOTE: terrain_ruggedness, MA, SA are time-invariant → absorbed by city FE")
print(f"  {'Variable':<28s} {'Coef':>8s} {'Std.Err':>8s} {'t':>8s} {'p':>8s}")
print(f"  {'-'*64}")
for i, col in enumerate(X_cols_avail):
    if np.isnan(se_fe[i]) or np.isinf(se_fe[i]):
        print(f"  {col:<28s} {'(absorbed)':>8s}")
    else:
        t_val = beta_fe[i] / se_fe[i]
        p_val = 2 * scipy_stats.t.sf(abs(t_val), df=n_total - X_fe.shape[1])
        print(f"  {col:<28s} {beta_fe[i]:>8.4f} {se_fe[i]:>8.4f} {t_val:>8.3f} {p_val:>8.4f}")

# ---- Panel Spatial Error (by-year stacked W) ----
# Build block-diagonal W: kron(I_T, W_N)
print(f"\n  [Panel SEM — Pooled with year dummies]")

# For panel SEM, we approximate by running GM_Error on pooled data
# (Proper panel SEM would use block-diagonal W, but this gives reasonable approximation)
try:
    # Use time-demeaned data? Or just pooled with year dummies
    X_pooled = np.column_stack([X_all, year_fe])
    pool_labels = X_cols_avail + [f'yr_{i}' for i in range(n_years_panel)]

    # Build block-diag W for pooled data
    from scipy.linalg import block_diag
    W_blocks = []
    for d in all_years_data:
        yr_cities_list = d['cities']
        W_block = W_df.loc[yr_cities_list, yr_cities_list].values
        W_blocks.append(W_block)
    W_pooled = block_diag(*W_blocks)
    w_pooled = full2W(W_pooled)

    sem_panel = GM_Error(y_all, X_pooled, w=w_pooled,
                         name_y=DV, name_x=pool_labels,
                         name_ds='NEV Panel', name_w='Block-Diag W')
    print(f"  Pseudo R² = {sem_panel.pr2:.4f}")
    for i, col in enumerate(X_cols_avail):
        coef = sem_panel.betas[i][0]
        se = sem_panel.std_err[i]
        z_val = sem_panel.z_stat[i][0]
        p_val = sem_panel.z_stat[i][1]
        print(f"  {col:<28s} {coef:>8.4f} {se:>8.4f} {z_val:>8.3f} {p_val:>8.4f}")
except Exception as e:
    print(f"  Panel SEM failed: {e}")

# ---- Panel SAR ----
print(f"\n  [Panel SAR — Pooled with year dummies]")
try:
    sar_panel = GM_Lag(y_all, X_pooled, w=w_pooled,
                       name_y=DV, name_x=pool_labels,
                       name_ds='NEV Panel', name_w='Block-Diag W')
    rho = sar_panel.rho[0] if hasattr(sar_panel.rho, '__len__') else sar_panel.rho
    print(f"  ρ = {rho:.4f} (z={sar_panel.z_stat[0][0]:.3f}, p={sar_panel.z_stat[0][1]:.4f})")
    print(f"  Pseudo R² = {sar_panel.pr2:.4f}")
    for i, col in enumerate(X_cols_avail):
        idx = 1 + i
        coef = sar_panel.betas[idx][0]
        se = sar_panel.std_err[idx]
        z_val = sar_panel.z_stat[idx][0] if idx < len(sar_panel.z_stat) else coef/se
        p_val = sar_panel.z_stat[idx][1] if idx < len(sar_panel.z_stat) else 1.0
        print(f"  {col:<28s} {coef:>8.4f} {se:>8.4f} {z_val:>8.3f} {p_val:>8.4f}")
except Exception as e:
    print(f"  Panel SAR failed: {e}")

# ============================================================
# Part D: SDM Effects Decomposition (2023 cross-section)
# ============================================================
print(f"\n{'='*70}")
print(f"  Part D: SDM Effects Decomposition (2023)")
print(f"{'='*70}")

print(f"\n  Using GM_Combo (Spatial Durbin) for direct/indirect/total effects...")
try:
    combo = GM_Combo(y, X, w=w_reg, name_y=DV, name_x=X_cols_avail,
                     name_ds='NEV SDM', name_w='Inverse Distance')
    rho_val = combo.rho[0] if hasattr(combo.rho, '__len__') else combo.rho
    n_vars = len(X_cols_avail)

    # Extract betas (direct) and W*betas (spatial lag of X)
    # combo.betas = [rho, β1..βk, θ1..θm] where m may differ from k
    n_total_betas = len(combo.betas) - 1  # exclude rho
    n_wx = n_total_betas - n_vars  # number of WX betas included
    beta_direct = np.array([combo.betas[1+i][0] for i in range(n_vars)])
    # WX betas — pad with zeros for variables without spatial lag
    beta_indirect = np.zeros(n_vars)
    for i in range(min(n_wx, n_vars)):
        beta_indirect[i] = combo.betas[1+n_vars+i][0]

    # LeSage & Pace (2009) decomposition:
    # Direct effect = (1/n) * trace(S_k)
    # Total effect = (1/n) * sum(S_k)
    # Indirect = Total - Direct
    # where S_k = (I - ρW)^{-1} * (β_k*I + θ_k*W)

    W_dense_reg = W_reg.copy()
    I_n = np.eye(len(reg_cities))
    # (I - ρW)^{-1}
    try:
        S_mult = np.linalg.inv(I_n - rho_val * W_dense_reg)
    except np.linalg.LinAlgError:
        S_mult = np.linalg.pinv(I_n - rho_val * W_dense_reg)

    effects = []
    for k in range(n_vars):
        S_k = S_mult @ (beta_direct[k] * I_n + beta_indirect[k] * W_dense_reg)
        direct_k = np.trace(S_k) / n
        total_k = S_k.sum() / n
        indirect_k = total_k - direct_k
        effects.append({
            'variable': X_cols_avail[k],
            'direct': direct_k,
            'indirect': indirect_k,
            'total': total_k,
        })

    print(f"  {'Variable':<28s} {'Direct':>10s} {'Indirect':>10s} {'Total':>10s}")
    print(f"  {'-'*60}")
    for e in effects:
        print(f"  {e['variable']:<28s} {e['direct']:>10.4f} {e['indirect']:>10.4f} {e['total']:>10.4f}")

    # Report the spatial spillover ratio
    print(f"\n  Spatial spillover ratios (|Indirect/Direct|):")
    for e in effects:
        ratio = abs(e['indirect'] / e['direct']) if abs(e['direct']) > 0.001 else float('inf')
        print(f"    {e['variable']:<30s} {ratio:.3f}")
except Exception as e:
    print(f"  Effects decomposition failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Part E: Robustness Checks
# ============================================================
print(f"\n{'='*70}")
print(f"  Part E: Robustness Checks")
print(f"{'='*70}")

# E1: Alternative DV — NEV POI count (121-city subsample)
print(f"\n  [E1] Alternative DV: NEV POI count (121 cities)")
if DV_ALT in df_yr.columns:
    poi_data = df_yr.set_index('city')[DV_ALT].dropna()
    poi_cities = [c for c in reg_cities if c in poi_data.index and poi_data[c] > 0]
    if len(poi_cities) >= 50:
        y_poi = poi_data[poi_cities].values
        # Log-transform since count data is skewed
        y_poi = np.log1p(y_poi)

        X_poi_df = df_reg.loc[poi_cities, X_cols_avail].dropna()
        poi_final = [c for c in poi_cities if c in X_poi_df.index]
        y_poi = np.log1p(poi_data[poi_final].values)
        X_poi = X_poi_df.loc[poi_final].values
        X_poi = (X_poi - X_poi.mean(axis=0)) / X_poi.std(axis=0)

        W_poi = W_df.loc[poi_final, poi_final].values
        w_poi = full2W(W_poi, ids=poi_final)

        # Moran's I of POI
        mi_poi = Moran(y_poi, w_poi)
        print(f"  Moran's I (log POI count) = {mi_poi.I:.4f} (p={mi_poi.p_sim:.4f})")
        print(f"  N = {len(poi_final)} cities")

        # OLS with POI as DV
        ols_poi = OLS(y_poi, X_poi, w=w_poi, name_y='log(NEV POI)', name_x=X_cols_avail,
                      name_ds='NEV POI', name_w='W', spat_diag=True, moran=True)
        print(f"  OLS R² = {ols_poi.r2:.4f}")
        for i, col in enumerate(X_cols_avail):
            t_val = ols_poi.t_stat[i][0]
            p_val = 2 * scipy_stats.t.sf(abs(t_val), df=len(y_poi) - len(X_cols_avail) - 1)
            sig = '***' if p_val < 0.01 else ('**' if p_val < 0.05 else ('*' if p_val < 0.10 else ''))
            if p_val < 0.10:
                print(f"    {col:<30s} β={ols_poi.betas[i][0]:>8.4f} (p={p_val:.4f}) {sig}")
        print(f"  Residual Moran's I = {ols_poi.moran_res[0]:.4f} (p={ols_poi.moran_res[1]:.4f})")
else:
    print(f"  {DV_ALT} not available in data")

# E2: Alternative spatial weights — k-NN (k=10)
print(f"\n  [E2] Alternative W: k-Nearest Neighbors (k=10)")
from sklearn.neighbors import NearestNeighbors

# Get coordinates for regression cities
dist_df = pd.read_csv(os.path.join(OUTPUT_DIR, "city_distance_matrix.csv"), index_col=0, encoding='utf-8-sig')
dist_reg = dist_df.loc[reg_cities, reg_cities].values

k = 10
W_knn = np.zeros((len(reg_cities), len(reg_cities)))
for i in range(len(reg_cities)):
    dist_i = dist_reg[i].copy()
    dist_i[i] = np.inf  # exclude self
    nn_idx = np.argpartition(dist_i, k)[:k]
    W_knn[i, nn_idx] = 1.0 / (dist_i[nn_idx] + 1e-6)
    W_knn[i] /= W_knn[i].sum()

np.fill_diagonal(W_knn, 0)
w_knn = full2W(W_knn, ids=reg_cities)

ols_knn = OLS(y, X, w=w_knn, name_y=DV, name_x=X_cols_avail,
              name_ds='NEV (kNN W)', name_w='k-NN (k=10)',
              spat_diag=True, moran=True)
print(f"  OLS R² = {ols_knn.r2:.4f}")
print(f"  Residual Moran's I (k-NN) = {ols_knn.moran_res[0]:.4f} (p={ols_knn.moran_res[1]:.4f})")

# Compare with inverse distance W
print(f"  Reference: Residual Moran's I (inverse dist) = {ols.moran_res[0]:.4f} (p={ols.moran_res[1]:.4f})")

# ============================================================
# Save summary
# ============================================================
print(f"\n{'='*70}")
print(f"  Full Spatial Econometric Analysis — COMPLETE")
print(f"{'='*70}")

# Collect all key results into one summary table
summary_rows = []
# OLS baseline
for i, col in enumerate(X_cols_avail):
    t_val = ols.t_stat[i][0]
    p_val = 2 * scipy_stats.t.sf(abs(t_val), df=n - len(X_cols_avail) - 1)
    summary_rows.append({
        'model': 'OLS (2023)',
        'variable': col,
        'coef': ols.betas[i][0],
        'se': ols.std_err[i],
        'p_value': p_val,
    })
# Panel FE
for i, col in enumerate(X_cols_avail):
    t_val = beta_fe[i] / se_fe[i]
    p_val = 2 * scipy_stats.t.sf(abs(t_val), df=n_total - X_fe.shape[1])
    summary_rows.append({
        'model': 'Panel FE',
        'variable': col,
        'coef': beta_fe[i],
        'se': se_fe[i],
        'p_value': p_val,
    })

summary_df = pd.DataFrame(summary_rows)
summary_out = os.path.join(OUTPUT_DIR, "spatial_regression_summary.csv")
summary_df.to_csv(summary_out, index=False, encoding='utf-8-sig')
print(f"\n  Regression summary saved: {summary_out}")

# Save effects decomposition
if 'effects' in dir() and effects:
    effects_df = pd.DataFrame(effects)
    effects_out = os.path.join(OUTPUT_DIR, "sdm_effects_decomposition.csv")
    effects_df.to_csv(effects_out, index=False, encoding='utf-8-sig')
    print(f"  SDM effects saved: {effects_out}")
