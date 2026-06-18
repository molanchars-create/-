"""
Spatial Econometric Analysis V2 — Theory-Consistent Specification
=================================================================
Maps directly to theoretical hypotheses H1-H5:
  Y = ρWY + β₁·Link + β₂·Sub + β₃·(Link×Sub) + θ₁·W·Link + θ₂·W·Sub + γX + FE

DV options:
  A) log(1 + nev_poi_count) — actual firm agglomeration (N≈121)
  B) gwr_nev_nonpolicy = gwr_nev_total - gwr_nev_policy (N≈284)

Key mapping:
  λ_M (M-firm share)     ↔ DV (NEV agglomeration)
  s_r (subsidy)          ↔ gwr_nev_policy (policy attention)
  P_{B,r} (battery price) ↔ 1/supplier_access_log (inverse relationship)
  Link × Sub interaction ↔ β₃ < 0 (H3: substitution effect)
"""
import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

from libpysal.weights import full2W
from scipy import stats as scipy_stats
import spreg
from spreg import OLS, GM_Lag, GM_Error, GM_Combo

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(OUTPUT_DIR, "数据文件")
PANEL_WIDE = os.path.join(DATA_DIR, "city_panel_wide.csv")
W_MATRIX = os.path.join(DATA_DIR, "spatial_weights_W.csv")

# ============================================================
# Load data
# ============================================================
print("=" * 70)
print("  Spatial Econometrics V2 — Theory-Consistent SDM")
print("=" * 70)

panel = pd.read_csv(PANEL_WIDE, encoding='utf-8-sig')
W_df = pd.read_csv(W_MATRIX, index_col=0, encoding='utf-8-sig')
W_cities_all = list(W_df.columns)

YEAR = 2023
df = panel[panel['year'] == YEAR].copy()

# ---- Construct variables ----
# DV-A: log NEV POI count (firm agglomeration)
df['log_nev_poi'] = np.log1p(df['nev_poi_count'])

# DV-B: non-policy NEV mentions (government attention minus policy)
df['gwr_nev_nonpolicy'] = df['gwr_nev_total_mentions'] - df['gwr_nev_policy']

# Sub: policy mentions intensity (proxy for local government NEV subsidy/promotion)
df['subsidy_policy'] = df['gwr_nev_policy']

# Link: supplier access (log) — higher SA = lower effective P_B
df['link_supplier'] = df['supplier_access_log']

# Link × Sub interaction (de-meaned to reduce multicollinearity)
link_mean = df['link_supplier'].mean()
sub_mean = df['subsidy_policy'].mean()
df['link_x_sub'] = (df['link_supplier'] - link_mean) * (df['subsidy_policy'] - sub_mean)

# Controls
df['pop_log'] = np.log1p(df['常住人口（万人）'])
df['gdp_pc_log'] = np.log1p(df['人均地区生产总值（元）'])

print(f"\n  Year: {YEAR}")
print(f"  Total cities in panel: {len(df)}")

# ============================================================
# Run models for both DV specifications
# ============================================================

for dv_spec in ['A', 'B']:
    if dv_spec == 'A':
        dv_name = 'log_nev_poi'
        dv_label = 'log(1+NEV_POI)'
        # For POI, also include market_access as control
        x_vars = ['link_supplier', 'subsidy_policy', 'link_x_sub',
                   'market_access_log', 'pop_log', 'gdp_pc_log']
        x_labels = ['Link(SA)', 'Sub(Policy)', 'Link×Sub', 'MA', 'ln(Pop)', 'ln(GDPpc)']
    else:
        dv_name = 'gwr_nev_nonpolicy'
        dv_label = 'GWR_NonPolicy'
        x_vars = ['link_supplier', 'subsidy_policy', 'link_x_sub',
                   'market_access_log', 'terrain_ruggedness', 'pop_log', 'gdp_pc_log']
        x_labels = ['Link(SA)', 'Sub(Policy)', 'Link×Sub', 'MA', 'Ruggedness', 'ln(Pop)', 'ln(GDPpc)']

    print(f"\n{'='*70}")
    print(f"  Specification {dv_spec}: DV = {dv_label}")
    print(f"{'='*70}")

    # Prepare regression sample
    cols_needed = [dv_name] + x_vars
    df_reg = df[['city'] + cols_needed].dropna()

    # Intersect with W matrix cities
    avail = sorted(set(W_cities_all) & set(df_reg['city'].values))
    df_reg = df_reg.set_index('city').loc[avail].reset_index()
    reg_cities = df_reg['city'].values.tolist()
    n = len(reg_cities)
    print(f"  N = {n} cities")

    if n < 80:
        print(f"  [SKIP] Too few cities for spatial model")
        continue

    # Build W
    W_reg = W_df.loc[reg_cities, reg_cities].values
    w_reg = full2W(W_reg, ids=reg_cities)

    # Extract y, X (standardized)
    y = df_reg[dv_name].values.astype(float)
    X_raw = df_reg[x_vars].values.astype(float)
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X_std[X_std == 0] = 1.0
    X = (X_raw - X_mean) / X_std

    # ---- Descriptive ----
    print(f"\n  Descriptives:")
    print(f"  {'Variable':<20s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
    print(f"  {'-'*52}")
    for col, label in [(dv_name, dv_label)] + list(zip(x_vars, x_labels)):
        s = df_reg[col]
        print(f"  {label:<20s} {s.mean():>8.3f} {s.std():>8.3f} {s.min():>8.3f} {s.max():>8.3f}")

    # ---- OLS Baseline ----
    ols = OLS(y, X, w=w_reg, name_y=dv_label, name_x=x_vars,
              name_ds='NEV', name_w='InvDist', spat_diag=True, moran=True)
    print(f"\n  [OLS Baseline]")
    print(f"  R² = {ols.r2:.4f}, Adj R² = {ols.ar2:.4f}")
    print(f"  Moran's I (residuals) = {ols.moran_res[0]:.4f} (p={ols.moran_res[1]:.4f})")

    k = len(x_vars)
    print(f"  {'Variable':<20s} {'Coef':>8s} {'Std.Err':>8s} {'t':>8s} {'p':>8s} {'Sig':>5s}")
    print(f"  {'-'*62}")
    for i in range(k):
        coef = ols.betas[i][0]
        se = ols.std_err[i]
        t_val = ols.t_stat[i][0]
        p_val = 2 * scipy_stats.t.sf(abs(t_val), df=n - k - 1)
        sig = '***' if p_val < 0.01 else ('**' if p_val < 0.05 else ('*' if p_val < 0.10 else ''))
        print(f"  {x_vars[i]:<20s} {coef:>8.4f} {se:>8.4f} {t_val:>8.3f} {p_val:>8.4f} {sig:>5s}")

    # ---- GM_Combo (SDM approximation) ----
    print(f"\n  [Spatial Durbin Model — GM_Combo]")
    try:
        combo = GM_Combo(y, X, w=w_reg, name_y=dv_label, name_x=x_vars,
                         name_ds='NEV SDM', name_w='InvDist')
        rho_val = combo.rho[0] if hasattr(combo.rho, '__len__') else combo.rho
        n_betas = len(combo.betas) - 1  # exclude rho
        n_wx = n_betas - k  # number of WX coefficients

        # Extract direct and indirect betas
        beta_direct = np.array([combo.betas[1+i][0] for i in range(k)])
        beta_indirect = np.zeros(k)
        for i in range(min(n_wx, k)):
            beta_indirect[i] = combo.betas[1+k+i][0]

        # Spatial multiplier
        W_dense = W_reg.copy()
        I_n = np.eye(n)
        try:
            S_mult = np.linalg.inv(I_n - rho_val * W_dense)
        except np.linalg.LinAlgError:
            S_mult = np.linalg.pinv(I_n - rho_val * W_dense)

        # LeSage-Pace effects decomposition
        print(f"  ρ = {rho_val:.4f}")
        print(f"  {'Variable':<20s} {'Direct':>10s} {'Indirect':>10s} {'Total':>10s}")
        print(f"  {'-'*52}")
        effects = []
        for i in range(k):
            S_k = S_mult @ (beta_direct[i] * I_n + beta_indirect[i] * W_dense)
            direct_k = np.trace(S_k) / n
            total_k = S_k.sum() / n
            indirect_k = total_k - direct_k
            effects.append({
                'variable': x_vars[i], 'direct': direct_k,
                'indirect': indirect_k, 'total': total_k
            })
            print(f"  {x_vars[i]:<20s} {direct_k:>10.4f} {indirect_k:>10.4f} {total_k:>10.4f}")

        # ---- Hypothesis Testing ----
        print(f"\n  {'='*60}")
        print(f"  HYPOTHESIS TESTS (Theory vs Empirics)")
        print(f"  {'='*60}")

        # H1: Forward Linkage — β_Link > 0
        link_idx = x_vars.index('link_supplier')
        link_total = effects[link_idx]['total']
        link_direct = effects[link_idx]['direct']
        print(f"\n  H1 (Forward Linkage): ∂Y/∂Link > 0")
        print(f"    Theory predicts: β₁ > 0 (better supplier access → more agglomeration)")
        print(f"    Empirical: Direct={link_direct:.4f}, Total={link_total:.4f}")
        if link_total > 0:
            print(f"    => CONSISTENT with theory (total effect positive)")
        else:
            print(f"    => INCONSISTENT with theory (total effect negative)")

        # H2: Subsidy Main Effect — β_Sub > 0
        sub_idx = x_vars.index('subsidy_policy')
        sub_total = effects[sub_idx]['total']
        sub_direct = effects[sub_idx]['direct']
        print(f"\n  H2 (Subsidy Main Effect): ∂Y/∂Sub > 0")
        print(f"    Theory predicts: β₂ > 0 (higher subsidy → more agglomeration)")
        print(f"    Empirical: Direct={sub_direct:.4f}, Total={sub_total:.4f}")
        if sub_total > 0:
            print(f"    => CONSISTENT with theory")
        else:
            print(f"    => INCONSISTENT with theory")

        # H3: Substitution Effect — β_Link×Sub < 0
        inter_idx = x_vars.index('link_x_sub')
        inter_total = effects[inter_idx]['total']
        inter_direct = effects[inter_idx]['direct']
        print(f"\n  H3 (Substitution/Interaction): ∂²Y/(∂Link ∂Sub) < 0")
        print(f"    Theory predicts: β₃ < 0 (subsidy reduces dependence on local suppliers)")
        print(f"    Structural model: ∂²Π/(∂P_B ∂s) > 0 → Link ∝ 1/P_B → β₃ < 0")
        print(f"    Empirical: Direct={inter_direct:.4f}, Total={inter_total:.4f}")
        if inter_total < 0:
            print(f"    => CONSISTENT with theory (negative interaction)")
        else:
            print(f"    => INCONSISTENT with theory (positive interaction)")

        # H5: Spatial Spillover — θ_Sub < 0 (competitive)
        if 'subsidy_policy' in x_vars and n_wx >= k:
            sub_wx_idx = min(sub_idx, n_wx - 1)
            sub_wx = beta_indirect[sub_idx]
            print(f"\n  H5 (Spatial Competition): θ_Sub < 0")
            print(f"    Theory predicts: neighboring subsidy → competitive pressure → negative spillover")
            print(f"    Empirical: θ_Sub (W·Sub) = {sub_wx:.4f}")
            if sub_wx < 0:
                print(f"    => CONSISTENT with theory")
            else:
                print(f"    => INCONSISTENT (or positive synergy dominates)")

        # Save effects
        effects_df = pd.DataFrame(effects)
        effects_out = os.path.join(DATA_DIR, f"sdm_effects_v2_{dv_spec}.csv")
        effects_df.to_csv(effects_out, index=False, encoding='utf-8-sig')

    except Exception as e:
        print(f"  SDM failed: {e}")
        import traceback
        traceback.print_exc()

# ============================================================
# Summary: Theory-vs-Empirics Mapping Table
# ============================================================
print(f"\n{'='*70}")
print(f"  THEORY-EMPIRICS QUALITATIVE COMPARISON")
print(f"{'='*70}")
print(f"""
  Theoretical Model                    Empirical Counterpart
  ───────────────────────────────────────────────────────────
  λ_M (M-firm location share)          log(1+NEV_POI_count) [Spec A]
                                        GWR non-policy mentions [Spec B]

  s_r (local subsidy rate)             gwr_nev_policy (GWR policy keyword count)

  P_{{B,r}} (battery price index)       1 / supplier_access_log
     (lower P_B = better vertical link → higher SA)

  n_{{B,r}} (B-firm count)              supplier_access_log (VL-weighted supplier density)

  Cross-partial H > 0                  β₃ < 0 in Link×Sub interaction
  (subsidy ↓ sensitivity to P_B)       (subsidy ↓ marginal effect of Link on Y)

  Bifurcation / hysteresis             Spatial regime switching?
  (τ_B break ≠ sustain)               (spatial heterogeneity: East vs West)
""")

print("=" * 70)
print("  Analysis Complete")
print("=" * 70)
