"""
Post-process NEV POI data:
1. Load raw POI collection
2. Remove noise/false positives (non-NEV businesses caught by keyword matching)
3. Classify POIs by NEV industry chain segment
4. Compute city-level NEV agglomeration indices:
   - NEV firm count per city
   - NEV firm density (per km² or per capita)
   - Location quotient (LQ) for NEV sectors
5. Merge into city panel
"""
import pandas as pd
import numpy as np
import os
import glob
import sys
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = r"C:\Users\牧原\Desktop\A\工作区\学术研究工作区\06_NEV_产业空间集聚"
PANEL_WIDE = os.path.join(OUTPUT_DIR, "city_panel_wide.csv")
PANEL_LONG = os.path.join(OUTPUT_DIR, "city_panel_long.csv")

# Find latest POI collection file
poi_files = glob.glob(os.path.join(OUTPUT_DIR, "nev_poi_collection_*.csv"))
if not poi_files:
    print("ERROR: No POI collection files found!")
    sys.exit(1)

latest_poi = sorted(poi_files)[-1]
print(f"Loading: {os.path.basename(latest_poi)}")

df = pd.read_csv(latest_poi, encoding='utf-8-sig')
print(f"Raw POIs: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nChain segments:")
print(df['chain_segment'].value_counts().to_string())

# ============================================================
# STEP 1: Remove noise and false positives
# ============================================================

# Keywords that indicate non-NEV businesses caught by broad keyword matching
NOISE_KEYWORDS = [
    # Non-NEV manufacturing
    '卷帘门', '雨棚', '防盗门', '铝合金门窗', '塑钢', '彩钢',
    '不锈钢加工', '铁艺', '焊接', '钣金', '五金加工',
    # Non-NEV vehicles
    '电动自行车', '电瓶车', '三轮车', '老年代步车', '观光车',
    '叉车', '铲车', '挖掘机', '拖拉机', '农用车',
    # Non-NEV charging (phone/electronics charging)
    '手机充电', '充电宝', '数据线',
    # Non-NEV solar
    '太阳能热水器', '太阳能路灯',
    # General/irrelevant
    '废品回收', '二手车', '汽车美容', '洗车', '汽车租赁',
    '驾校', '停车场', '加油站',
]

def is_noise(name, poi_type, chain):
    """Check if a POI is likely noise"""
    name_str = str(name).lower()
    type_str = str(poi_type).lower()

    for kw in NOISE_KEYWORDS:
        if kw in name_str or kw in type_str:
            return True
    return False

# Apply noise filter
before = len(df)
df['is_noise'] = df.apply(lambda r: is_noise(r.get('name', ''), r.get('poi_type', ''), r.get('chain_segment', '')), axis=1)
df_clean = df[~df['is_noise']].copy()
removed = before - len(df_clean)
print(f"\n  Noise removed: {removed} ({100*removed/before:.1f}%)")
print(f"  Clean POIs: {len(df_clean)}")

# ============================================================
# STEP 2: Reclassify and verify chain segments
# ============================================================

# More specific classification rules
# Map raw search categories to clean chain positions
CHAIN_POSITION_MAP = {
    '整车制造': '整车制造',
    '电池/上游': '电池及上游材料',
    '充电设施': '充电基础设施',
    '电机电控': '电机电控',
    '汽车零部件': '汽车零部件',
    '零部件': '汽车零部件',
    'NEV销售': '销售与服务',
    'NEV维修': '销售与服务',
    '销售服务': '销售与服务',
    '太阳能设备': '清洁能源装备',
    '清洁能源': '清洁能源装备',
}

df_clean['chain_position'] = df_clean['chain_segment'].map(CHAIN_POSITION_MAP)

print(f"\nChain positions after reclassification:")
print(df_clean['chain_position'].value_counts().to_string())

# ============================================================
# STEP 3: City-level aggregation
# ============================================================

# Count POIs by city and chain position
city_poi = df_clean.groupby(['city', 'chain_position']).size().unstack(fill_value=0)
city_poi['total_nev_poi'] = city_poi.sum(axis=1)

print(f"\nCity-level POI counts:")
print(f"  Cities with POIs: {len(city_poi)}")
print(f"  Total POIs after cleaning: {int(city_poi['total_nev_poi'].sum())}")
print(f"\n  Top 15 cities by NEV POI count:")
top15 = city_poi.nlargest(15, 'total_nev_poi')
for city, row in top15.iterrows():
    parts = []
    for col in city_poi.columns:
        if col != 'total_nev_poi' and row[col] > 0:
            parts.append(f"{col}={int(row[col])}")
    print(f"    {city:<10s} {int(row['total_nev_poi']):>5d}  ({', '.join(parts)})")

# ============================================================
# STEP 4: NEV Agglomeration Indices
# ============================================================

# Load panel for city-level economic data
panel = pd.read_csv(PANEL_WIDE, encoding='utf-8-sig')

# Merge POI counts with panel
# Harmonize city names: POI uses 北京市/上海市, panel uses 北京/上海
CITY_RENAME = {'北京市': '北京', '上海市': '上海', '天津市': '天津'}
city_poi = city_poi.rename(index=CITY_RENAME)
df_clean['city'] = df_clean['city'].replace(CITY_RENAME)

poi_cities = set(city_poi.index)
panel_cities = set(panel['city'].unique())
common = poi_cities & panel_cities
print(f"\n  POI cities: {len(poi_cities)}, Panel cities: {len(panel_cities)}")
print(f"  Common: {len(common)}")
only_poi = poi_cities - panel_cities
if only_poi:
    print(f"  Only in POI ({len(only_poi)}): {sorted(only_poi)}")

# Compute location quotient for NEV industries
# LQ_ci = (NEV_POI_ci / total_POI_c) / (NEV_POI_national / total_POI_national)
total_nev_national = city_poi['total_nev_poi'].sum()

# Also compute POI density (POIs per 10,000 population)
# and POI per 1000 km² area

for yr in sorted(panel['year'].unique()):
    yr_mask = panel['year'] == yr

    # Map total NEV POI count
    panel.loc[yr_mask, 'nev_poi_count'] = panel.loc[yr_mask, 'city'].map(
        lambda c: city_poi.loc[c, 'total_nev_poi'] if c in city_poi.index else 0
    )

    # Map chain-specific counts
    for pos in city_poi.columns:
        if pos != 'total_nev_poi':
            col_name = f'nev_poi_{pos}'
            panel.loc[yr_mask, col_name] = panel.loc[yr_mask, 'city'].map(
                lambda c, p=pos: city_poi.loc[c, p] if c in city_poi.index and p in city_poi.columns else 0
            )

    # POI density per 10,000 population
    pop = panel.loc[yr_mask, '常住人口（万人）']
    panel.loc[yr_mask, 'nev_poi_density_pop'] = panel.loc[yr_mask, 'nev_poi_count'] / pop.replace(0, np.nan)

print(f"\n  NEV POI count by year (same across years — static POI snapshot):")
for yr in sorted(panel['year'].unique()):
    yr_df = panel[panel['year'] == yr]
    nonzero = (yr_df['nev_poi_count'] > 0).sum()
    total = len(yr_df)
    mean_count = yr_df['nev_poi_count'].mean()
    print(f"    {yr}: {nonzero}/{total} cities have NEV POIs, mean={mean_count:.1f}")

# ============================================================
# STEP 5: Save processed data
# ============================================================

# Save clean POI dataset
poi_clean_file = os.path.join(OUTPUT_DIR, "nev_poi_clean.csv")
df_clean.to_csv(poi_clean_file, index=False, encoding='utf-8-sig')
print(f"\n  Clean POIs saved: {poi_clean_file}")

# Save city-level POI statistics
poi_stats_file = os.path.join(OUTPUT_DIR, "nev_poi_city_stats.csv")
city_poi.to_csv(poi_stats_file, encoding='utf-8-sig')
print(f"  City POI stats saved: {poi_stats_file}")

# Update panel
panel.to_csv(PANEL_WIDE, index=False, encoding='utf-8-sig')
print(f"  Panel updated: {PANEL_WIDE}")

# Add to long panel
panel_long = pd.read_csv(PANEL_LONG, encoding='utf-8-sig')
new_rows = []
for _, r in panel.iterrows():
    for col in ['nev_poi_count', 'nev_poi_density_pop']:
        if col in panel.columns and pd.notna(r.get(col)):
            new_rows.append({'city': r['city'], 'year': r['year'],
                           'indicator': col, 'value': r[col]})

if new_rows:
    new_df = pd.DataFrame(new_rows)
    panel_long = pd.concat([panel_long, new_df], ignore_index=True)
    panel_long.to_csv(PANEL_LONG, index=False, encoding='utf-8-sig')
    print(f"  Long panel updated: {PANEL_LONG} ({len(panel_long)} rows)")

# ============================================================
print(f"\n{'='*70}")
print(f"  NEV POI Processing Summary")
print(f"{'='*70}")
print(f"  Raw POIs: {before}")
print(f"  Noise removed: {removed} ({100*removed/before:.1f}%)")
print(f"  Clean POIs: {len(df_clean)}")
print(f"  Cities with NEV POIs: {len(city_poi)}")
print(f"  Chain positions: {list(df_clean['chain_position'].unique())}")
print(f"\n  NOTE: POI data is a cross-sectional snapshot.")
print(f"  For panel analysis, need POI data by year or use")
print(f"  cross-sectional agglomeration model.")
print(f"{'='*70}")
