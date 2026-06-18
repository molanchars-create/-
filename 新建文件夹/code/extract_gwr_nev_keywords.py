"""
Extract NEV-related keyword frequencies from city government work reports (2019-2025).
Build a city-year panel of government NEV attention/support indicators.

Data source: D:\EPS与国泰安数据\政府工作报告2019-2025\
File naming: {province}-{city}{year}.txt
"""
import pandas as pd
import numpy as np
import os
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

GWR_DIR = r"D:\EPS与国泰安数据\政府工作报告2019-2025"
OUTPUT_DIR = r"C:\Users\牧原\Desktop\A\工作区\学术研究工作区\06_NEV_产业空间集聚"
PANEL_WIDE = os.path.join(OUTPUT_DIR, "city_panel_wide.csv")
PANEL_LONG = os.path.join(OUTPUT_DIR, "city_panel_long.csv")

# ============================================================
# NEV keyword categories
# ============================================================
NEV_KEYWORDS = {
    'nev_mention': [
        '新能源汽车', '新能源车', '电动汽车', '电动车',
        '智能网联汽车', '智能网联新能源',
    ],
    'nev_charging': [
        '充电桩', '充电设施', '充电站', '充电基础', '换电站', '换电设施',
    ],
    'nev_battery': [
        '动力电池', '锂电池', '锂电', '电池产业', '电池制造',
        '电池材料', '固态电池', '燃料电池',
    ],
    'nev_components': [
        '电机电控', '驱动电机', '汽车零部件', '整车制造',
        '汽车电子', '车规级',
    ],
    'nev_policy': [
        '新能源.*产业', '汽车产业.*新能源', '电动化',
        '车路云', '车联网', '自动驾驶', '智能驾驶',
        '新能源汽车.*集群', '汽车.*产业链',
    ],
    'nev_clean_energy': [
        '光伏', '太阳能', '储能', '清洁能源',
        '新能源装备', '绿色能源',
    ],
}

# Total NEV attention score (simple sum of all categories)
ALL_NEV_TERMS = []
for terms in NEV_KEYWORDS.values():
    ALL_NEV_TERMS.extend(terms)


def count_keywords(text, term_list):
    """Count total occurrences of terms in text using regex"""
    count = 0
    for term in term_list:
        count += len(re.findall(term, text))
    return count


def parse_filename(filename):
    """Extract province, city, year from filename like '安徽省-合肥市2025.txt'"""
    base = filename.replace('.txt', '')
    # Split on last occurrence of 4-digit year
    match = re.match(r'(.+)-(.+?)(\d{4})$', base)
    if match:
        province = match.group(1)
        city = match.group(2)
        year = int(match.group(3))
        return province, city, year
    return None, None, None


# ============================================================
# Process all reports
# ============================================================
print("=" * 70)
print("  Extracting NEV Keywords from Government Work Reports")
print("=" * 70)

records = []
errors = []
total_files = 0

for year_dir in sorted(os.listdir(GWR_DIR)):
    year_path = os.path.join(GWR_DIR, year_dir)
    if not os.path.isdir(year_path):
        continue

    txt_files = [f for f in os.listdir(year_path) if f.endswith('.txt')]
    total_files += len(txt_files)

    for filename in txt_files:
        province, city, year = parse_filename(filename)
        if city is None:
            errors.append(f"Parse error: {filename}")
            continue

        filepath = os.path.join(year_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            try:
                with open(filepath, 'r', encoding='gbk') as f:
                    text = f.read()
            except Exception as e:
                errors.append(f"Encoding error: {filename}: {e}")
                continue

        rec = {
            'province': province,
            'city': city,
            'year': year,
            'text_length': len(text),
        }

        # Count each keyword category
        for cat_name, term_list in NEV_KEYWORDS.items():
            rec[cat_name] = count_keywords(text, term_list)

        # Total NEV attention score
        rec['nev_total_mentions'] = sum(rec[cat] for cat in NEV_KEYWORDS)

        records.append(rec)

print(f"\n  Total files found: {total_files}")
print(f"  Successfully processed: {len(records)}")
if errors:
    print(f"  Errors: {len(errors)}")
    for e in errors[:10]:
        print(f"    {e}")

# ============================================================
# Build city-year panel
# ============================================================
gwr_df = pd.DataFrame(records)
print(f"\n  Year coverage:")
for yr in sorted(gwr_df['year'].unique()):
    n_cities = len(gwr_df[gwr_df['year'] == yr])
    n_nev = (gwr_df[gwr_df['year'] == yr]['nev_total_mentions'] > 0).sum()
    avg = gwr_df[gwr_df['year'] == yr]['nev_total_mentions'].mean()
    print(f"    {yr}: {n_cities} cities, {n_nev} mention NEV, avg mentions={avg:.1f}")

# Top cities by NEV mention (average across years)
city_avg = gwr_df.groupby('city')['nev_total_mentions'].agg(['mean', 'count'])
city_avg = city_avg[city_avg['count'] >= 3].sort_values('mean', ascending=False)
print(f"\n  Top 20 cities by avg NEV mentions (≥3 years):")
for city, row in city_avg.head(20).iterrows():
    print(f"    {city:<10s} mean={row['mean']:.1f}  (n={int(row['count'])})")

# ============================================================
# Merge with panel
# ============================================================
panel = pd.read_csv(PANEL_WIDE, encoding='utf-8-sig')

# Check city name match
gwr_cities = set(gwr_df['city'].unique())
panel_cities = set(panel['city'].unique())
common = gwr_cities & panel_cities
print(f"\n  GWR cities: {len(gwr_cities)}")
print(f"  Panel cities: {len(panel_cities)}")
print(f"  Common: {len(common)}")

only_gwr = gwr_cities - panel_cities
if only_gwr:
    print(f"  Only in GWR ({len(only_gwr)}): {sorted(only_gwr)[:20]}")

only_panel = panel_cities - gwr_cities
if only_panel:
    print(f"  Only in panel ({len(only_panel)}): {sorted(only_panel)[:20]}")

# Merge: match on city + year
# First, align city names — try with/without 市 suffix
gwr_df['city_clean'] = gwr_df['city']

# Build lookup for cities that don't match
for gwr_city in list(only_gwr):
    # Try adding/removing 市
    if gwr_city.endswith('市'):
        alt = gwr_city[:-1]
    else:
        alt = gwr_city + '市'
    if alt in panel_cities:
        gwr_df.loc[gwr_df['city'] == gwr_city, 'city_clean'] = alt

# Re-check match
gwr_cities_clean = set(gwr_df['city_clean'].unique())
common2 = gwr_cities_clean & panel_cities
still_unmapped = gwr_cities_clean - panel_cities
print(f"  After 市 fix: {len(common2)} common, {len(still_unmapped)} still unmapped")
if still_unmapped:
    print(f"  Still unmapped: {sorted(still_unmapped)[:20]}")

# Merge GWR indicators into panel
# For each year, map city -> nev indicator values
for yr in sorted(gwr_df['year'].unique()):
    yr_gwr = gwr_df[gwr_df['year'] == yr]
    yr_mask = panel['year'] == yr

    for cat in list(NEV_KEYWORDS.keys()) + ['nev_total_mentions']:
        col_name = f'gwr_{cat}'
        mapping = dict(zip(yr_gwr['city_clean'], yr_gwr[cat]))
        panel.loc[yr_mask, col_name] = panel.loc[yr_mask, 'city'].map(mapping)

# Fill missing with 0 (cities without reports = no mention found)
for cat in list(NEV_KEYWORDS.keys()) + ['nev_total_mentions']:
    col_name = f'gwr_{cat}'
    panel[col_name] = panel[col_name].fillna(0).astype(int)

print(f"\n  Panel columns added: gwr_nev_mention, gwr_nev_charging, ...")
for yr in sorted(panel['year'].unique()):
    yr_df = panel[panel['year'] == yr]
    nn = (yr_df['gwr_nev_total_mentions'] > 0).sum()
    total = len(yr_df)
    avg = yr_df['gwr_nev_total_mentions'].mean()
    print(f"    {yr}: {nn}/{total} cities mention NEV, avg={avg:.1f}")

# ============================================================
# Save
# ============================================================

# Save GWR keyword data separately
gwr_out = os.path.join(OUTPUT_DIR, "gwr_nev_keywords.csv")
gwr_df.to_csv(gwr_out, index=False, encoding='utf-8-sig')
print(f"\n  GWR keywords saved: {gwr_out}")

# Update panel
panel.to_csv(PANEL_WIDE, index=False, encoding='utf-8-sig')
print(f"  Panel updated: {PANEL_WIDE}")

# Update long panel
panel_long = pd.read_csv(PANEL_LONG, encoding='utf-8-sig')

gwr_long_rows = []
indicator_labels = {
    'gwr_nev_mention': 'GWR-新能源汽车提及次数',
    'gwr_nev_charging': 'GWR-充电设施提及次数',
    'gwr_nev_battery': 'GWR-电池产业提及次数',
    'gwr_nev_components': 'GWR-汽车零部件提及次数',
    'gwr_nev_policy': 'GWR-NEV政策提及次数',
    'gwr_nev_clean_energy': 'GWR-清洁能源提及次数',
    'gwr_nev_total_mentions': 'GWR-NEV总提及次数',
}

for _, r in panel.iterrows():
    for col, label in indicator_labels.items():
        if col in panel.columns and pd.notna(r.get(col)):
            gwr_long_rows.append({
                'city': r['city'],
                'year': r['year'],
                'indicator': label,
                'value': int(r[col]),
            })

if gwr_long_rows:
    gwr_long_df = pd.DataFrame(gwr_long_rows)
    panel_long = pd.concat([panel_long, gwr_long_df], ignore_index=True)
    panel_long.to_csv(PANEL_LONG, index=False, encoding='utf-8-sig')
    print(f"  Long panel updated: {PANEL_LONG} ({len(panel_long)} rows)")

# ============================================================
print(f"\n{'='*70}")
print(f"  GWR NEV Keyword Extraction Summary")
print(f"{'='*70}")
print(f"  Reports processed: {len(records)}")
print(f"  Years: {sorted(gwr_df['year'].unique())}")
print(f"  Cities covered: {len(gwr_cities)}")
print(f"  Keywords: {len(ALL_NEV_TERMS)} terms in {len(NEV_KEYWORDS)} categories")
print(f"  Indicators added to panel: {len(indicator_labels)}")
print(f"{'='*70}")
