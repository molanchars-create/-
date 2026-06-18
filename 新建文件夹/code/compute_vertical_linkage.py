"""
Compute Vertical Linkage Index from CEADS MRIO tables (2015, 2017, 2018, 2020)

Vertical linkage (Venables 1996):
  VL_{ir} = 0.5 * (BL_{ir} + FL_{ir})

  BL_j = sum_{i,r} a_{ij}^{rs}  — backward linkage (column sum of A)
  FL_i = sum_{j,s} a_{ij}^{rs}  — forward linkage (row sum of A)
  a_{ij}^{rs} = z_{ij}^{rs} / x_j^s

MRIO structure (CEADS format):
  - 4 header rows (0: title, 1: province+sector, 2: sector codes, 3: sector codes again)
  - Data starts at row 4
  - Col 0: province (needs forward-fill), Col 1: sector name, Col 2: sector code
  - Cols 3..1304: intermediate use (31 prov × 42 sectors)
  - Cols 1305+: final demand
"""
import pandas as pd
import numpy as np
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

BASE = r"D:\EPS与国泰安数据"
OUTPUT_DIR = r"C:\Users\牧原\Desktop\A\工作区\学术研究工作区\06_NEV_产业空间集聚"

MRIO_FILES = {
    2015: ("6128b91470f926be0.xlsx", "Table"),
    2017: ("63b4aaf9401322060.xlsx", "Table_2017"),
    2018: ("MRIO 2018.xlsx", "Table_2018"),
    2020: ("MRIO 2020.xlsx", "Table_2020"),
}

N_PROVINCES = 31
N_SECTORS = 42
N_INTERMEDIATE = N_PROVINCES * N_SECTORS  # 1302

PROVINCE_NAMES = {'北京','天津','河北','山西','内蒙古','辽宁','吉林','黑龙江',
                  '上海','江苏','浙江','安徽','福建','江西','山东','河南',
                  '湖北','湖南','广东','广西','海南','重庆','四川','贵州',
                  '云南','西藏','陕西','甘肃','青海','宁夏','新疆'}

# Also accept "内蒙" as variant of "内蒙古"
PROVINCE_ALIASES = {'内蒙': '内蒙古'}

# NEV-related sectors
NEV_SECTORS = {
    12: '化学产品',
    14: '金属冶炼和压延加工品',
    15: '金属制品',
    16: '通用设备',
    17: '专用设备',
    18: '交通运输设备',
    19: '电气机械和器材',
    20: '通信设备/计算机/电子',
    29: '交通运输/仓储/邮政',
}

SECTOR_NAMES_CN = [
    '农林牧渔产品和服务', '煤炭采选产品', '石油和天然气开采产品', '金属矿采选产品',
    '非金属矿和其他矿采选产品', '食品和烟草', '纺织品', '纺织服装鞋帽皮革羽绒及其制品',
    '木材加工品和家具', '造纸印刷和文教体育用品', '石油、炼焦产品和核燃料加工品',
    '化学产品', '非金属矿物制品', '金属冶炼和压延加工品', '金属制品',
    '通用设备', '专用设备', '交通运输设备', '电气机械和器材',
    '通信设备、计算机和其他电子设备', '仪器仪表', '其他制造产品',
    '金属制品、机械和设备修理服务', '电力、热力的生产和供应', '燃气生产和供应',
    '水的生产和供应', '建筑', '批发和零售', '交通运输、仓储和邮政',
    '住宿和餐饮', '信息传输、软件和信息技术服务', '金融', '房地产',
    '租赁和商务服务', '科学研究', '技术服务', '水利、环境和公共设施管理',
    '居民服务、修理和其他服务', '教育', '卫生和社会工作',
    '文化、体育和娱乐', '公共管理、社会保障和社会组织'
]


def parse_mrio_table(filepath, sheet_name):
    """Parse CEADS MRIO: return Z, A, row_info, col_info, total_output"""
    df = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
    n_rows, n_cols = df.shape

    # Header rows:
    #   Row 0: title
    #   Row 1: province names (merged, one per 42-col block) + NaN for rest
    #   Row 2: sector NAMES (农林牧渔产品和服务, 煤炭采选产品, ...)
    #   Row 3: sector CODES (01, 02, 03, ...)
    #   Row 4+: data
    DATA_START = 4

    # ---- Column mapping ----
    header_r1 = df.iloc[1, :].copy()  # Province names (only at block starts)
    header_r2 = df.iloc[2, :].copy()  # Sector names
    header_r3 = df.iloc[3, :].copy()  # Sector codes

    col_province = []
    col_sector_code = []
    col_sector_name = []

    # Forward-fill province across each 42-col block
    current_prov = None
    for j in range(3, 3 + N_INTERMEDIATE):
        val = header_r1.iloc[j]
        if pd.notna(val):
            val_str = str(val).strip()
            if val_str in PROVINCE_NAMES or val_str in PROVINCE_ALIASES:
                current_prov = PROVINCE_ALIASES.get(val_str, val_str)

        col_province.append(current_prov)

        # Sector name from row 2, sector code from row 3
        sname = str(header_r2.iloc[j]).strip() if pd.notna(header_r2.iloc[j]) else ''
        scode = str(header_r3.iloc[j]).strip().zfill(2) if pd.notna(header_r3.iloc[j]) else ''
        col_sector_code.append(scode)
        col_sector_name.append(sname)

    # ---- Row data extraction ----
    data = df.iloc[DATA_START:, :].copy()
    data.columns = range(n_cols)

    # Forward-fill province names in col 0
    last_prov = None
    prov_filled = []
    for i_idx in range(len(data)):
        v = data.iloc[i_idx, 0]
        if pd.notna(v):
            vs = str(v).strip()
            if vs in PROVINCE_NAMES or vs in PROVINCE_ALIASES:
                last_prov = PROVINCE_ALIASES.get(vs, vs)
        prov_filled.append(last_prov)

    # Build mask: only rows that have valid province + numeric data
    valid_mask = []
    row_provinces = []
    row_sector_codes = []
    row_sector_names = []

    for i_idx in range(len(data)):
        prov = prov_filled[i_idx]
        if prov is None:
            valid_mask.append(False)
            continue

        sec_name = str(data.iloc[i_idx, 1]).strip() if pd.notna(data.iloc[i_idx, 1]) else ''
        sec_code = str(data.iloc[i_idx, 2]).strip().zfill(2) if pd.notna(data.iloc[i_idx, 2]) else ''

        # Check if this looks like a data row (has numeric value in col 3)
        try:
            float(data.iloc[i_idx, 3])
            is_data = True
        except (ValueError, TypeError):
            is_data = False

        if is_data and len(sec_code) >= 1:
            valid_mask.append(True)
            row_provinces.append(prov)
            row_sector_codes.append(sec_code)
            row_sector_names.append(sec_name)
        else:
            valid_mask.append(False)

    # Filter to valid data rows
    valid_indices = [i for i, m in enumerate(valid_mask) if m]
    data_valid = data.iloc[valid_indices].copy()

    print(f"    Valid data rows: {len(data_valid)} (expected ~{N_INTERMEDIATE})")

    # ---- Extract Z and final demand ----
    Z_cols = list(range(3, 3 + N_INTERMEDIATE))
    Z = data_valid[Z_cols].values.astype(float)

    fd_start = 3 + N_INTERMEDIATE
    # Final demand cols: last few columns, skip any that are all-nan or non-numeric
    fd_data = []
    for j in range(fd_start, n_cols):
        try:
            col_vals = pd.to_numeric(data_valid[j], errors='coerce').fillna(0).values
            fd_data.append(col_vals)
        except:
            pass

    if fd_data:
        FD = np.column_stack(fd_data)
    else:
        FD = np.zeros((len(data_valid), 1))

    # ---- Compute totals ----
    row_intermediate = Z.sum(axis=1)  # row sum of intermediate use
    row_final_demand = FD.sum(axis=1)
    row_total_output = row_intermediate + row_final_demand  # total output x_i^r

    # Build row key -> total output mapping
    row_output_map = {}
    for i in range(len(data_valid)):
        key = (row_provinces[i], row_sector_codes[i])
        row_output_map[key] = row_total_output[i]

    # Get column output (x_j^s) for each column j
    col_total_output = np.zeros(N_INTERMEDIATE)
    for j in range(N_INTERMEDIATE):
        key = (col_province[j], col_sector_code[j])
        col_total_output[j] = row_output_map.get(key, 0)

    # Avoid division by zero
    col_total_output[col_total_output == 0] = 1.0

    # ---- Direct input coefficient matrix A ----
    A = Z / col_total_output[np.newaxis, :]

    # ---- Row info ----
    row_info = pd.DataFrame({
        'province': row_provinces,
        'sector_code': row_sector_codes,
        'sector_name': row_sector_names,
        'total_output': row_total_output,
    })

    col_info = pd.DataFrame({
        'province': col_province,
        'sector_code': col_sector_code,
        'sector_name': col_sector_name,
        'total_output': col_total_output,
    })

    return A, row_info, col_info


def compute_all_linkages(A, row_info, col_info):
    """Compute backward, forward, and vertical linkages"""
    # BL: column sum of A (total direct intermediate input per unit output)
    BL = pd.Series(A.sum(axis=0), name='BL')

    # FL: row sum of A (total intermediate output supplied per unit output — Ghosh)
    FL = pd.Series(A.sum(axis=1), name='FL')

    # VL: vertical linkage = average of BL and FL for same sector
    # BL is per column (j,s), FL is per row (i,r)
    # For each province-sector, VL = 0.5*(BL_{ir} + FL_{ir})
    # BL indexed by column (same order as col_info)
    # FL indexed by row (same order as row_info)
    # Need to align by (province, sector_code)

    col_link = col_info.copy()
    col_link['BL'] = BL.values
    col_link['FL'] = np.nan  # placeholder
    col_link['VL'] = np.nan

    # Build FL lookup
    fl_lookup = {}
    for i in range(len(row_info)):
        key = (row_info.iloc[i]['province'], row_info.iloc[i]['sector_code'])
        fl_lookup[key] = FL.iloc[i]

    # Assign FL and VL
    for j in range(len(col_link)):
        key = (col_link.iloc[j]['province'], col_link.iloc[j]['sector_code'])
        fl_val = fl_lookup.get(key, np.nan)
        col_link.at[j, 'FL'] = fl_val
        if pd.notna(fl_val):
            col_link.at[j, 'VL'] = 0.5 * (col_link.iloc[j]['BL'] + fl_val)

    return col_link


# ============= MAIN =============
print("=" * 70)
print("  Computing Vertical Linkage Indices from CEADS MRIO")
print("=" * 70)

all_linkages = {}

for year, (filename, sheet) in MRIO_FILES.items():
    filepath = os.path.join(BASE, filename)
    if not os.path.exists(filepath):
        print(f"\n  [{year}] NOT FOUND: {filename}")
        continue

    size_mb = os.path.getsize(filepath) // (1024 * 1024)
    print(f"\n  [{year}] Parsing {filename} ({size_mb}MB)...")

    A, row_info, col_info = parse_mrio_table(filepath, sheet)
    print(f"    A shape: {A.shape} | Nonzero: {(A > 0).sum():,} ({100*(A>0).sum()/A.size:.1f}%)")
    print(f"    Mean A (nonzero): {A[A>0].mean():.5f}")

    linkages = compute_all_linkages(A, row_info, col_info)
    linkages['year'] = year
    all_linkages[year] = linkages

    # Print NEV sector VL summary
    print(f"\n    NEV Sector Vertical Linkages ({year}):")
    print(f"    {'Sector':<28s} {'Code':>4s}  {'VL mean':>8s}  {'BL mean':>8s}  {'FL mean':>8s}  {'#Prov':>5s}")
    print(f"    {'-'*65}")
    for code in sorted(NEV_SECTORS.keys()):
        scode = str(code).zfill(2)
        sub = linkages[linkages['sector_code'] == scode]
        if len(sub) > 0:
            print(f"    {NEV_SECTORS[code]:<28s} {scode:>4s}  {sub['VL'].mean():>8.4f}  {sub['BL'].mean():>8.4f}  {sub['FL'].mean():>8.4f}  {len(sub):>5d}")

# Save combined results
if all_linkages:
    combined = pd.concat(all_linkages.values(), ignore_index=True)
    linkage_file = os.path.join(OUTPUT_DIR, "vertical_linkage_indices.csv")
    combined.to_csv(linkage_file, index=False, encoding='utf-8-sig')
    print(f"\n{'='*70}")
    print(f"  Linkages saved: {linkage_file}")
    print(f"  {len(combined)} province-sector-year observations")
    print(f"  Years: {sorted(combined['year'].unique())}")
    print(f"{'='*70}")

    # Core NEV sector by province
    core = combined[combined['sector_code'] == '18']
    print(f"\n  Core NEV Sector (18: 交通运输设备) VL by Province:")
    pivot = core.pivot_table(values='VL', index='province', columns='year', aggfunc='first')
    print(pivot.round(3).to_string())

print(f"\nDone.")
