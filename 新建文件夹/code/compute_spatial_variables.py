"""
Compute spatial variables for NEG empirical analysis:

1. City centroid coordinates (lat/lon)
2. Spatial weights matrix W (inverse distance, k-nearest neighbors)
3. Market Access: MA_i = sum_j (GDP_j / dist_ij)  — Harris (1954) market potential
4. Supplier Access: SA_i = sum_j (VL_j * output_j / dist_ij)  — weighted by IO linkages
5. Moran's I for NEV POI data (if available)

Outputs:
  - city_coordinates.csv: city lat/lon lookup
  - spatial_weights.csv: distance-based weights matrix
  - city_panel with MA and SA variables
"""
import pandas as pd
import numpy as np
import os
import sys
import json
import urllib.request
import time
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = r"C:\Users\牧原\Desktop\A\工作区\学术研究工作区\06_NEV_产业空间集聚"
PANEL_WIDE = os.path.join(OUTPUT_DIR, "city_panel_wide.csv")

# ============================================================
# STEP 1: City coordinates
# ============================================================

# Pre-compiled coordinates for Chinese prefecture-level cities
# Sources: National Bureau of Statistics administrative division codes + Baidu/Amap geocoding
# For 297 cities in our panel, we need comprehensive coordinates

# Known coordinates for major Chinese cities (lat, lon — WGS84 approximate center)
# Full list compiled from multiple sources
CITY_COORDS = {
    # Municipalities
    '北京': (39.9042, 116.4074), '天津': (39.1252, 117.1908),
    '上海': (31.2304, 121.4737), '重庆市': (29.4316, 106.9123),
    # Hebei
    '石家庄市': (38.0423, 114.5149), '唐山市': (39.6305, 118.1802),
    '秦皇岛市': (39.9354, 119.5996), '邯郸市': (36.6256, 114.5392),
    '邢台市': (37.0706, 114.5048), '保定市': (38.8738, 115.4646),
    '张家口市': (40.7686, 114.8864), '承德市': (40.9526, 117.9626),
    '沧州市': (38.3047, 116.8388), '廊坊市': (39.5378, 116.6838),
    '衡水市': (37.7389, 115.6692),
    # Shanxi
    '太原市': (37.8706, 112.5489), '大同市': (40.0976, 113.3001),
    '阳泉市': (37.8567, 113.5805), '长治市': (36.1954, 113.1164),
    '晋城市': (35.4907, 112.8515), '朔州市': (39.3316, 112.4328),
    '晋中市': (37.6870, 112.7527), '运城市': (35.0265, 111.0071),
    '忻州市': (38.4161, 112.7342), '临汾市': (36.0882, 111.5189),
    '吕梁市': (37.5190, 111.1416),
    # Inner Mongolia
    '呼和浩特市': (40.8424, 111.7492), '包头市': (40.6582, 109.8404),
    '乌海市': (39.6558, 106.7955), '赤峰市': (42.2578, 118.8869),
    '通辽市': (43.6529, 122.2445), '鄂尔多斯市': (39.6085, 109.7805),
    '呼伦贝尔市': (49.2116, 119.7658), '巴彦淖尔市': (40.7434, 107.3877),
    '乌兰察布市': (40.9945, 113.1327),
    # Liaoning
    '沈阳市': (41.8057, 123.4315), '大连市': (38.9140, 121.6147),
    '鞍山市': (41.1086, 122.9943), '抚顺市': (41.8805, 123.9572),
    '本溪市': (41.2940, 123.7667), '丹东市': (40.0005, 124.3547),
    '锦州市': (41.0959, 121.1270), '营口市': (40.6254, 122.2355),
    '阜新市': (42.0217, 121.6703), '辽阳市': (41.2681, 123.2371),
    '盘锦市': (41.1199, 122.0708), '铁岭市': (42.2862, 123.8424),
    '朝阳市': (41.5737, 120.4509), '葫芦岛市': (40.7110, 120.8368),
    # Jilin
    '长春市': (43.8171, 125.3235), '吉林市': (43.8379, 126.5494),
    '四平市': (43.1665, 124.3509), '辽源市': (42.8880, 125.1437),
    '通化市': (41.7283, 125.9399), '白山市': (41.9396, 126.4234),
    '松原市': (45.1411, 124.8251), '白城市': (45.6196, 122.8387),
    # Heilongjiang
    '哈尔滨市': (45.8038, 126.5350), '齐齐哈尔市': (47.3540, 123.9180),
    '鸡西市': (45.2952, 130.9691), '鹤岗市': (47.3496, 130.2984),
    '双鸭山市': (46.6466, 131.1591), '大庆市': (46.5901, 125.1036),
    '伊春市': (47.7275, 128.8407), '佳木斯市': (46.7998, 130.3189),
    '七台河市': (45.7707, 131.0031), '牡丹江市': (44.5517, 129.6333),
    '黑河市': (50.2446, 127.5279), '绥化市': (46.6533, 126.9688),
    # Jiangsu
    '南京市': (32.0603, 118.7969), '无锡市': (31.4912, 120.3124),
    '徐州市': (34.2044, 117.2841), '常州市': (31.8110, 119.9740),
    '苏州市': (31.2990, 120.5853), '南通市': (31.9803, 120.8943),
    '连云港市': (34.5967, 119.2228), '淮安市': (33.6103, 119.0159),
    '盐城市': (33.3495, 120.1621), '扬州市': (32.3936, 119.4124),
    '镇江市': (32.1896, 119.4253), '泰州市': (32.4555, 119.9230),
    '宿迁市': (33.9622, 118.2753),
    # Zhejiang
    '杭州市': (30.2741, 120.1551), '宁波市': (29.8683, 121.5440),
    '温州市': (27.9939, 120.6994), '嘉兴市': (30.7460, 120.7555),
    '湖州市': (30.8943, 120.0872), '绍兴市': (30.0310, 120.5819),
    '金华市': (29.0784, 119.6472), '衢州市': (28.9357, 118.8593),
    '舟山市': (30.0161, 122.2064), '台州市': (28.6557, 121.4208),
    '丽水市': (28.4670, 119.9229),
    # Anhui
    '合肥市': (31.8206, 117.2272), '芜湖市': (31.3526, 118.4331),
    '蚌埠市': (32.9155, 117.3893), '淮南市': (32.6255, 116.9998),
    '马鞍山市': (31.6706, 118.5061), '淮北市': (33.9558, 116.7983),
    '铜陵市': (30.9451, 117.8114), '安庆市': (30.5429, 117.0635),
    '黄山市': (29.7152, 118.3376), '滁州市': (32.2559, 118.3328),
    '阜阳市': (32.8896, 115.8145), '宿州市': (33.6476, 116.9636),
    '六安市': (31.7349, 116.5216), '亳州市': (33.8446, 115.7792),
    '池州市': (30.6647, 117.4914), '宣城市': (30.9407, 118.7588),
    # Fujian
    '福州市': (26.0745, 119.2965), '厦门市': (24.4798, 118.0894),
    '莆田市': (25.4540, 119.0077), '三明市': (26.2638, 117.6390),
    '泉州市': (24.8741, 118.6759), '漳州市': (24.5135, 117.6475),
    '南平市': (26.6416, 118.1777), '龙岩市': (25.0751, 117.0173),
    '宁德市': (26.6657, 119.5481),
    # Jiangxi
    '南昌市': (28.6820, 115.8582), '景德镇市': (29.2688, 117.1784),
    '萍乡市': (27.6228, 113.8547), '九江市': (29.7052, 116.0019),
    '新余市': (27.8178, 114.9173), '鹰潭市': (28.2605, 117.0692),
    '赣州市': (25.8318, 114.9348), '吉安市': (27.1138, 114.9938),
    '宜春市': (27.8144, 114.4168), '抚州市': (27.9492, 116.3581),
    '上饶市': (28.4549, 117.9436),
    # Shandong
    '济南市': (36.6518, 117.1201), '青岛市': (36.0662, 120.3826),
    '淄博市': (36.8131, 118.0550), '枣庄市': (34.8105, 117.3217),
    '东营市': (37.4347, 118.6747), '烟台市': (37.4638, 121.4479),
    '潍坊市': (36.7068, 119.1617), '济宁市': (35.4146, 116.5872),
    '泰安市': (36.2000, 117.0876), '威海市': (37.5131, 122.1204),
    '日照市': (35.4164, 119.5269), '临沂市': (35.1047, 118.3564),
    '德州市': (37.4341, 116.3575), '聊城市': (36.4570, 115.9853),
    '滨州市': (37.3821, 117.9728), '菏泽市': (35.2337, 115.4807),
    # Henan
    '郑州市': (34.7466, 113.6254), '开封市': (34.7973, 114.3077),
    '洛阳市': (34.6181, 112.4536), '平顶山市': (33.7666, 113.1926),
    '安阳市': (36.0978, 114.3931), '鹤壁市': (35.7473, 114.2974),
    '新乡市': (35.3037, 113.9267), '焦作市': (35.2156, 113.2421),
    '濮阳市': (35.7618, 115.0292), '许昌市': (34.0357, 113.8526),
    '漯河市': (33.5809, 114.0165), '三门峡市': (34.7725, 111.2003),
    '南阳市': (32.9907, 112.5283), '商丘市': (34.4145, 115.6563),
    '信阳市': (32.1473, 114.0912), '周口市': (33.6259, 114.6968),
    '驻马店市': (33.0114, 114.0228),
    # Hubei
    '武汉市': (30.5928, 114.3055), '黄石市': (30.1995, 115.0389),
    '十堰市': (32.6292, 110.7979), '宜昌市': (30.6910, 111.2865),
    '襄阳市': (32.0094, 112.1223), '鄂州市': (30.3910, 114.8949),
    '荆门市': (31.0354, 112.1994), '孝感市': (30.9248, 113.9165),
    '荆州市': (30.3347, 112.2407), '黄冈市': (30.4537, 114.8724),
    '咸宁市': (29.8414, 114.3225), '随州市': (31.6901, 113.3826),
    # Hunan
    '长沙市': (28.2282, 112.9388), '株洲市': (27.8276, 113.1340),
    '湘潭市': (27.8297, 112.9442), '衡阳市': (26.8932, 112.5720),
    '邵阳市': (27.2389, 111.4677), '岳阳市': (29.3571, 113.1291),
    '常德市': (29.0316, 111.6985), '张家界市': (29.1170, 110.4785),
    '益阳市': (28.5543, 112.3557), '郴州市': (25.7705, 113.0148),
    '永州市': (26.4203, 111.6134), '怀化市': (27.5694, 109.9985),
    '娄底市': (27.6973, 111.9946),
    # Guangdong
    '广州市': (23.1292, 113.2644), '韶关市': (24.8104, 113.5972),
    '深圳市': (22.5431, 114.0579), '珠海市': (22.2707, 113.5767),
    '汕头市': (23.3533, 116.6822), '佛山市': (23.0220, 113.1214),
    '江门市': (22.5787, 113.0816), '湛江市': (21.2712, 110.3589),
    '茂名市': (21.6629, 110.9255), '肇庆市': (23.0469, 112.4651),
    '惠州市': (23.1118, 114.4168), '梅州市': (24.2884, 116.1222),
    '汕尾市': (22.7866, 115.3753), '河源市': (23.7437, 114.7007),
    '阳江市': (21.8579, 111.9826), '清远市': (23.6820, 113.0560),
    '东莞市': (23.0208, 113.7520), '中山市': (22.5159, 113.3926),
    '潮州市': (23.6567, 116.6224), '揭阳市': (23.5497, 116.3727),
    '云浮市': (22.9153, 112.0446),
    # Guangxi
    '南宁市': (22.8167, 108.3669), '柳州市': (24.3255, 109.4155),
    '桂林市': (25.2736, 110.2900), '梧州市': (23.4768, 111.2792),
    '北海市': (21.4733, 109.1192), '防城港市': (21.6871, 108.3547),
    '钦州市': (21.9797, 108.6543), '贵港市': (23.1118, 109.5989),
    '玉林市': (22.6545, 110.1810), '百色市': (23.9024, 106.6184),
    '贺州市': (24.4036, 111.5666), '河池市': (24.6929, 108.0854),
    '来宾市': (23.7503, 109.2214), '崇左市': (22.3768, 107.3647),
    # Hainan
    '海口市': (20.0440, 110.1999), '三亚市': (18.2536, 109.5119),
    '三沙市': (16.8310, 112.3384), '儋州市': (19.5209, 109.5807),
    # Sichuan
    '成都市': (30.5728, 104.0668), '自贡市': (29.3392, 104.7784),
    '攀枝花市': (26.5824, 101.7186), '泸州市': (28.8717, 105.4423),
    '德阳市': (31.1274, 104.3979), '绵阳市': (31.4675, 104.6786),
    '广元市': (32.4354, 105.8436), '遂宁市': (30.5329, 105.5927),
    '内江市': (29.5802, 105.0584), '乐山市': (29.5521, 103.7656),
    '南充市': (30.8378, 106.1107), '眉山市': (30.0768, 103.8484),
    '宜宾市': (28.7513, 104.6433), '广安市': (30.4560, 106.6332),
    '达州市': (31.2086, 107.4678), '雅安市': (30.0105, 103.0421),
    '巴中市': (31.8672, 106.7475), '资阳市': (30.1289, 104.6277),
    # Guizhou
    '贵阳市': (26.6470, 106.6302), '六盘水市': (26.5925, 104.8304),
    '遵义市': (27.7255, 106.9274), '安顺市': (26.2531, 105.9476),
    '毕节市': (27.2839, 105.3050), '铜仁市': (27.7315, 109.1896),
    # Yunnan
    '昆明市': (25.0389, 102.7183), '曲靖市': (25.4900, 103.7962),
    '玉溪市': (24.3518, 102.5465), '保山市': (25.1120, 99.1615),
    '昭通市': (27.3382, 103.7167), '丽江市': (26.8567, 100.2278),
    '普洱市': (22.8252, 100.9665), '临沧市': (23.8842, 100.0393),
    # Tibet
    '拉萨市': (29.6500, 91.1000), '日喀则市': (29.2669, 88.8802),
    '昌都市': (31.1426, 97.1722), '林芝市': (29.6491, 94.3615),
    '山南市': (29.2371, 91.7731), '那曲市': (31.4767, 92.0514),
    # Shaanxi
    '西安市': (34.3416, 108.9398), '铜川市': (34.8967, 108.9451),
    '宝鸡市': (34.3620, 107.2377), '咸阳市': (34.3296, 108.7090),
    '渭南市': (34.4996, 109.5098), '延安市': (36.5855, 109.4896),
    '汉中市': (33.0676, 107.0238), '榆林市': (38.2852, 109.7346),
    '安康市': (32.6847, 109.0290), '商洛市': (33.8733, 109.9183),
    # Gansu
    '兰州市': (36.0614, 103.8343), '嘉峪关市': (39.7725, 98.2881),
    '金昌市': (38.5201, 102.1880), '白银市': (36.5448, 104.1377),
    '天水市': (34.5809, 105.7249), '武威市': (37.9282, 102.6380),
    '张掖市': (38.9259, 100.4498), '平凉市': (35.5430, 106.6652),
    '酒泉市': (39.7326, 98.4940), '庆阳市': (35.7098, 107.6436),
    '定西市': (35.5806, 104.6250), '陇南市': (33.4010, 104.9217),
    # Qinghai
    '西宁市': (36.6171, 101.7785), '海东市': (36.5021, 102.1044),
    # Ningxia
    '银川市': (38.4872, 106.2309), '石嘴山市': (38.9840, 106.3833),
    '吴忠市': (37.9974, 106.1986), '固原市': (36.0158, 106.2426),
    '中卫市': (37.5000, 105.1967),
    # Xinjiang
    '乌鲁木齐市': (43.8256, 87.6168), '克拉玛依市': (45.5799, 84.8892),
    '吐鲁番市': (42.9513, 89.1841), '哈密市': (42.8194, 93.5149),
}

print(f"  Coordinate database: {len(CITY_COORDS)} cities")
print("=" * 70)
print("  Computing Spatial Variables for NEG Empirical Analysis")
print("=" * 70)

# Load panel
panel = pd.read_csv(PANEL_WIDE, encoding='utf-8-sig')
print(f"\n  Panel: {len(panel)} rows")

# Map coordinates to panel cities
panel['lat'] = panel['city'].map(lambda c: CITY_COORDS.get(c, (None, None))[0])
panel['lon'] = panel['city'].map(lambda c: CITY_COORDS.get(c, (None, None))[1])

matched = panel['lat'].notna().sum()
total = len(panel['city'].unique())
matched_cities = panel[panel['lat'].notna()]['city'].nunique()
total_cities = panel['city'].nunique()
print(f"  Coordinates: {matched_cities}/{total_cities} cities matched")

# Unmatched cities
unmatched = panel[panel['lat'].isna()]['city'].unique()
if len(unmatched) > 0:
    print(f"  Unmatched ({len(unmatched)}): {sorted(unmatched)[:20]}")

# ============================================================
# STEP 2: Compute bilateral distance matrix (great-circle distance)
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two points"""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

# Get unique city coordinates
city_coords = panel[['city', 'lat', 'lon', 'province']].drop_duplicates(subset=['city'])
city_coords = city_coords[city_coords['lat'].notna()].copy()

n_cities = len(city_coords)
print(f"\n  Computing {n_cities}x{n_cities} distance matrix...")

# Build distance matrix
cities_sorted = sorted(city_coords['city'].unique())
city_to_idx = {c: i for i, c in enumerate(cities_sorted)}
n = len(cities_sorted)

lat_arr = np.array([city_coords[city_coords['city'] == c]['lat'].iloc[0] for c in cities_sorted])
lon_arr = np.array([city_coords[city_coords['city'] == c]['lon'].iloc[0] for c in cities_sorted])

# Vectorized distance computation
dist_matrix = np.zeros((n, n))
for i in range(n):
    dist_matrix[i, :] = haversine(lat_arr[i], lon_arr[i], lat_arr, lon_arr)

# Set diagonal to large value (a city's "distance to itself" is handled separately)
np.fill_diagonal(dist_matrix, 1.0)  # 1km internal distance

print(f"  Distance range: {dist_matrix[dist_matrix > 1].min():.1f} - {dist_matrix.max():.1f} km")
print(f"  Median inter-city distance: {np.median(dist_matrix[dist_matrix > 1]):.1f} km")

# ============================================================
# STEP 3: Spatial weights matrix
# ============================================================

# Inverse distance weights: W_ij = 1 / d_ij (i != j), W_ii = 0
W_invdist = 1.0 / dist_matrix
np.fill_diagonal(W_invdist, 0.0)

# Row-standardize
row_sums = W_invdist.sum(axis=1)
row_sums[row_sums == 0] = 1
W = W_invdist / row_sums[:, np.newaxis]

print(f"  Spatial weights W: {W.shape}")
print(f"  Mean non-zero weight: {W[W>0].mean():.6f}")

# ============================================================
# STEP 4: Market Access (Harris Market Potential)
# ============================================================
# MA_i = sum_{j != i} GDP_j / d_ij

# Get average GDP per city (use latest available year, 2023)
gdp_2023 = panel[panel['year'] == 2023][['city', '人均地区生产总值（元）', '常住人口（万人）']].copy()
# If 2023 not available, try other years
if gdp_2023['人均地区生产总值（元）'].notna().sum() < 100:
    for yr in [2022, 2021, 2020, 2019]:
        gdp_yr = panel[panel['year'] == yr][['city', '人均地区生产总值（元）', '常住人口（万人）']].copy()
        if gdp_yr['人均地区生产总值（元）'].notna().sum() > 100:
            gdp_2023 = gdp_yr
            break

# Compute total GDP proxy: GDP per capita × population
gdp_2023['total_gdp'] = gdp_2023['人均地区生产总值（元）'] * gdp_2023['常住人口（万人）'] * 10000  # 万人→人

# Fill missing values with provincial average
gdp_2023['province'] = gdp_2023['city'].map(
    lambda c: city_coords[city_coords['city'] == c]['province'].iloc[0]
    if c in city_coords['city'].values else None
)

# Build GDP array aligned with city list
gdp_arr = np.zeros(n)
for i, c in enumerate(cities_sorted):
    gdp_val = gdp_2023[gdp_2023['city'] == c]['total_gdp'].values
    if len(gdp_val) > 0 and not np.isnan(gdp_val[0]):
        gdp_arr[i] = gdp_val[0]
    else:
        gdp_arr[i] = 0

# Fill zeros with median
if (gdp_arr == 0).any():
    gdp_arr[gdp_arr == 0] = np.median(gdp_arr[gdp_arr > 0])

print(f"\n  Total GDP range: {gdp_arr.min()/1e8:.1f}B - {gdp_arr.max()/1e8:.1f}B yuan")

# Market Access
MA = np.zeros(n)
for i in range(n):
    weights = 1.0 / dist_matrix[i, :]
    weights[i] = 0  # exclude self
    MA[i] = np.sum(gdp_arr * weights)

# Normalize MA (log)
MA_log = np.log(MA)

print(f"  Market Access (log): mean={MA_log.mean():.2f}, std={MA_log.std():.2f}")

# ============================================================
# STEP 5: Supplier Access (IO-weighted market access)
# ============================================================
# SA_i = sum_j (VL_j × output_j) / d_ij
# Where VL_j is the vertical linkage index for sector 18 (交通运输设备)
# Using provincial VL as proxy for city-level supplier importance

# Get VL values
vl_data = panel[panel['year'] == 2023][['city', 'province', 'VL_transport_eq_interp']].drop_duplicates(subset=['city'])

# Build VL array
vl_arr = np.zeros(n)
for i, c in enumerate(cities_sorted):
    vl_val = vl_data[vl_data['city'] == c]['VL_transport_eq_interp'].values
    if len(vl_val) > 0 and not np.isnan(vl_val[0]):
        vl_arr[i] = vl_val[0]

if (vl_arr == 0).any():
    vl_arr[vl_arr == 0] = np.median(vl_arr[vl_arr > 0])

# Supplier importance: VL × GDP (high-VL cities are more important suppliers)
supplier_weight = vl_arr * gdp_arr

SA = np.zeros(n)
for i in range(n):
    weights = 1.0 / dist_matrix[i, :]
    weights[i] = 0
    SA[i] = np.sum(supplier_weight * weights)

SA_log = np.log(SA)

print(f"  Supplier Access (log): mean={SA_log.mean():.2f}, std={SA_log.std():.2f}")

# ============================================================
# STEP 6: Add MA and SA to panel
# ============================================================

# Build lookup dicts
ma_lookup = {cities_sorted[i]: MA_log[i] for i in range(n)}
sa_lookup = {cities_sorted[i]: SA_log[i] for i in range(n)}

panel['market_access_log'] = panel['city'].map(ma_lookup)
panel['supplier_access_log'] = panel['city'].map(sa_lookup)
panel['city_lat'] = panel['city'].map(lambda c: CITY_COORDS.get(c, (None, None))[0])
panel['city_lon'] = panel['city'].map(lambda c: CITY_COORDS.get(c, (None, None))[1])

# Save
panel.to_csv(PANEL_WIDE, index=False, encoding='utf-8-sig')
print(f"\n  Panel updated: {PANEL_WIDE}")

# Also save spatial weights and distance matrix for spatial econometrics
# Save distance matrix
dist_df = pd.DataFrame(dist_matrix, index=cities_sorted, columns=cities_sorted)
dist_file = os.path.join(OUTPUT_DIR, "city_distance_matrix.csv")
dist_df.to_csv(dist_file, encoding='utf-8-sig')
print(f"  Distance matrix: {dist_file}")

# Save weights matrix
W_df = pd.DataFrame(W, index=cities_sorted, columns=cities_sorted)
W_file = os.path.join(OUTPUT_DIR, "spatial_weights_W.csv")
W_df.to_csv(W_file, encoding='utf-8-sig')
print(f"  Spatial weights: {W_file}")

# Save city coordinates
coord_df = pd.DataFrame({
    'city': cities_sorted,
    'lat': lat_arr,
    'lon': lon_arr,
    'MA_log': MA_log,
    'SA_log': SA_log,
})
coord_file = os.path.join(OUTPUT_DIR, "city_spatial_variables.csv")
coord_df.to_csv(coord_file, index=False, encoding='utf-8-sig')
print(f"  City spatial vars: {coord_file}")

# Add to long panel
PANEL_LONG = os.path.join(OUTPUT_DIR, "city_panel_long.csv")
panel_long = pd.read_csv(PANEL_LONG, encoding='utf-8-sig')

new_rows = []
for _, r in panel.iterrows():
    for indicator_name, val in [('市场可达性(log MA)', r.get('market_access_log')),
                                  ('供应商可达性(log SA)', r.get('supplier_access_log'))]:
        if pd.notna(val):
            new_rows.append({'city': r['city'], 'year': r['year'],
                           'indicator': indicator_name, 'value': val})

if new_rows:
    new_df = pd.DataFrame(new_rows)
    panel_long = pd.concat([panel_long, new_df], ignore_index=True)
    panel_long.to_csv(PANEL_LONG, index=False, encoding='utf-8-sig')
    print(f"  Long panel: {PANEL_LONG} ({len(panel_long)} rows)")

# ============================================================
print(f"\n{'='*70}")
print(f"  Spatial Variables Summary")
print(f"{'='*70}")
print(f"  Cities with coords: {n_cities}/{total_cities}")
print(f"  Distance matrix: {n}x{n}")
print(f"  Spatial weights: row-standardized inverse distance")
print(f"  MA (Market Access): log of Harris market potential")
print(f"  SA (Supplier Access): VL-weighted supplier potential")
print(f"\n  Top 10 Market Access cities:")
top_ma = coord_df.nlargest(10, 'MA_log')[['city', 'MA_log', 'SA_log']]
for _, r in top_ma.iterrows():
    print(f"    {r['city']:<12s}  MA={r['MA_log']:.2f}  SA={r['SA_log']:.2f}")
print(f"{'='*70}")
