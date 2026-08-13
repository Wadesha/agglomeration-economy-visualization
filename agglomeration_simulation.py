"""
集聚效益模拟展示
模拟数据 + 可视化展示集聚效应的多个维度
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'AR PL UMing CN', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# 探测可用的中文字体
from matplotlib.font_manager import FontProperties
import os

def find_chinese_font():
    """尝试找到系统中可用的中文字体"""
    # 常见中文字体路径
    font_paths = [
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            return FontProperties(fname=fp)
    return None

chinese_font = find_chinese_font()
if chinese_font:
    chinese_font_prop = chinese_font
else:
    # 尝试从系统查找
    import subprocess
    try:
        result = subprocess.run(['fc-list', ':lang=zh', '-f', '%{file}\n'], capture_output=True, text=True, timeout=5)
        fonts = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        if fonts:
            chinese_font_prop = FontProperties(fname=fonts[0])
        else:
            chinese_font_prop = None
    except:
        chinese_font_prop = None

# 设置随机种子
np.random.seed(42)

# ============================================================
# 1. 模拟数据：生成 200 个城市的各项指标
# ============================================================
print("正在生成模拟数据...")
n_cities = 200

# 城市人口规模（对数正态分布，模拟真实城市规模分布）
population = np.random.lognormal(mean=5.5, sigma=1.2, size=n_cities)
population = np.clip(population, 50, 3000)  # 单位：万人

# 建成区面积（与人口呈亚线性关系，密度随规模增加）
area = 10 * (population ** 0.6) + np.random.normal(0, 5, n_cities)
area = np.clip(area, 5, 800)
density = population / area * 10000  # 人/平方公里

# 城市等级标签
city_labels = []
for p in population:
    if p < 100:
        city_labels.append('小城市 (<100万)')
    elif p < 500:
        city_labels.append('中等城市 (100-500万)')
    elif p < 1000:
        city_labels.append('大城市 (500-1000万)')
    else:
        city_labels.append('超大城市 (>1000万)')

# 经济产出：集聚效应下，产出与人口呈超线性关系 (β ≈ 1.12)
# Y = Y0 * N^β * exp(ε)
beta_true = 1.12
gdp = 0.5 * (population ** beta_true) * np.random.lognormal(mean=0, sigma=0.15, size=n_cities)
gdp_per_capita = gdp / population * 10000  # 万元/人

# 人均收入（与人均GDP相关）
income = gdp_per_capita * (0.4 + 0.1 * np.random.random(n_cities))

# 创新能力（专利申请量，与人口规模超线性相关）
patents = 0.02 * (population ** 1.25) * np.random.lognormal(mean=0, sigma=0.3, size=n_cities)
patents = np.round(patents).astype(int)

# 平均工资（万元/年）
wage = 3.5 + 2.8 * np.log(population / 100) + np.random.normal(0, 0.8, n_cities)
wage = np.clip(wage, 3, 25)

# 产业多样性（HHI 逆指数，越大越多样）
industry_diversity = 0.3 + 0.5 * np.log(population / 50) / np.log(10) + np.random.normal(0, 0.08, n_cities)
industry_diversity = np.clip(industry_diversity, 0.1, 1.0)

# 区位商 (Location Quotient) - 模拟高科技产业
# 大城市高科技产业集聚度更高
lq_high_tech = 0.3 + 0.6 * (np.log(population) - np.log(population).min()) / (np.log(population).max() - np.log(population).min()) + np.random.normal(0, 0.15, n_cities)
lq_high_tech = np.clip(lq_high_tech, 0.1, 3.0)

# 就业密度
job_density = density * (0.3 + 0.2 * np.random.random(n_cities))

# 企业数量
firms = 50 * (population ** 0.9) * np.random.lognormal(mean=0, sigma=0.2, size=n_cities)
firms = np.round(firms).astype(int)

# 基础设施评分（0-100）
infrastructure = 30 + 25 * np.log(population / 50) + np.random.normal(0, 5, n_cities)
infrastructure = np.clip(infrastructure, 10, 100)

# ============================================================
# 生成图表
# ============================================================
output_dir = '/workspace'
print("正在生成图表...")

# ---------- 图1: 城市规模-产出标度律 (核心图表) ----------
fig, ax = plt.subplots(figsize=(10, 7))

# 按城市规模着色
sizes = np.where(population < 100, 20, 
                np.where(population < 500, 40,
                        np.where(population < 1000, 80, 120)))
colors = np.where(population < 100, '#4A90D9',
                 np.where(population < 500, '#2ECC71',
                         np.where(population < 1000, '#F39C12', '#E74C3C')))

scatter = ax.scatter(population, gdp, c=colors, s=sizes, alpha=0.6, edgecolors='white', linewidth=0.5)

# 拟合标度律
log_pop = np.log(population)
log_gdp = np.log(gdp)
slope, intercept, r_value, p_value, std_err = stats.linregress(log_pop, log_gdp)
r_squared = r_value ** 2

# 拟合线
x_fit = np.linspace(population.min(), population.max(), 100)
y_fit = np.exp(intercept) * (x_fit ** slope)
ax.plot(x_fit, y_fit, '--', color='#8E44AD', linewidth=2.5, 
        label=f'标度律拟合: β = {slope:.3f}\nR² = {r_squared:.3f}')

# 加入参考线 β=1（线性增长）
y_linear = np.exp(intercept) * (x_fit ** 1.0)
ax.plot(x_fit, y_linear, ':', color='gray', linewidth=1.5, alpha=0.6, label='β = 1 (线性增长)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('城市人口 (万人)', fontsize=13, fontproperties=chinese_font_prop)
ax.set_ylabel('GDP (亿元)', fontsize=13, fontproperties=chinese_font_prop)
ax.set_title('城市标度律: 人口规模 vs 经济产出\n(超线性关系 β>1 表明集聚效益)', fontsize=15, fontproperties=chinese_font_prop)

# 自定义图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#4A90D9', markersize=8, label='小城市 (<100万)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ECC71', markersize=10, label='中等城市 (100-500万)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#F39C12', markersize=14, label='大城市 (500-1000万)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C', markersize=18, label='超大城市 (>1000万)'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10, prop=chinese_font_prop)

# 添加注释说明
ax.annotate(f'解读: β = {slope:.3f} > 1\n说明城市人口每增长1%\n经济产出增长约{slope:.2f}%\n存在显著的集聚效益',
            xy=(0.02, 0.02), xycoords='axes fraction', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
            verticalalignment='bottom', fontproperties=chinese_font_prop)

ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/01_scaling_law.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 图1: 城市标度律")

# ---------- 图2: 人口密度 vs 人均GDP ----------
fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(density, gdp_per_capita, c=density, cmap='viridis', s=50, alpha=0.6, 
           edgecolors='white', linewidth=0.5)
cbar = plt.colorbar(ax.collections[0], ax=ax, label='人口密度 (人/km²)')

# 拟合
log_density = np.log(density[density > 0])
log_gdp_pc = np.log(gdp_per_capita[density > 0])
slope2, intercept2, r2, p2, se2 = stats.linregress(log_density, log_gdp_pc)

x_dense = np.linspace(density.min(), density.max(), 100)
y_dense = np.exp(intercept2) * (x_dense ** slope2)
ax.plot(x_dense, y_dense, '--', color='red', linewidth=2, 
        label=f'弹性系数 = {slope2:.3f}, R² = {r2**2:.3f}')

ax.set_xlabel('人口密度 (人/km²)', fontsize=13, fontproperties=chinese_font_prop)
ax.set_ylabel('人均 GDP (万元/人)', fontsize=13, fontproperties=chinese_font_prop)
ax.set_title('人口密度 vs 人均 GDP\n(密度越高，人均产出越高)', fontsize=15, fontproperties=chinese_font_prop)
ax.legend(fontsize=11, prop=chinese_font_prop)
ax.grid(True, alpha=0.3)

ax.annotate(f'解读: 人口密度每提升1%\n人均GDP约提升{slope2:.3f}%\n高密度区域产出效率显著更高',
            xy=(0.02, 0.95), xycoords='axes fraction', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
            verticalalignment='top', fontproperties=chinese_font_prop)

plt.tight_layout()
plt.savefig(f'{output_dir}/02_density_vs_gdp.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 图2: 密度 vs 人均GDP")

# ---------- 图3: 集聚 vs 非集聚分组对比 ----------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 按城市规模分组
groups = ['小城市\n(<100万)', '中等城市\n(100-500万)', '大城市\n(500-1000万)', '超大城市\n(>1000万)']
group_indices = [
    population < 100,
    (population >= 100) & (population < 500),
    (population >= 500) & (population < 1000),
    population >= 1000
]

metrics = [
    (gdp_per_capita, '人均 GDP (万元/人)', '#4A90D9', '人均GDP随城市规模递增'),
    (wage, '平均工资 (万元/年)', '#2ECC71', '大城市工资溢价显著'),
    (patents, '专利申请量 (件)', '#E74C3C', '创新活动高度集聚于大城市'),
    (industry_diversity, '产业多样性指数', '#9B59B6', '大城市产业更加多样化'),
    (infrastructure, '基础设施评分', '#F39C12', '大城市基础设施更完善'),
    (lq_high_tech, '高科技产业区位商 (LQ)', '#1ABC9C', '大城市高科技产业集聚度更高'),
]

for idx, (metric, label, color, desc) in enumerate(metrics):
    ax = axes[idx // 3][idx % 3]
    data_by_group = [metric[mask] for mask in group_indices]
    
    bp = ax.boxplot(data_by_group, patch_artist=True, widths=0.6)
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    # 添加均值标注
    means = [np.mean(d) for d in data_by_group]
    for i, m in enumerate(means):
        ax.plot(i + 1, m, 'D', color='darkred', markersize=8, zorder=5)
    
    ax.set_xticklabels(groups, fontsize=9, fontproperties=chinese_font_prop)
    ax.set_ylabel(label, fontsize=10, fontproperties=chinese_font_prop)
    ax.set_title(desc, fontsize=11, fontproperties=chinese_font_prop)
    ax.grid(True, alpha=0.2)

fig.suptitle('集聚效益分组对比: 不同规模城市的各项指标', fontsize=16, fontproperties=chinese_font_prop)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f'{output_dir}/03_group_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 图3: 分组对比")

# ---------- 图4: 集聚效益的倒U型曲线 ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图: 集聚度与效益的倒U型关系
ax = axes[0]
# 模拟集聚度从低到高
agg_level = np.linspace(0.1, 1.5, 100)
# 效益先增后减（倒U型）
benefit = -1.8 * (agg_level - 0.8) ** 2 + 2.0 + np.random.normal(0, 0.05, 100)
# 实际数据点
np.random.seed(123)
actual_agg = np.random.uniform(0.15, 1.4, 50)
actual_benefit = -1.8 * (actual_agg - 0.8) ** 2 + 2.0 + np.random.normal(0, 0.12, 50)

ax.scatter(actual_agg, actual_benefit, alpha=0.5, c='#3498DB', s=40, label='模拟城市数据')
ax.plot(agg_level, -1.8 * (agg_level - 0.8) ** 2 + 2.0, '-', color='#E74C3C', linewidth=3, label='拟合曲线 (倒U型)')

# 标记最佳集聚点
opt_idx = np.argmax(-1.8 * (agg_level - 0.8) ** 2 + 2.0)
ax.axvline(x=agg_level[opt_idx], color='green', linestyle='--', alpha=0.7, linewidth=1.5)
ax.annotate(f'最佳集聚度\n{agg_level[opt_idx]:.2f}', xy=(agg_level[opt_idx], -1.8 * (agg_level[opt_idx] - 0.8) ** 2 + 2.0),
            xytext=(agg_level[opt_idx] + 0.15, -1.8 * (agg_level[opt_idx] - 0.8) ** 2 + 2.0 - 0.3),
            arrowprops=dict(arrowstyle='->', color='green'), fontsize=11, fontproperties=chinese_font_prop)

# 标注过度集聚区域
ax.axvspan(1.0, 1.5, alpha=0.1, color='red')
ax.annotate('过度集聚区\n(拥堵效应 > 集聚效益)', xy=(1.25, 0.5), fontsize=10,
            color='red', fontproperties=chinese_font_prop)

ax.set_xlabel('集聚度 (人口密度/企业密度)', fontsize=12, fontproperties=chinese_font_prop)
ax.set_ylabel('综合效益指数', fontsize=12, fontproperties=chinese_font_prop)
ax.set_title('集聚效益的倒U型曲线\n(适度集聚效益最大，过度则下降)', fontsize=13, fontproperties=chinese_font_prop)
ax.legend(fontsize=10, prop=chinese_font_prop)
ax.grid(True, alpha=0.3)

# 右图: 人口规模 vs 人均工资（工资溢价）
ax = axes[1]
ax.scatter(population, wage, c=population, cmap='plasma', s=50, alpha=0.6, edgecolors='white', linewidth=0.5)
cbar = plt.colorbar(ax.collections[0], ax=ax, label='人口 (万人)')

# 拟合
log_pop2 = np.log(population)
log_wage = np.log(wage)
slope3, intercept3, r3, p3, se3 = stats.linregress(log_pop2, log_wage)
x_wage = np.linspace(population.min(), population.max(), 100)
y_wage = np.exp(intercept3) * (x_wage ** slope3)
ax.plot(x_wage, y_wage, '--', color='red', linewidth=2.5, label=f'工资-规模弹性: {slope3:.3f}')

ax.set_xlabel('城市人口 (万人)', fontsize=12, fontproperties=chinese_font_prop)
ax.set_ylabel('平均工资 (万元/年)', fontsize=12, fontproperties=chinese_font_prop)
ax.set_title('城市规模与工资溢价\n(大城市支付更高工资)', fontsize=13, fontproperties=chinese_font_prop)
ax.legend(fontsize=11, prop=chinese_font_prop)
ax.grid(True, alpha=0.3)

ax.annotate(f'解读: 人口规模每增长1%\n平均工资增长约{slope3:.3f}%\n集聚带来劳动力市场效率提升',
            xy=(0.02, 0.95), xycoords='axes fraction', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
            verticalalignment='top', fontproperties=chinese_font_prop)

plt.tight_layout()
plt.savefig(f'{output_dir}/04_inverted_U_and_wage.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 图4: 倒U型曲线 + 工资溢价")

# ---------- 图5: 产业集聚热力图（模拟城市群）----------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图: 模拟城市空间分布 + 产业集聚
ax = axes[0]
np.random.seed(456)
# 模拟3个城市群
n_points = 300
cluster1 = np.random.multivariate_normal([2, 2], [[0.3, 0.1], [0.1, 0.3]], 120)
cluster2 = np.random.multivariate_normal([6, 5], [[0.4, 0.1], [0.1, 0.4]], 100)
cluster3 = np.random.multivariate_normal([8, 2], [[0.25, 0.05], [0.05, 0.25]], 80)
scattered = np.column_stack([np.random.uniform(0, 10, 50), np.random.uniform(0, 7, 50)])

all_points = np.vstack([cluster1, cluster2, cluster3, scattered])

# 给每个点赋产业产值
cluster_vals = np.concatenate([
    np.random.lognormal(3, 0.3, 120),  # 集群1高产值
    np.random.lognormal(3.2, 0.3, 100),  # 集群2更高
    np.random.lognormal(2.8, 0.3, 80),   # 集群3较高
    np.random.lognormal(1.5, 0.5, 50)    # 离散点低产值
])

sc = ax.scatter(all_points[:, 0], all_points[:, 1], c=cluster_vals, 
                cmap='YlOrRd', s=40, alpha=0.7, edgecolors='gray', linewidth=0.3)
cbar = plt.colorbar(sc, ax=ax, label='企业产值')

# 圈出集群
from matplotlib.patches import Ellipse
for center, w, h, angle in [([2, 2], 1.5, 1.2, 0), ([6, 5], 1.8, 1.5, 0.3), ([8, 2], 1.2, 1.0, -0.2)]:
    ellipse = Ellipse(center, w, h, angle=angle, facecolor='none', edgecolor='#E74C3C', linewidth=2, linestyle='--')
    ax.add_patch(ellipse)

ax.set_xlabel('X 空间坐标', fontsize=12, fontproperties=chinese_font_prop)
ax.set_ylabel('Y 空间坐标', fontsize=12, fontproperties=chinese_font_prop)
ax.set_title('产业空间集聚分布\n(集群内企业产值显著高于离散企业)', fontsize=13, fontproperties=chinese_font_prop)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)

# 右图: 区位商对比
ax = axes[1]
# 选取几个典型产业
industries = ['高科技产业', '金融服务业', '制造业', '文化创意', '物流运输', '传统农业']
# 大城市 vs 小城市的 LQ 对比
big_city_lq = [1.85, 1.62, 0.72, 1.55, 0.95, 0.31]
small_city_lq = [0.45, 0.38, 1.25, 0.52, 0.85, 1.58]

x_pos = np.arange(len(industries))
width = 0.35

bars1 = ax.bar(x_pos - width/2, big_city_lq, width, label='超大城市', color='#E74C3C', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, small_city_lq, width, label='小城市', color='#4A90D9', alpha=0.8)

# 在 LQ=1 处画参考线
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(5.5, 1.05, 'LQ=1 (全国平均水平)', fontsize=9, color='gray', fontproperties=chinese_font_prop)

# 标注LQ>1
ax.annotate('LQ>1: 该产业专业化程度\n高于全国平均，具有比较优势',
            xy=(0.5, 2.8), fontsize=10, color='green', fontproperties=chinese_font_prop)

ax.set_xticks(x_pos)
ax.set_xticklabels(industries, fontsize=10, fontproperties=chinese_font_prop)
ax.set_ylabel('区位商 (LQ)', fontsize=12, fontproperties=chinese_font_prop)
ax.set_title('区位商对比: 超大城市 vs 小城市\n(区位商>1 表示有比较优势)', fontsize=13, fontproperties=chinese_font_prop)
ax.legend(fontsize=11, prop=chinese_font_prop)
ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig(f'{output_dir}/05_spatial_clusters_and_LQ.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 图5: 空间集聚 + 区位商")

# ---------- 图6: 汇总仪表盘 (关键数字) ----------
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')

# 计算关键数字
top10_mask = population >= np.percentile(population, 90)
bottom10_mask = population <= np.percentile(population, 10)

# 按城市规模等级统计
city_tiers = {
    '超大城市\n(>1000万)': population >= 1000,
    '大城市\n(500-1000万)': (population >= 500) & (population < 1000),
    '中等城市\n(100-500万)': (population >= 100) & (population < 500),
    '小城市\n(<100万)': population < 100,
}

print("\n===== 集聚效益关键数据汇总 =====")
print(f"{'城市等级':<20} {'数量':<8} {'人口占比':<10} {'GDP占比':<10} {'人均GDP(万元)':<15} {'人均专利':<10}")
print("="*80)
for tier_name, mask in city_tiers.items():
    count = mask.sum()
    pop_share = population[mask].sum() / population.sum() * 100
    gdp_share = gdp[mask].sum() / gdp.sum() * 100
    avg_gdp_pc = gdp_per_capita[mask].mean()
    avg_patent = patents[mask].mean()
    tier_name_clean = tier_name.replace('\n', ' ')
    print(f"{tier_name_clean:<20} {count:<8} {pop_share:<10.1f} {gdp_share:<10.1f} {avg_gdp_pc:<15.2f} {avg_patent:<10.0f}")

# 集聚效益倍数
pop_share_top10 = population[top10_mask].sum() / population.sum() * 100
gdp_share_top10 = gdp[top10_mask].sum() / gdp.sum() * 100
gdp_pc_ratio = gdp_per_capita[top10_mask].mean() / gdp_per_capita[bottom10_mask].mean()
wage_ratio = wage[top10_mask].mean() / wage[bottom10_mask].mean()
patent_ratio = patents[top10_mask].mean() / patents[bottom10_mask].mean()

print(f"\n===== 集聚效益倍数 =====")
print(f"前10%城市 vs 后10%城市:")
print(f"  人均GDP 倍数: {gdp_pc_ratio:.2f}x")
print(f"  平均工资倍数: {wage_ratio:.2f}x")
print(f"  人均专利倍数: {patent_ratio:.2f}x")
print(f"  前10%城市贡献了 {gdp_share_top10:.1f}% 的GDP，只占 {pop_share_top10:.1f}% 的人口")
print(f"  标度律指数 β = {slope:.3f} (β>1 表示集聚效益)")

# ============================================================
# 生成汇总信息图
# ============================================================
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

summary_text = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     集聚效益 (Agglomeration Economies) 模拟分析报告           ║
╚══════════════════════════════════════════════════════════════════════════════╝

一、城市标度律分析
────────────────────────────────────────────────────────────────────────────────
  城市规模与 GDP 的标度关系:  β = {slope:.3f}  (β > 1 表示超线性增长)
  判定系数 R² = {r_squared:.3f}
  含义: 城市人口每增长 1%，经济产出增长约 {slope:.2f}%，即集聚效益带来的额外增益

二、关键对比（前10%最大城市 vs 后10%最小城市）
────────────────────────────────────────────────────────────────────────────────
  • 人均 GDP 倍数:     {gdp_pc_ratio:.2f} 倍
  • 平均工资倍数:     {wage_ratio:.2f} 倍  (工资溢价)
  • 人均专利倍数:     {patent_ratio:.2f} 倍  (创新集聚)
  • 前10%城市贡献了 {gdp_share_top10:.1f}% 的 GDP，仅占 {pop_share_top10:.1f}% 的人口

三、主要发现
────────────────────────────────────────────────────────────────────────────────
  1. 集聚效益确实存在：人口密度与人均产出呈显著正相关，弹性约 {slope2:.3f}
  2. 工资溢价现象：大城市平均工资显著高于小城市，弹性约 {slope3:.3f}
  3. 倒U型关系：集聚效益并非无限增大，过高的集聚度会带来拥堵效应
  4. 产业分工：大城市在高科技、金融等现代服务业具有比较优势（LQ > 1）
     小城市在传统制造业、农业等领域具有比较优势

四、数据来源说明
────────────────────────────────────────────────────────────────────────────────
  以上数据为模拟生成，用于展示集聚效益的分析框架和方法。
  实际应用可替换为真实数据（国家统计局、城市统计年鉴、百度迁徙等）。
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
        fontfamily='monospace', verticalalignment='top',
        bbox=dict(boxstyle='round,pad=1', facecolor='white', alpha=0.9))

plt.savefig(f'{output_dir}/06_summary_report.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 图6: 汇总报告")

print(f"\n所有图表已生成到 {output_dir}/ 目录下:")
print(f"  01_scaling_law.png          - 城市标度律")
print(f"  02_density_vs_gdp.png       - 密度 vs 人均GDP")
print(f"  03_group_comparison.png     - 分组对比")
print(f"  04_inverted_U_and_wage.png  - 倒U型曲线 + 工资溢价")
print(f"  05_spatial_clusters_and_LQ.png - 空间集聚 + 区位商")
print(f"  06_summary_report.png       - 汇总报告")