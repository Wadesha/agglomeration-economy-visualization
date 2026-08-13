"""
集聚效益 - 普通人体感版
用普通人日常能感知到的数据展示集聚效益
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'AR PL UMing CN', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

from matplotlib.font_manager import FontProperties
import os

def find_chinese_font():
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

np.random.seed(42)
output_dir = '/workspace'

# ============================================================
# 模拟 4 种典型城市的数据（普通人能感知的维度）
# ============================================================
city_types = ['小县城\n(20万)', '地级市\n(200万)', '省会城市\n(800万)', '一线城市\n(2000万+)']
# 对应的城市规模
city_sizes = [20, 200, 800, 2500]

# 每个类型模拟 50 个样本
n_samples = 50
all_data = {}

# ---- 1. 外卖/餐饮 ----
# 每万人拥有的外卖商家数
delivery_per_10k = {
    '小县城': np.random.normal(8, 2, n_samples),
    '地级市': np.random.normal(25, 5, n_samples),
    '省会城市': np.random.normal(55, 8, n_samples),
    '一线城市': np.random.normal(90, 12, n_samples),
}
# 外卖平均送达时间（分钟）—— 大城市反而快，因为商家密集
delivery_time = {
    '小县城': np.random.normal(45, 8, n_samples),
    '地级市': np.random.normal(38, 5, n_samples),
    '省会城市': np.random.normal(30, 4, n_samples),
    '一线城市': np.random.normal(26, 3, n_samples),
}
# 餐饮种类得分
food_variety = {
    '小县城': np.random.normal(3, 1, n_samples),
    '地级市': np.random.normal(5.5, 1.2, n_samples),
    '省会城市': np.random.normal(8, 1.5, n_samples),
    '一线城市': np.random.normal(9.5, 0.8, n_samples),
}

# ---- 2. 出行/交通 ----
# 每公里地铁线路数
subway_density = {
    '小县城': np.random.normal(0, 0, n_samples),
    '地级市': np.random.normal(0.3, 0.3, n_samples),
    '省会城市': np.random.normal(2.5, 0.8, n_samples),
    '一线城市': np.random.normal(6, 1.5, n_samples),
}
# 平均通勤时间（分钟）—— 大城市更长，这是集聚的代价
commute_time = {
    '小县城': np.random.normal(15, 5, n_samples),
    '地级市': np.random.normal(28, 7, n_samples),
    '省会城市': np.random.normal(40, 8, n_samples),
    '一线城市': np.random.normal(55, 10, n_samples),
}
# 共享单车密度（辆/平方公里）
bike_share = {
    '小县城': np.random.normal(5, 3, n_samples),
    '地级市': np.random.normal(30, 10, n_samples),
    '省会城市': np.random.normal(80, 20, n_samples),
    '一线城市': np.random.normal(150, 30, n_samples),
}

# ---- 3. 工作/收入 ----
# 平均月薪（元）
monthly_salary = {
    '小县城': np.random.normal(3500, 500, n_samples),
    '地级市': np.random.normal(5500, 800, n_samples),
    '省会城市': np.random.normal(8500, 1500, n_samples),
    '一线城市': np.random.normal(13000, 3000, n_samples),
}
# 附近1公里内的工作岗位数（万）
nearby_jobs = {
    '小县城': np.random.normal(0.3, 0.1, n_samples),
    '地级市': np.random.normal(2, 0.5, n_samples),
    '省会城市': np.random.normal(8, 2, n_samples),
    '一线城市': np.random.normal(25, 5, n_samples),
}
# 跳槽机会（每万人招聘岗位数）
job_opportunities = {
    '小县城': np.random.normal(5, 2, n_samples),
    '地级市': np.random.normal(20, 5, n_samples),
    '省会城市': np.random.normal(50, 10, n_samples),
    '一线城市': np.random.normal(80, 15, n_samples),
}

# ---- 4. 生活便利 ----
# 便利店密度（家/平方公里）
convenience_store = {
    '小县城': np.random.normal(1.5, 0.5, n_samples),
    '地级市': np.random.normal(5, 1.5, n_samples),
    '省会城市': np.random.normal(15, 4, n_samples),
    '一线城市': np.random.normal(30, 8, n_samples),
}
# 咖啡店密度（家/平方公里）
coffee_shop = {
    '小县城': np.random.normal(0.3, 0.2, n_samples),
    '地级市': np.random.normal(2, 0.8, n_samples),
    '省会城市': np.random.normal(8, 3, n_samples),
    '一线城市': np.random.normal(20, 5, n_samples),
}
# 24小时营业场所（便利店/药店等，家/平方公里）
night_services = {
    '小县城': np.random.normal(0.5, 0.3, n_samples),
    '地级市': np.random.normal(3, 1, n_samples),
    '省会城市': np.random.normal(10, 3, n_samples),
    '一线城市': np.random.normal(25, 6, n_samples),
}

# ---- 5. 社交/娱乐 ----
# 每月可参加的社交活动（场）
social_events = {
    '小县城': np.random.normal(1, 0.5, n_samples),
    '地级市': np.random.normal(3, 1, n_samples),
    '省会城市': np.random.normal(8, 2, n_samples),
    '一线城市': np.random.normal(20, 5, n_samples),
}
# 演唱会/展览/演出频率（场/月）
entertainment = {
    '小县城': np.random.normal(0.2, 0.2, n_samples),
    '地级市': np.random.normal(1.5, 0.8, n_samples),
    '省会城市': np.random.normal(8, 3, n_samples),
    '一线城市': np.random.normal(25, 8, n_samples),
}
# 交友软件匹配对象数量（匹配人数/周）
dating_options = {
    '小县城': np.random.normal(3, 2, n_samples),
    '地级市': np.random.normal(15, 5, n_samples),
    '省会城市': np.random.normal(40, 10, n_samples),
    '一线城市': np.random.normal(80, 20, n_samples),
}

# ---- 6. 生活成本（集聚的代价） ----
# 房租（元/月，一居室）
rent = {
    '小县城': np.random.normal(500, 150, n_samples),
    '地级市': np.random.normal(1200, 300, n_samples),
    '省会城市': np.random.normal(2500, 600, n_samples),
    '一线城市': np.random.normal(5000, 1500, n_samples),
}
# 房价（元/平米）
housing_price = {
    '小县城': np.random.normal(4000, 1000, n_samples),
    '地级市': np.random.normal(8000, 2000, n_samples),
    '省会城市': np.random.normal(18000, 5000, n_samples),
    '一线城市': np.random.normal(45000, 15000, n_samples),
}

# 存钱速度 = 月薪 - 房租 - 基本生活开销（元/月）
savings_rate = {}
for key in ['小县城', '地级市', '省会城市', '一线城市']:
    base_cost = {'小县城': 1500, '地级市': 2000, '省会城市': 3000, '一线城市': 4000}
    savings_rate[key] = monthly_salary[key] - rent[key] - base_cost[key]

# ---- 生活便利综合指数 ----
# 综合打分（0-100），融合了外卖、交通、便利店、社交等
convenience_index = {}
for key, weight in [('小县城', 0.5), ('地级市', 0.7), ('省会城市', 0.9), ('一线城市', 1.0)]:
    convenience_index[key] = (
        delivery_per_10k[key] / 90 * 25 +
        food_variety[key] / 10 * 15 +
        bike_share[key] / 150 * 10 +
        convenience_store[key] / 30 * 15 +
        coffee_shop[key] / 20 * 10 +
        social_events[key] / 20 * 15 +
        night_services[key] / 25 * 10
    ) * weight + np.random.normal(0, 3, n_samples)
    convenience_index[key] = np.clip(convenience_index[key], 0, 100)


# ============================================================
# 图表生成
# ============================================================
print("正在生成普通人体感版图表...")

# ---------- 图1: 普通人生活"痛点 vs 爽点"雷达图 ----------
fig, axes = plt.subplots(1, 4, figsize=(18, 5), subplot_kw=dict(polar=True))

# 选择6个代表性维度，进行归一化
radar_labels = ['外卖速度\n(快)', '收入水平\n(高)', '娱乐选择\n(多)', '社交机会\n(多)', '生活便利\n(高)', '通勤时间\n(短)']
# 注意通勤时间需要反转（越小越好）
# 最大值参考
max_vals = {
    '小县城': [8, 3500, 0.2, 1, 15, 45],
    '地级市': [25, 5500, 1.5, 3, 28, 38],
    '省会城市': [55, 8500, 8, 8, 40, 30],
    '一线城市': [90, 13000, 25, 20, 55, 26],
}
# 归一化到 0-100 分
def normalize(val, max_val, reverse=False):
    if reverse:
        return max(0, min(100, (1 - val / max_val) * 100))
    return max(0, min(100, val / max_val * 100))

for idx, (key, ax) in enumerate(zip(['小县城', '地级市', '省会城市', '一线城市'], axes)):
    values = [
        normalize(delivery_per_10k[key].mean(), 90, False),      # 外卖商家密度
        normalize(monthly_salary[key].mean(), 13000, False),      # 月薪
        normalize(entertainment[key].mean(), 25, False),          # 娱乐
        normalize(social_events[key].mean(), 20, False),          # 社交
        normalize(convenience_index[key].mean(), 100, False),     # 生活便利
        normalize(commute_time[key].mean(), 55, True),            # 通勤（反转）
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    colors = ['#95A5A6', '#3498DB', '#F39C12', '#E74C3C']
    ax.fill(angles, values, color=colors[idx], alpha=0.25)
    ax.plot(angles, values, color=colors[idx], linewidth=2, label=key)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=8, fontproperties=chinese_font_prop)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=7)
    ax.set_title(key, fontsize=14, fontproperties=chinese_font_prop, color=colors[idx], fontweight='bold')

fig.suptitle('普通人生活体验雷达图: 城市规模越大 = 生活越便利?', fontsize=16, fontproperties=chinese_font_prop)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(f'{output_dir}/D1_radar_daily_life.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ D1: 雷达图")

# ---------- 图2: 收入 vs 房租（"北上广深 vs 老家"对比） ----------
fig, ax = plt.subplots(figsize=(10, 7))

x_pos = np.arange(len(city_types))
width = 0.35

means_salary = [monthly_salary[k].mean() for k in ['小县城', '地级市', '省会城市', '一线城市']]
means_rent = [rent[k].mean() for k in ['小县城', '地级市', '省会城市', '一线城市']]
means_savings = [savings_rate[k].mean() for k in ['小县城', '地级市', '省会城市', '一线城市']]

bars1 = ax.bar(x_pos - width/2, means_salary, width, label='平均月薪', color='#2ECC71', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, means_rent, width, label='月租(一居室)', color='#E74C3C', alpha=0.8)

# 标注数值
for i, (s, r) in enumerate(zip(means_salary, means_rent)):
    ax.text(i - width/2, s + 200, f'{s:.0f}', ha='center', fontsize=10, fontweight='bold')
    ax.text(i + width/2, r + 200, f'{r:.0f}', ha='center', fontsize=10, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(city_types, fontsize=11, fontproperties=chinese_font_prop)
ax.set_ylabel('元/月', fontsize=12, fontproperties=chinese_font_prop)
ax.set_title('月薪 vs 房租: 大城市赚得多，但花得也多', fontsize=15, fontproperties=chinese_font_prop)
ax.legend(fontsize=12, prop=chinese_font_prop)
ax.grid(True, alpha=0.3, axis='y')

# 添加每月可存钱注释
note_text = '每月可存钱 (月薪 - 房租 - 基本生活费):\n'
for i, k in enumerate(['小县城', '地级市', '省会城市', '一线城市']):
    sv = savings_rate[k].mean()
    note_text += f'  {k.replace(chr(10), "")}: {sv:.0f}元/月\n'
ax.annotate(note_text, xy=(0.02, 0.97), xycoords='axes fraction', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9),
            verticalalignment='top', fontproperties=chinese_font_prop)

plt.tight_layout()
plt.savefig(f'{output_dir}/D2_income_vs_rent.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ D2: 收入 vs 房租")

# ---------- 图3: 生活便利 vs 通勤时间（"痛并快乐着"） ----------
fig, ax = plt.subplots(figsize=(10, 7))

# 四类城市分别取点
city_keys = ['小县城', '地级市', '省会城市', '一线城市']
city_colors = ['#95A5A6', '#3498DB', '#F39C12', '#E74C3C']
city_markers = ['o', 's', 'D', '^']

for key, color, marker in zip(city_keys, city_colors, city_markers):
    x = commute_time[key]
    y = convenience_index[key]
    ax.scatter(x, y, c=color, s=60, alpha=0.6, marker=marker, label=key, edgecolors='white', linewidth=0.5)
    # 标注均值点
    ax.scatter(x.mean(), y.mean(), c=color, s=200, marker='*', edgecolors='black', linewidth=1, zorder=5)
    ax.annotate(f'{key.replace(chr(10), "")}\n(均值为★)', 
                xy=(x.mean(), y.mean()), xytext=(x.mean() + 3, y.mean() + 2),
                fontsize=10, color=color, fontweight='bold', fontproperties=chinese_font_prop)

ax.set_xlabel('平均通勤时间 (分钟)', fontsize=13, fontproperties=chinese_font_prop)
ax.set_ylabel('生活便利综合指数 (0-100)', fontsize=13, fontproperties=chinese_font_prop)
ax.set_title('"痛并快乐着": 通勤时间越长，生活越便利?', fontsize=15, fontproperties=chinese_font_prop)
ax.legend(fontsize=11, prop=chinese_font_prop)
ax.grid(True, alpha=0.3)

# 分割线
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.4)
ax.axvline(x=30, color='gray', linestyle='--', alpha=0.4)
ax.text(5, 52, '便利>50', fontsize=9, color='gray')
ax.text(32, 5, '通勤>30min', fontsize=9, color='gray')

ax.annotate('小县城: 通勤15分钟，但没啥可玩的', xy=(12, 15), fontsize=9,
            fontproperties=chinese_font_prop, color='#95A5A6')
ax.annotate('一线城市: 通勤1小时，但什么都有', xy=(50, 82), fontsize=9,
            fontproperties=chinese_font_prop, color='#E74C3C')

plt.tight_layout()
plt.savefig(f'{output_dir}/D3_convenience_vs_commute.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ D3: 便利 vs 通勤")

# ---------- 图4: 日常生活的"集聚红利"对比气泡图 ----------
fig, ax = plt.subplots(figsize=(12, 8))

# 用气泡图综合展示多个维度
# X轴: 城市规模, Y轴: 月薪, 气泡大小: 娱乐选择, 颜色: 房租高低
for key, color, marker in zip(city_keys, city_colors, city_markers):
    pop = {'小县城': 20, '地级市': 200, '省会城市': 800, '一线城市': 2500}[key]
    salary_mean = monthly_salary[key].mean()
    ent_mean = entertainment[key].mean()
    rent_mean = rent[key].mean()
    
    ax.scatter(pop, salary_mean, s=ent_mean * 30, c=[rent_mean], 
               cmap='YlOrRd', vmin=0, vmax=6000, alpha=0.7, edgecolors='black', linewidth=1)
    ax.annotate(f'{key.replace(chr(10), "")}\n月薪{salary_mean:.0f}元\n房租{rent_mean:.0f}元\n娱乐{ent_mean:.1f}场/月',
                xy=(pop, salary_mean), fontsize=10, fontproperties=chinese_font_prop,
                ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('城市人口 (万人)', fontsize=13, fontproperties=chinese_font_prop)
ax.set_ylabel('平均月薪 (元)', fontsize=13, fontproperties=chinese_font_prop)
ax.set_title('集聚红利 vs 成本: 城市越大 → 薪资越高、娱乐越多、但房租也越贵', 
             fontsize=14, fontproperties=chinese_font_prop)
ax.grid(True, alpha=0.3)

# 添加图例说明
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='气泡大小 = 娱乐丰富度'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', markersize=10, label='颜色越深 = 房租越高'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10, prop=chinese_font_prop)

plt.tight_layout()
plt.savefig(f'{output_dir}/D4_bubble_red_vs_cost.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ D4: 红利 vs 成本气泡图")

# ---------- 图5: "小镇 vs 大城" 生活对比清单 ----------
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

# 用表格形式呈现
rows = [
    ('维度', '小县城 (20万)', '地级市 (200万)', '省会城市 (800万)', '一线城市 (2000万+)'),
    ('外卖商家密度', f'{delivery_per_10k["小县城"].mean():.0f} 家/万人', 
     f'{delivery_per_10k["地级市"].mean():.0f} 家/万人', 
     f'{delivery_per_10k["省会城市"].mean():.0f} 家/万人',
     f'{delivery_per_10k["一线城市"].mean():.0f} 家/万人'),
    ('外卖送达时间', f'{delivery_time["小县城"].mean():.0f} 分钟',
     f'{delivery_time["地级市"].mean():.0f} 分钟',
     f'{delivery_time["省会城市"].mean():.0f} 分钟',
     f'{delivery_time["一线城市"].mean():.0f} 分钟'),
    ('餐饮种类评分', f'{food_variety["小县城"].mean():.1f}/10',
     f'{food_variety["地级市"].mean():.1f}/10',
     f'{food_variety["省会城市"].mean():.1f}/10',
     f'{food_variety["一线城市"].mean():.1f}/10'),
    ('平均通勤时间', f'{commute_time["小县城"].mean():.0f} 分钟',
     f'{commute_time["地级市"].mean():.0f} 分钟',
     f'{commute_time["省会城市"].mean():.0f} 分钟',
     f'{commute_time["一线城市"].mean():.0f} 分钟'),
    ('共享单车密度', f'{bike_share["小县城"].mean():.0f} 辆/km²',
     f'{bike_share["地级市"].mean():.0f} 辆/km²',
     f'{bike_share["省会城市"].mean():.0f} 辆/km²',
     f'{bike_share["一线城市"].mean():.0f} 辆/km²'),
    ('平均月薪', f'{monthly_salary["小县城"].mean():.0f} 元',
     f'{monthly_salary["地级市"].mean():.0f} 元',
     f'{monthly_salary["省会城市"].mean():.0f} 元',
     f'{monthly_salary["一线城市"].mean():.0f} 元'),
    ('附近工作机会', f'{nearby_jobs["小县城"].mean():.1f} 万个',
     f'{nearby_jobs["地级市"].mean():.1f} 万个',
     f'{nearby_jobs["省会城市"].mean():.1f} 万个',
     f'{nearby_jobs["一线城市"].mean():.1f} 万个'),
    ('一居室房租', f'{rent["小县城"].mean():.0f} 元/月',
     f'{rent["地级市"].mean():.0f} 元/月',
     f'{rent["省会城市"].mean():.0f} 元/月',
     f'{rent["一线城市"].mean():.0f} 元/月'),
    ('每月可存钱', f'{savings_rate["小县城"].mean():.0f} 元',
     f'{savings_rate["地级市"].mean():.0f} 元',
     f'{savings_rate["省会城市"].mean():.0f} 元',
     f'{savings_rate["一线城市"].mean():.0f} 元'),
    ('咖啡店密度', f'{coffee_shop["小县城"].mean():.1f} 家/km²',
     f'{coffee_shop["地级市"].mean():.1f} 家/km²',
     f'{coffee_shop["省会城市"].mean():.1f} 家/km²',
     f'{coffee_shop["一线城市"].mean():.1f} 家/km²'),
    ('社交活动', f'{social_events["小县城"].mean():.0f} 场/月',
     f'{social_events["地级市"].mean():.0f} 场/月',
     f'{social_events["省会城市"].mean():.0f} 场/月',
     f'{social_events["一线城市"].mean():.0f} 场/月'),
    ('演唱会/演出', f'{entertainment["小县城"].mean():.1f} 场/月',
     f'{entertainment["地级市"].mean():.1f} 场/月',
     f'{entertainment["省会城市"].mean():.1f} 场/月',
     f'{entertainment["一线城市"].mean():.1f} 场/月'),
    ('交友匹配对象', f'{dating_options["小县城"].mean():.0f} 人/周',
     f'{dating_options["地级市"].mean():.0f} 人/周',
     f'{dating_options["省会城市"].mean():.0f} 人/周',
     f'{dating_options["一线城市"].mean():.0f} 人/周'),
    ('24h营业场所', f'{night_services["小县城"].mean():.1f} 家/km²',
     f'{night_services["地级市"].mean():.1f} 家/km²',
     f'{night_services["省会城市"].mean():.1f} 家/km²',
     f'{night_services["一线城市"].mean():.1f} 家/km²'),
    ('生活便利指数', f'{convenience_index["小县城"].mean():.0f}/100',
     f'{convenience_index["地级市"].mean():.0f}/100',
     f'{convenience_index["省会城市"].mean():.0f}/100',
     f'{convenience_index["一线城市"].mean():.0f}/100'),
]

# 创建表格
table = ax.table(cellText=rows[1:], colLabels=rows[0], 
                 cellLoc='center', loc='center',
                 colWidths=[0.18, 0.18, 0.18, 0.18, 0.18])

# 美化表格
table.auto_set_font_size(False)
table.set_fontsize(10)
# 表头样式
for j in range(5):
    cell = table[0, j]
    cell.set_facecolor('#2C3E50')
    cell.set_text_props(color='white', fontweight='bold', fontproperties=chinese_font_prop)
# 隔行变色
for i in range(1, len(rows)):
    for j in range(5):
        cell = table[i, j]
        if i % 2 == 0:
            cell.set_facecolor('#F2F3F4')
        else:
            cell.set_facecolor('white')
        # 第一列加粗
        if j == 0:
            cell.set_text_props(fontweight='bold', fontproperties=chinese_font_prop)
        else:
            cell.set_text_props(fontproperties=chinese_font_prop)

# 缩放表格
table.scale(1, 1.5)

ax.set_title('普通人生活体验清单: 小镇 vs 大城', fontsize=18, fontproperties=chinese_font_prop, 
             fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f'{output_dir}/D5_comparison_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ D5: 对比清单表格")

# ---------- 图6: "大城市值得吗?" 综合权衡 ----------
fig, ax = plt.subplots(figsize=(10, 7))

# 维度: 收入/机会/便利 vs 房租/通勤/竞争
# 用分组柱状图展示"得到"和"付出"
categories = ['月薪', '工作机会', '生活便利', '娱乐丰富', '社交机会', '餐饮选择']
gains = {
    '小县城': [normalize(monthly_salary['小县城'].mean(), 13000, False),
              normalize(nearby_jobs['小县城'].mean(), 25, False),
              normalize(convenience_index['小县城'].mean(), 100, False),
              normalize(entertainment['小县城'].mean(), 25, False),
              normalize(social_events['小县城'].mean(), 20, False),
              normalize(food_variety['小县城'].mean(), 10, False)],
    '一线城市': [normalize(monthly_salary['一线城市'].mean(), 13000, False),
               normalize(nearby_jobs['一线城市'].mean(), 25, False),
               normalize(convenience_index['一线城市'].mean(), 100, False),
               normalize(entertainment['一线城市'].mean(), 25, False),
               normalize(social_events['一线城市'].mean(), 20, False),
               normalize(food_variety['一线城市'].mean(), 10, False)],
}
costs = {
    '小县城': [normalize(rent['小县城'].mean(), 5000, False),
              normalize(commute_time['小县城'].mean(), 55, True),
              normalize(50, 50, False),  # 竞争压力(人为设定)
              normalize(20, 20, False),  # 无聊感(人为设定)
              normalize(30, 30, False),  # 发展受限(人为设定)
              normalize(15, 15, False)],  # 选择匮乏(人为设定)
    '一线城市': [normalize(rent['一线城市'].mean(), 5000, False),
               normalize(commute_time['一线城市'].mean(), 55, True),
               normalize(80, 50, False),
               normalize(85, 20, False),
               normalize(80, 30, False),
               normalize(85, 15, False)],
}

x = np.arange(len(categories))
width = 0.2

# 小县城: 得到 vs 付出
ax.bar(x - width*1.5, gains['小县城'], width, label='小县城-得到', color='#3498DB', alpha=0.8)
ax.bar(x - width*0.5, costs['小县城'], width, label='小县城-付出', color='#85C1E9', alpha=0.6)
# 一线城市: 得到 vs 付出
ax.bar(x + width*0.5, gains['一线城市'], width, label='一线城市-得到', color='#E74C3C', alpha=0.8)
ax.bar(x + width*1.5, costs['一线城市'], width, label='一线城市-付出', color='#F1948A', alpha=0.6)

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12, fontproperties=chinese_font_prop)
ax.set_ylabel('得分 (0-100 归一化)', fontsize=12, fontproperties=chinese_font_prop)
ax.set_title('"得到 vs 付出": 一线城市 vs 小县域的全面权衡', fontsize=15, fontproperties=chinese_font_prop)
ax.legend(fontsize=11, loc='upper left', prop=chinese_font_prop)
ax.grid(True, alpha=0.3, axis='y')

# 净收益计算
for i, cat in enumerate(categories):
    net_small = gains['小县城'][i] - costs['小县城'][i]
    net_big = gains['一线城市'][i] - costs['一线城市'][i]
    ax.annotate(f'净: {net_small:.0f}', xy=(i - width*1.5, gains['小县城'][i] + 1), 
                fontsize=7, color='#3498DB', ha='center')
    ax.annotate(f'净: {net_big:.0f}', xy=(i + width*1.5, gains['一线城市'][i] + 1), 
                fontsize=7, color='#E74C3C', ha='center')

plt.tight_layout()
plt.savefig(f'{output_dir}/D6_tradeoff.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ D6: 得到 vs 付出权衡图")

# ---------- 汇总信息 ----------
print(f"\n===== 普通人体感版集聚效益 =====")
print(f"所有图表已生成到 {output_dir}/ 目录下:")
print(f"  D1_radar_daily_life.png          - 生活体验雷达图")
print(f"  D2_income_vs_rent.png            - 收入 vs 房租对比")
print(f"  D3_convenience_vs_commute.png    - 生活便利 vs 通勤时间")
print(f"  D4_bubble_red_vs_cost.png        - 红利 vs 成本气泡图")
print(f"  D5_comparison_table.png          - 生活对比清单表格")
print(f"  D6_tradeoff.png                  - 得到 vs 付出权衡图")