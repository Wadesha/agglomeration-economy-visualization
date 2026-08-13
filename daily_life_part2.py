"""
集聚效益 - 普通人体感版 Part 2
更多贴近生活的有趣维度
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置字体
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

# 四类城市
city_keys = ['小县城\n(20万)', '地级市\n(200万)', '省会城市\n(800万)', '一线城市\n(2000万+)']
city_keys_flat = ['小县城', '地级市', '省会城市', '一线城市']
n_samples = 50

# ============================================================
# 新维度数据
# ============================================================

# ---- 1. 医疗 ----
# 三甲医院数量
hospital_rank3 = {
    '小县城': np.random.poisson(0.3, n_samples),
    '地级市': np.random.poisson(3, n_samples),
    '省会城市': np.random.poisson(15, n_samples),
    '一线城市': np.random.poisson(40, n_samples),
}
# 挂专家号等待天数
specialist_wait = {
    '小县城': np.random.normal(14, 5, n_samples),    # 小县城可能要等很久
    '地级市': np.random.normal(10, 3, n_samples),
    '省会城市': np.random.normal(7, 2, n_samples),
    '一线城市': np.random.normal(5, 1.5, n_samples),  # 虽然人多但医生也多
}
# 诊所/药店密度（家/平方公里）
clinic_density = {
    '小县城': np.random.normal(0.5, 0.2, n_samples),
    '地级市': np.random.normal(2, 0.5, n_samples),
    '省会城市': np.random.normal(6, 1.5, n_samples),
    '一线城市': np.random.normal(12, 3, n_samples),
}
# 120急救平均到达时间（分钟）
ambulance_time = {
    '小县城': np.random.normal(20, 7, n_samples),
    '地级市': np.random.normal(15, 4, n_samples),
    '省会城市': np.random.normal(12, 3, n_samples),
    '一线城市': np.random.normal(10, 2, n_samples),
}

# ---- 2. 教育 ----
# 学区房溢价（同地段非学区 vs 学区房差价，元/平米）
school_district_premium = {
    '小县城': np.random.normal(500, 200, n_samples),
    '地级市': np.random.normal(2000, 500, n_samples),
    '省会城市': np.random.normal(5000, 1500, n_samples),
    '一线城市': np.random.normal(12000, 4000, n_samples),
}
# 补习班密度（家/万人）
tutoring_density = {
    '小县城': np.random.normal(1.5, 0.5, n_samples),
    '地级市': np.random.normal(5, 1.5, n_samples),
    '省会城市': np.random.normal(12, 3, n_samples),
    '一线城市': np.random.normal(20, 5, n_samples),
}
# 985/211大学数量
top_universities = {
    '小县城': np.random.poisson(0, n_samples),
    '地级市': np.random.poisson(0.2, n_samples),
    '省会城市': np.random.poisson(3, n_samples),
    '一线城市': np.random.poisson(8, n_samples),
}
# 图书馆/书店密度
library_density = {
    '小县城': np.random.normal(0.1, 0.05, n_samples),
    '地级市': np.random.normal(0.5, 0.2, n_samples),
    '省会城市': np.random.normal(1.5, 0.5, n_samples),
    '一线城市': np.random.normal(3, 1, n_samples),
}

# ---- 3. 夜生活/年轻人消费 ----
# 酒吧/夜店密度（家/平方公里）
bar_density = {
    '小县城': np.random.normal(0.1, 0.1, n_samples),
    '地级市': np.random.normal(0.8, 0.3, n_samples),
    '省会城市': np.random.normal(3, 1, n_samples),
    '一线城市': np.random.normal(8, 2, n_samples),
}
# 深夜营业（0点后）商家比例（%）
late_night_ratio = {
    '小县城': np.random.normal(2, 1, n_samples),
    '地级市': np.random.normal(8, 3, n_samples),
    '省会城市': np.random.normal(20, 5, n_samples),
    '一线城市': np.random.normal(35, 8, n_samples),
}
# 密室/剧本杀/桌游店密度
escape_room = {
    '小县城': np.random.normal(0.05, 0.05, n_samples),
    '地级市': np.random.normal(0.3, 0.15, n_samples),
    '省会城市': np.random.normal(1.2, 0.4, n_samples),
    '一线城市': np.random.normal(3, 1, n_samples),
}

# ---- 4. 宠物经济 ----
# 宠物医院密度（家/10万人）
pet_hospital = {
    '小县城': np.random.normal(0.5, 0.3, n_samples),
    '地级市': np.random.normal(1.5, 0.5, n_samples),
    '省会城市': np.random.normal(3, 1, n_samples),
    '一线城市': np.random.normal(5, 1.5, n_samples),
}
# 宠物公园/宠物友好商场
pet_friendly = {
    '小县城': np.random.normal(0.3, 0.3, n_samples),
    '地级市': np.random.normal(2, 1, n_samples),
    '省会城市': np.random.normal(8, 3, n_samples),
    '一线城市': np.random.normal(20, 5, n_samples),
}
# 养宠比例（%）
pet_ownership = {
    '小县城': np.random.normal(8, 3, n_samples),
    '地级市': np.random.normal(15, 4, n_samples),
    '省会城市': np.random.normal(22, 5, n_samples),
    '一线城市': np.random.normal(25, 5, n_samples),
}

# ---- 5. 菜市场/物价（集聚带来的低价） ----
# 猪肉价格（元/斤）
pork_price = {
    '小县城': np.random.normal(22, 3, n_samples),
    '地级市': np.random.normal(20, 2, n_samples),
    '省会城市': np.random.normal(18, 2, n_samples),
    '一线城市': np.random.normal(17, 1.5, n_samples),  # 物流集聚反而便宜
}
# 蔬菜种类（种/菜市场）
vegetable_variety = {
    '小县城': np.random.normal(30, 8, n_samples),
    '地级市': np.random.normal(50, 10, n_samples),
    '省会城市': np.random.normal(80, 15, n_samples),
    '一线城市': np.random.normal(100, 15, n_samples),
}
# 菜市场密度（个/平方公里）
market_density = {
    '小县城': np.random.normal(0.2, 0.1, n_samples),
    '地级市': np.random.normal(0.5, 0.2, n_samples),
    '省会城市': np.random.normal(1.2, 0.4, n_samples),
    '一线城市': np.random.normal(2, 0.5, n_samples),
}
# 生鲜电商次日达覆盖率（%）
fresh_delivery = {
    '小县城': np.random.normal(15, 10, n_samples),
    '地级市': np.random.normal(50, 15, n_samples),
    '省会城市': np.random.normal(85, 10, n_samples),
    '一线城市': np.random.normal(98, 3, n_samples),
}

# ---- 6. 二手交易/循环经济 ----
# 闲鱼/转转活跃度（每万人发布量/天）
second_hand = {
    '小县城': np.random.normal(5, 3, n_samples),
    '地级市': np.random.normal(20, 8, n_samples),
    '省会城市': np.random.normal(60, 15, n_samples),
    '一线城市': np.random.normal(120, 30, n_samples),
}
# 二手物品成交速度（天）
second_hand_speed = {
    '小县城': np.random.normal(30, 10, n_samples),
    '地级市': np.random.normal(15, 5, n_samples),
    '省会城市': np.random.normal(7, 2, n_samples),
    '一线城市': np.random.normal(3, 1, n_samples),  # 人多，东西卖得快
}

# ---- 7. 孤独感悖论 ----
# 微信好友数（真实的活跃联系人）
active_contacts = {
    '小县城': np.random.normal(30, 10, n_samples),
    '地级市': np.random.normal(50, 15, n_samples),
    '省会城市': np.random.normal(70, 20, n_samples),
    '一线城市': np.random.normal(80, 25, n_samples),
}
# 每周实际见面朋友数
friends_met_weekly = {
    '小县城': np.random.normal(5, 2, n_samples),
    '地级市': np.random.normal(4, 1.5, n_samples),
    '省会城市': np.random.normal(3, 1.5, n_samples),
    '一线城市': np.random.normal(2.5, 1.5, n_samples),  # 人越多反而越孤独
}
# 孤独感自评（1-10，越高越孤独）
loneliness = {
    '小县城': np.random.normal(3, 1.5, n_samples),
    '地级市': np.random.normal(4, 1.5, n_samples),
    '省会城市': np.random.normal(5.5, 1.5, n_samples),
    '一线城市': np.random.normal(6.5, 1.5, n_samples),
}
# 社区活动参与频率（次/月）
community_activity = {
    '小县城': np.random.normal(3, 1.5, n_samples),
    '地级市': np.random.normal(2, 1, n_samples),
    '省会城市': np.random.normal(1, 0.8, n_samples),
    '一线城市': np.random.normal(0.5, 0.5, n_samples),
}

# ---- 8. 体育/健身 ----
# 健身房密度（家/万人）
gym_density = {
    '小县城': np.random.normal(0.3, 0.2, n_samples),
    '地级市': np.random.normal(1, 0.4, n_samples),
    '省会城市': np.random.normal(2.5, 0.8, n_samples),
    '一线城市': np.random.normal(4, 1.2, n_samples),
}
# 人均公园绿地面积（平米）
park_per_capita = {
    '小县城': np.random.normal(15, 5, n_samples),
    '地级市': np.random.normal(12, 3, n_samples),
    '省会城市': np.random.normal(10, 3, n_samples),
    '一线城市': np.random.normal(8, 2, n_samples),  # 人多，人均绿地少
}
# 运动场馆（篮球场/游泳馆等，个/10万人）
sports_venues = {
    '小县城': np.random.normal(5, 2, n_samples),
    '地级市': np.random.normal(8, 3, n_samples),
    '省会城市': np.random.normal(12, 4, n_samples),
    '一线城市': np.random.normal(15, 4, n_samples),
}
# 马拉松/城市跑活动（场/年）
running_events = {
    '小县城': np.random.normal(0.5, 0.5, n_samples),
    '地级市': np.random.normal(3, 2, n_samples),
    '省会城市': np.random.normal(12, 4, n_samples),
    '一线城市': np.random.normal(30, 8, n_samples),
}

# ---- 9. 停车 ----
# 平均停车费（元/小时）
parking_fee = {
    '小县城': np.random.normal(2, 1, n_samples),
    '地级市': np.random.normal(5, 2, n_samples),
    '省会城市': np.random.normal(10, 3, n_samples),
    '一线城市': np.random.normal(20, 5, n_samples),
}
# 找车位平均时间（分钟）
parking_search = {
    '小县城': np.random.normal(3, 2, n_samples),
    '地级市': np.random.normal(8, 4, n_samples),
    '省会城市': np.random.normal(15, 5, n_samples),
    '一线城市': np.random.normal(25, 8, n_samples),
}

# ---- 10. 快递/网购 ----
# 包邮率（%）
free_shipping = {
    '小县城': np.random.normal(40, 10, n_samples),
    '地级市': np.random.normal(65, 10, n_samples),
    '省会城市': np.random.normal(85, 8, n_samples),
    '一线城市': np.random.normal(95, 5, n_samples),
}
# 快递平均送达天数
delivery_days = {
    '小县城': np.random.normal(4.5, 1, n_samples),
    '地级市': np.random.normal(3.5, 0.8, n_samples),
    '省会城市': np.random.normal(2.5, 0.5, n_samples),
    '一线城市': np.random.normal(1.8, 0.4, n_samples),
}

# ---- 补充 from Part 1（用于交叉引用） ----
# 平均月薪（元）
monthly_salary = {
    '小县城': np.random.normal(3500, 500, n_samples),
    '地级市': np.random.normal(5500, 800, n_samples),
    '省会城市': np.random.normal(8500, 1500, n_samples),
    '一线城市': np.random.normal(13000, 3000, n_samples),
}
# 房租（元/月，一居室）
rent = {
    '小县城': np.random.normal(500, 150, n_samples),
    '地级市': np.random.normal(1200, 300, n_samples),
    '省会城市': np.random.normal(2500, 600, n_samples),
    '一线城市': np.random.normal(5000, 1500, n_samples),
}
# 平均通勤时间（分钟）
commute_time = {
    '小县城': np.random.normal(15, 5, n_samples),
    '地级市': np.random.normal(28, 7, n_samples),
    '省会城市': np.random.normal(40, 8, n_samples),
    '一线城市': np.random.normal(55, 10, n_samples),
}
# 外卖平均送达时间（分钟）
delivery_time = {
    '小县城': np.random.normal(45, 8, n_samples),
    '地级市': np.random.normal(38, 5, n_samples),
    '省会城市': np.random.normal(30, 4, n_samples),
    '一线城市': np.random.normal(26, 3, n_samples),
}
# 演唱会/展览/演出频率（场/月）
entertainment = {
    '小县城': np.random.normal(0.2, 0.2, n_samples),
    '地级市': np.random.normal(1.5, 0.8, n_samples),
    '省会城市': np.random.normal(8, 3, n_samples),
    '一线城市': np.random.normal(25, 8, n_samples),
}
# 跳槽机会（每万人招聘岗位数）
job_opportunities = {
    '小县城': np.random.normal(5, 2, n_samples),
    '地级市': np.random.normal(20, 5, n_samples),
    '省会城市': np.random.normal(50, 10, n_samples),
    '一线城市': np.random.normal(80, 15, n_samples),
}
# 24小时营业场所（家/平方公里）
night_services = {
    '小县城': np.random.normal(0.5, 0.3, n_samples),
    '地级市': np.random.normal(3, 1, n_samples),
    '省会城市': np.random.normal(10, 3, n_samples),
    '一线城市': np.random.normal(25, 6, n_samples),
}

# 存储所有数据以便后续使用
all_data = {
    '医疗': {
        '三甲医院数': hospital_rank3,
        '专家号等待(天)': specialist_wait,
        '药店密度': clinic_density,
        '120到达(分钟)': ambulance_time,
    },
    '教育': {
        '学区房溢价(元/㎡)': school_district_premium,
        '补习班密度': tutoring_density,
        '985/211大学数': top_universities,
        '图书馆/书店密度': library_density,
    },
    '夜生活': {
        '酒吧密度': bar_density,
        '深夜营业比例(%)': late_night_ratio,
        '密室/剧本杀密度': escape_room,
    },
    '宠物经济': {
        '宠物医院(家/10万人)': pet_hospital,
        '宠物友好场所': pet_friendly,
        '养宠比例(%)': pet_ownership,
    },
    '菜市/物价': {
        '猪肉价格(元/斤)': pork_price,
        '蔬菜种类': vegetable_variety,
        '菜市场密度': market_density,
        '生鲜配送覆盖率(%)': fresh_delivery,
    },
    '二手交易': {
        '闲鱼活跃度': second_hand,
        '成交速度(天)': second_hand_speed,
    },
    '社交/孤独': {
        '活跃联系人': active_contacts,
        '每周见面朋友': friends_met_weekly,
        '孤独感评分': loneliness,
        '社区活动(次/月)': community_activity,
    },
    '体育健身': {
        '健身房密度': gym_density,
        '人均公园绿地(㎡)': park_per_capita,
        '运动场馆': sports_venues,
        '马拉松/年': running_events,
    },
    '停车': {
        '停车费(元/小时)': parking_fee,
        '找车位(分钟)': parking_search,
    },
    '快递': {
        '包邮率(%)': free_shipping,
        '送达天数': delivery_days,
    },
}


# ============================================================
# 图表生成
# ============================================================
print("正在生成 Part 2 普通人体感图表...")

# ---------- 图1: "大城市 vs 小县城" 有趣对比 · 双面展示 ----------
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

# 提取均值
def mean_val(data_dict, key):
    return data_dict[key].mean()

# 小县城和一线城市的数据
small = '小县城'
big = '一线城市'

# 左边：小县城占优（小城市更好的方面）
small_advantages = [
    ('通勤时间短', f'{mean_val(commute_time, small):.0f} 分钟 vs {mean_val(commute_time, big):.0f} 分钟', '✅'),
    ('房租便宜', f'{mean_val(rent, small):.0f} 元 vs {mean_val(rent, big):.0f} 元', '✅'),
    ('停车方便', f'{mean_val(parking_search, small):.0f} 分钟 vs {mean_val(parking_search, big):.0f} 分钟', '✅'),
    ('邻里更熟', f'{mean_val(community_activity, small):.1f} 次/月 vs {mean_val(community_activity, big):.1f} 次/月', '✅'),
    ('通勤幸福', f'{mean_val(commute_time, small):.0f} 分钟 vs {mean_val(commute_time, big):.0f} 分钟', '✅'),
    ('人均公园多', f'{mean_val(park_per_capita, small):.0f} ㎡ vs {mean_val(park_per_capita, big):.0f} ㎡', '✅'),
    ('孤独感低', f'{mean_val(loneliness, small):.1f}/10 vs {mean_val(loneliness, big):.1f}/10', '✅'),
    ('猪肉便宜', f'{mean_val(pork_price, small):.0f} 元/斤 vs {mean_val(pork_price, big):.0f} 元/斤', '❌'),  # 小县城反而贵
]

# 右边：一线城市占优（大城市更好的方面）
big_advantages = [
    ('月薪更高', f'{mean_val(monthly_salary, big):.0f} 元 vs {mean_val(monthly_salary, small):.0f} 元', '✅'),
    ('外卖更快', f'{mean_val(delivery_time, big):.0f} 分钟 vs {mean_val(delivery_time, small):.0f} 分钟', '✅'),
    ('医疗更好', f'{mean_val(hospital_rank3, big):.0f} 家三甲 vs {mean_val(hospital_rank3, small):.0f} 家', '✅'),
    ('娱乐更多', f'{mean_val(entertainment, big):.1f} 场/月 vs {mean_val(entertainment, small):.1f} 场/月', '✅'),
    ('二手流通快', f'{mean_val(second_hand_speed, big):.0f} 天成交 vs {mean_val(second_hand_speed, small):.0f} 天', '✅'),
    ('包邮率高', f'{mean_val(free_shipping, big):.0f}% vs {mean_val(free_shipping, small):.0f}%', '✅'),
    ('夜生活丰富', f'{mean_val(late_night_ratio, big):.0f}% 深夜营业 vs {mean_val(late_night_ratio, small):.0f}%', '✅'),
    ('跳槽机会多', f'{mean_val(job_opportunities, big):.0f} 岗位/万人 vs {mean_val(job_opportunities, small):.0f}', '✅'),
]

# 绘制双栏布局
# 左标题
ax.text(0.01, 0.95, '小县城更胜一筹', fontsize=18, fontweight='bold', color='#3498DB', 
        transform=ax.transAxes, fontproperties=chinese_font_prop)
ax.text(0.01, 0.90, '（小城市的舒适生活）', fontsize=12, color='#85C1E9',
        transform=ax.transAxes, fontproperties=chinese_font_prop)

# 右标题
ax.text(0.51, 0.95, '一线城市更胜一筹', fontsize=18, fontweight='bold', color='#E74C3C', 
        transform=ax.transAxes, fontproperties=chinese_font_prop)
ax.text(0.51, 0.90, '（大城市的集聚红利）', fontsize=12, color='#F1948A',
        transform=ax.transAxes, fontproperties=chinese_font_prop)

# 中间分隔线
ax.axvline(x=0.50, ymin=0.05, ymax=0.88, color='gray', linewidth=1, linestyle='--', alpha=0.5)
ax.text(0.49, 0.45, 'VS', fontsize=20, fontweight='bold', color='gray', 
        transform=ax.transAxes, ha='center', fontproperties=chinese_font_prop)

# 左边条目
for i, (title, detail, _) in enumerate(small_advantages):
    y = 0.82 - i * 0.10
    ax.text(0.03, y, f'  {title}', fontsize=12, fontweight='bold', color='#2C3E50',
            transform=ax.transAxes, fontproperties=chinese_font_prop)
    ax.text(0.03, y - 0.04, f'    {detail}', fontsize=10, color='#7F8C8D',
            transform=ax.transAxes, fontproperties=chinese_font_prop)

# 右边条目
for i, (title, detail, _) in enumerate(big_advantages):
    y = 0.82 - i * 0.10
    ax.text(0.53, y, f'  {title}', fontsize=12, fontweight='bold', color='#2C3E50',
            transform=ax.transAxes, fontproperties=chinese_font_prop)
    ax.text(0.53, y - 0.04, f'    {detail}', fontsize=10, color='#7F8C8D',
            transform=ax.transAxes, fontproperties=chinese_font_prop)

# 底部总结
summary = (
    '总结: 集聚效益不是"好"或"坏"的二选一，而是用"通勤时间 + 房租"去交换"收入 + 机会 + 便利"\n'
    '每个人都在用自己的选择给集聚效益投票——没有绝对正确的答案，只看你更在乎什么。'
)
ax.text(0.50, 0.02, summary, fontsize=11, color='#2C3E50', ha='center',
        transform=ax.transAxes, fontproperties=chinese_font_prop,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEF9E7', alpha=0.8))

ax.set_title('集聚效益的"AB面": 小县城 vs 一线城市', fontsize=20, fontweight='bold',
             fontproperties=chinese_font_prop, pad=20)

plt.tight_layout()
plt.savefig(f'{output_dir}/D7_AB_side.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ D7: AB面对比")

# ---------- 图2: "集聚悖论" (人越多越孤独) ----------
fig, ax = plt.subplots(figsize=(10, 7))

# 双轴图: 人口规模 vs 活跃联系人 & 孤独感
pop_sizes = [20, 200, 800, 2500]
contact_means = [active_contacts[k].mean() for k in city_keys_flat]
lonely_means = [loneliness[k].mean() for k in city_keys_flat]
friends_means = [friends_met_weekly[k].mean() for k in city_keys_flat]

color1 = '#3498DB'
color2 = '#E74C3C'
color3 = '#2ECC71'

# 柱状图: 微信好友数
bars = ax.bar(pop_sizes, contact_means, width=150, color=color1, alpha=0.3, label='微信活跃联系人', zorder=2)
# 标注数值
for bar, val in zip(bars, contact_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.0f}人', 
            ha='center', fontsize=10, color=color1, fontweight='bold')

# 折线: 每周见面朋友数
ax2 = ax.twinx()
line1, = ax2.plot(pop_sizes, friends_means, 'o-', color=color2, linewidth=3, markersize=12, 
                  label='每周实际见面的朋友', zorder=3)
for x, y in zip(pop_sizes, friends_means):
    ax2.annotate(f'{y:.1f}人', (x, y), textcoords="offset points", xytext=(0, 12),
                ha='center', fontsize=10, color=color2, fontweight='bold')

# 折线: 孤独感
line2, = ax2.plot(pop_sizes, lonely_means, 's--', color=color3, linewidth=3, markersize=12,
                  label='孤独感评分 (1-10)', zorder=3)
for x, y in zip(pop_sizes, lonely_means):
    ax2.annotate(f'{y:.1f}', (x, y), textcoords="offset points", xytext=(0, -15),
                ha='center', fontsize=10, color=color3, fontweight='bold')

ax.set_xlabel('城市人口 (万人)', fontsize=13, fontproperties=chinese_font_prop)
ax.set_ylabel('微信活跃联系人 (人)', fontsize=13, color=color1, fontproperties=chinese_font_prop)
ax2.set_ylabel('朋友数 / 孤独感', fontsize=13, color=color2, fontproperties=chinese_font_prop)
ax.set_title('集聚悖论: 人越多，联系人越多，但朋友越少、越孤独', fontsize=15, fontproperties=chinese_font_prop)
ax.set_xticks(pop_sizes)
ax.set_xticklabels(city_keys, fontsize=10, fontproperties=chinese_font_prop)

# 图例合并
lines = [bars, line1, line2]
labels = ['微信活跃联系人', '每周实际见面的朋友', '孤独感评分 (1-10)']
ax.legend(lines, labels, loc='upper center', fontsize=11, prop=chinese_font_prop)

ax.grid(True, alpha=0.3)
ax2.set_ylim(0, 10)

plt.tight_layout()
plt.savefig(f'{output_dir}/D8_loneliness_paradox.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ D8: 孤独感悖论")

# ---------- 图3: 夜生活对比 ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

# 左: 酒吧密度
ax = axes[0]
data = [bar_density[k] for k in city_keys_flat]
bp = ax.boxplot(data, patch_artist=True, widths=0.6)
colors_box = ['#95A5A6', '#3498DB', '#F39C12', '#E74C3C']
for patch, c in zip(bp['boxes'], colors_box):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(2)
ax.set_xticklabels(city_keys, fontsize=9, fontproperties=chinese_font_prop)
ax.set_ylabel('家/平方公里', fontsize=11, fontproperties=chinese_font_prop)
ax.set_title('酒吧/夜店密度', fontsize=13, fontproperties=chinese_font_prop)
ax.grid(True, alpha=0.2)
# 标注倍数
ratio = bar_density['一线城市'].mean() / max(bar_density['小县城'].mean(), 0.01)
ax.text(0.5, 0.95, f'一线城市是\n小县城的{ratio:.0f}倍', transform=ax.transAxes, fontsize=10,
        ha='center', fontproperties=chinese_font_prop, color='#E74C3C')

# 中: 深夜营业比例
ax = axes[1]
data = [late_night_ratio[k] for k in city_keys_flat]
bp = ax.boxplot(data, patch_artist=True, widths=0.6)
for patch, c in zip(bp['boxes'], colors_box):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(2)
ax.set_xticklabels(city_keys, fontsize=9, fontproperties=chinese_font_prop)
ax.set_ylabel('%', fontsize=11, fontproperties=chinese_font_prop)
ax.set_title('深夜营业 (0点后) 商家比例', fontsize=13, fontproperties=chinese_font_prop)
ax.grid(True, alpha=0.2)
ax.text(0.5, 0.95, f'一线城市每3家\n就有1家深夜营业', transform=ax.transAxes, fontsize=10,
        ha='center', fontproperties=chinese_font_prop, color='#E74C3C')

# 右: 24h便利店密度
ax = axes[2]
data = [night_services[k] for k in city_keys_flat]
bp = ax.boxplot(data, patch_artist=True, widths=0.6)
for patch, c in zip(bp['boxes'], colors_box):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(2)
ax.set_xticklabels(city_keys, fontsize=9, fontproperties=chinese_font_prop)
ax.set_ylabel('家/平方公里', fontsize=11, fontproperties=chinese_font_prop)
ax.set_title('24小时便利店密度', fontsize=13, fontproperties=chinese_font_prop)
ax.grid(True, alpha=0.2)
ratio2 = night_services['一线城市'].mean() / max(night_services['小县城'].mean(), 0.01)
ax.text(0.5, 0.95, f'一线城市是\n小县城的{ratio2:.0f}倍', transform=ax.transAxes, fontsize=10,
        ha='center', fontproperties=chinese_font_prop, color='#E74C3C')

fig.suptitle('夜生活集聚效应: 城市越大，夜越精彩', fontsize=16, fontproperties=chinese_font_prop)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'{output_dir}/D9_nightlife.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ D9: 夜生活对比")

# ---------- 图4: 医疗教育"焦虑指数" ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左: 医疗资源对比
ax = axes[0]
x = np.arange(len(city_keys))
width = 0.2

# 三甲医院数（需要缩放显示）
hosp_norm = [hospital_rank3[k].mean() / 40 * 100 for k in city_keys_flat]
wait_norm = [100 - specialist_wait[k].mean() / 20 * 100 for k in city_keys_flat]  # 等待时间越短越好
amb_norm = [100 - ambulance_time[k].mean() / 25 * 100 for k in city_keys_flat]

ax.bar(x - width, hosp_norm, width, label='三甲医院资源', color='#E74C3C', alpha=0.8)
ax.bar(x, wait_norm, width, label='挂专家号便利度', color='#3498DB', alpha=0.8)
ax.bar(x + width, amb_norm, width, label='120急救响应', color='#2ECC71', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(city_keys, fontsize=9, fontproperties=chinese_font_prop)
ax.set_ylabel('得分 (0-100, 越高越好)', fontsize=11, fontproperties=chinese_font_prop)
ax.set_title('医疗资源集聚: 大城市看病更方便?', fontsize=14, fontproperties=chinese_font_prop)
ax.legend(fontsize=10, loc='lower right', prop=chinese_font_prop)
ax.grid(True, alpha=0.2, axis='y')

# 右: 教育焦虑指数
ax = axes[1]
# 学区房溢价
premium_norm = [school_district_premium[k].mean() / 12000 * 100 for k in city_keys_flat]
# 补习班密度
tutor_norm = [tutoring_density[k].mean() / 20 * 100 for k in city_keys_flat]
# 大学资源
uni_norm = [top_universities[k].mean() / 8 * 100 for k in city_keys_flat]

ax.bar(x - width, premium_norm, width, label='学区房溢价压力', color='#E74C3C', alpha=0.8)
ax.bar(x, tutor_norm, width, label='补习班密度', color='#F39C12', alpha=0.8)
ax.bar(x + width, uni_norm, width, label='高等教育资源', color='#9B59B6', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(city_keys, fontsize=9, fontproperties=chinese_font_prop)
ax.set_ylabel('得分 (0-100)', fontsize=11, fontproperties=chinese_font_prop)
ax.set_title('教育"焦虑": 资源好但竞争也激烈', fontsize=14, fontproperties=chinese_font_prop)
ax.legend(fontsize=10, loc='upper left', prop=chinese_font_prop)
ax.grid(True, alpha=0.2, axis='y')

# 添加注释
ax.annotate('学区房溢价\n一线城市达1.2万/㎡\n是县城的24倍', xy=(2.8, 80), fontsize=9,
            fontproperties=chinese_font_prop, color='#E74C3C',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

fig.suptitle('医疗与教育: 集聚带来的"资源红利"和"竞争焦虑"', fontsize=15, fontproperties=chinese_font_prop)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'{output_dir}/D10_medical_education.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ D10: 医疗教育")

# ---------- 图5: "菜市场经济学" - 集聚带来的实惠 ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

# 左: 猪肉价格
ax = axes[0]
data = [pork_price[k] for k in city_keys_flat]
bp = ax.boxplot(data, patch_artist=True, widths=0.6)
for patch, c in zip(bp['boxes'], colors_box):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(2)
ax.set_xticklabels(city_keys, fontsize=9, fontproperties=chinese_font_prop)
ax.set_ylabel('元/斤', fontsize=11, fontproperties=chinese_font_prop)
ax.set_title('猪肉价格', fontsize=13, fontproperties=chinese_font_prop)
ax.grid(True, alpha=0.2)
# 趋势线
means = [pork_price[k].mean() for k in city_keys_flat]
ax.plot(range(1, 5), means, 'r--', linewidth=2, marker='o', markersize=8, color='#E74C3C', zorder=3)
ax.annotate('大城市反而便宜\n(物流集聚效应)', xy=(4, means[3]), fontsize=10,
            fontproperties=chinese_font_prop, color='#E74C3C',
            xytext=(3.5, means[3] + 2), arrowprops=dict(arrowstyle='->', color='#E74C3C'))

# 中: 蔬菜种类
ax = axes[1]
data = [vegetable_variety[k] for k in city_keys_flat]
bp = ax.boxplot(data, patch_artist=True, widths=0.6)
for patch, c in zip(bp['boxes'], colors_box):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(2)
ax.set_xticklabels(city_keys, fontsize=9, fontproperties=chinese_font_prop)
ax.set_ylabel('种', fontsize=11, fontproperties=chinese_font_prop)
ax.set_title('菜市场蔬菜种类', fontsize=13, fontproperties=chinese_font_prop)
ax.grid(True, alpha=0.2)
ax.annotate('种类多3倍+\n什么菜都能买到', xy=(4, 100), fontsize=10,
            fontproperties=chinese_font_prop, color='#2ECC71')

# 右: 生鲜电商覆盖率
ax = axes[2]
data = [fresh_delivery[k] for k in city_keys_flat]
bp = ax.boxplot(data, patch_artist=True, widths=0.6)
for patch, c in zip(bp['boxes'], colors_box):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(2)
ax.set_xticklabels(city_keys, fontsize=9, fontproperties=chinese_font_prop)
ax.set_ylabel('%', fontsize=11, fontproperties=chinese_font_prop)
ax.set_title('生鲜电商次日达覆盖率', fontsize=13, fontproperties=chinese_font_prop)
ax.grid(True, alpha=0.2)
ax.annotate('几乎100%覆盖\n小县城仅15%', xy=(4, 98), fontsize=10,
            fontproperties=chinese_font_prop, color='#2ECC71')

fig.suptitle('"菜市场经济学": 集聚让大城市生活更便宜、更方便?', fontsize=15, fontproperties=chinese_font_prop)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'{output_dir}/D11_food_market.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ D11: 菜市场经济学")

# ---------- 图6: 综合"人生选择计分卡" ----------
fig, ax = plt.subplots(figsize=(14, 9))
ax.axis('off')

# 构建一个综合计分卡表格
# 维度: 每个维度给出小县城和一线城市的得分

categories_score = [
    ('维度', '权重', '小县城', '得分', '一线城市', '得分'),
    ('月薪收入', '15%', f'{monthly_salary["小县城"].mean():.0f}元', '★★☆☆☆', f'{monthly_salary["一线城市"].mean():.0f}元', '★★★★★'),
    ('房租成本', '15%', f'{rent["小县城"].mean():.0f}元', '★★★★★', f'{rent["一线城市"].mean():.0f}元', '★☆☆☆☆'),
    ('通勤体验', '10%', f'{commute_time["小县城"].mean():.0f}分钟', '★★★★★', f'{commute_time["一线城市"].mean():.0f}分钟', '★★☆☆☆'),
    ('医疗资源', '10%', f'{hospital_rank3["小县城"].mean():.0f}家三甲', '★☆☆☆☆', f'{hospital_rank3["一线城市"].mean():.0f}家三甲', '★★★★★'),
    ('教育机会', '10%', f'{top_universities["小县城"].mean():.0f}所985/211', '★☆☆☆☆', f'{top_universities["一线城市"].mean():.0f}所985/211', '★★★★★'),
    ('生活便利', '10%', '基础满足', '★★☆☆☆', '应有尽有', '★★★★★'),
    ('夜生活/娱乐', '8%', f'{entertainment["小县城"].mean():.1f}场/月', '★☆☆☆☆', f'{entertainment["一线城市"].mean():.1f}场/月', '★★★★★'),
    ('社交圈子', '8%', '熟人社会', '★★★★☆', f'{friends_met_weekly["一线城市"].mean():.1f}人/周', '★★☆☆☆'),
    ('物价水平', '7%', f'{pork_price["小县城"].mean():.0f}元/斤猪肉', '★★★☆☆', f'{pork_price["一线城市"].mean():.0f}元/斤猪肉', '★★★★☆'),
    ('快递/网购', '7%', f'{delivery_days["小县城"].mean():.1f}天到', '★★☆☆☆', f'{delivery_days["一线城市"].mean():.1f}天到', '★★★★★'),
    ('二手交易', '5%', f'{second_hand_speed["小县城"].mean():.0f}天成交', '★☆☆☆☆', f'{second_hand_speed["一线城市"].mean():.0f}天成交', '★★★★★'),
    ('宠物友好', '5%', f'{pet_friendly["小县城"].mean():.0f}个场所', '★☆☆☆☆', f'{pet_friendly["一线城市"].mean():.0f}个场所', '★★★★★'),
]

# 加权总分
weights = [15, 15, 10, 10, 10, 10, 8, 8, 7, 7, 5, 5]
# 小县城得分（满分5分）
small_scores = [2, 5, 5, 1, 1, 2, 1, 4, 3, 2, 1, 1]
big_scores = [5, 1, 2, 5, 5, 5, 5, 2, 4, 5, 5, 5]
small_total = sum(s * w for s, w in zip(small_scores, weights)) / 100
big_total = sum(s * w for s, w in zip(big_scores, weights)) / 100

# 创建表格
table = ax.table(cellText=categories_score[1:], colLabels=categories_score[0],
                 cellLoc='center', loc='center',
                 colWidths=[0.15, 0.06, 0.15, 0.12, 0.15, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(10)

# 表头样式
for j in range(6):
    cell = table[0, j]
    cell.set_facecolor('#2C3E50')
    cell.set_text_props(color='white', fontweight='bold', fontproperties=chinese_font_prop)

# 数据行样式
for i in range(1, len(categories_score)):
    for j in range(6):
        cell = table[i, j]
        if i % 2 == 0:
            cell.set_facecolor('#F2F3F4')
        else:
            cell.set_facecolor('white')
        cell.set_text_props(fontproperties=chinese_font_prop)
        # 得分列着色
        if j == 3:  # 小县城得分
            text = cell.get_text().get_text()
            stars = text.count('★')
            if stars <= 2:
                cell.set_facecolor('#FADBD8')
            elif stars <= 3:
                cell.set_facecolor('#FEF9E7')
            else:
                cell.set_facecolor('#D5F5E3')
        if j == 5:  # 一线城市得分
            text = cell.get_text().get_text()
            stars = text.count('★')
            if stars <= 2:
                cell.set_facecolor('#FADBD8')
            elif stars <= 3:
                cell.set_facecolor('#FEF9E7')
            else:
                cell.set_facecolor('#D5F5E3')

table.scale(1, 1.5)

# 总分显示
ax.text(0.15, 0.06, f'加权总分: {small_total:.2f} / 5.00', fontsize=14, fontweight='bold',
        color='#3498DB', transform=ax.transAxes, fontproperties=chinese_font_prop)
ax.text(0.60, 0.06, f'加权总分: {big_total:.2f} / 5.00', fontsize=14, fontweight='bold',
        color='#E74C3C', transform=ax.transAxes, fontproperties=chinese_font_prop)

# 结论
if small_total > big_total:
    conclusion = '结论: 在这个权重体系下，小县城综合得分更高 (更适合追求安逸、低成本的生活)'
else:
    conclusion = '结论: 在这个权重体系下，一线城市综合得分更高 (更适合追求机会、丰富性的人生)'

# 提示用户可以调整权重
ax.text(0.50, 0.01, f'{conclusion}', fontsize=12, fontweight='bold', ha='center',
        transform=ax.transAxes, fontproperties=chinese_font_prop,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEF9E7', alpha=0.9))

ax.set_title('"人生选择计分卡": 按你的价值观打分', fontsize=18, fontweight='bold',
             fontproperties=chinese_font_prop, pad=20)

plt.tight_layout()
plt.savefig(f'{output_dir}/D12_scorecard.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ D12: 人生选择计分卡")

# 输出汇总
print(f"\n===== Part 2 图表清单 =====")
print(f"  D7_AB_side.png              - 集聚效益AB面")
print(f"  D8_loneliness_paradox.png   - 孤独感悖论")
print(f"  D9_nightlife.png            - 夜生活对比")
print(f"  D10_medical_education.png   - 医疗教育资源")
print(f"  D11_food_market.png         - 菜市场经济学")
print(f"  D12_scorecard.png           - 人生选择计分卡")