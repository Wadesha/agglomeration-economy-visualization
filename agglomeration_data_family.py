"""
=============================================================================
集聚效益数据家族 (Agglomeration Economy Data Family)
=============================================================================

本模块从底层理论出发，构建完整的集聚效益数据生成体系。

┌─────────────────────────────────────────────────────────────────────┐
│                    集聚效益知识框架 (Theory Stack)                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  第1层: 核心概念                                                    │
│  ├── 集聚效益 (Agglomeration Economies)                            │
│  │    = 经济活动在空间集中产生的额外效益                             │
│  │    = 个体之和 < 整体之和的"1+1>2"效应                             │
│  └── 理论基础: Marshall (1890) → Hoover (1937) → Jacobs (1969)      │
│                                                                     │
│  第2层: 三种集聚类型                                                │
│  ├── 城市化经济 (Urbanization Economies)                            │
│  │    = 城市整体规模带来的效益 (跨行业)                             │
│  │    例: 大城市有更好的机场、地铁、医院                             │
│  ├── 地方化经济 (Localization Economies)                            │
│  │    = 同一行业集聚带来的效益 (行业内)                             │
│  │    例: 硅谷的IT集群、义乌的小商品集群                             │
│  └── 内部规模经济 (Internal Scale Economies)                        │
│       = 企业自身规模扩大带来的效益                                   │
│       例: 工厂产能翻倍, 单位成本下降                                │
│                                                                     │
│  第3层: 三大集聚机制 (Marshallian Trinity)                          │
│  ├── 共享 (Sharing)                                                 │
│  │    ├── 共享基础设施 (道路、港口、电网)                           │
│  │    ├── 共享供应商 (专业化分工)                                   │
│  │    └── 共享劳动力池 (降低招聘/求职成本)                         │
│  ├── 匹配 (Matching)                                                │
│  │    ├── 更好的劳-岗匹配 (技能对口)                               │
│  │    ├── 更快的匹配速度 (职位空缺填充快)                          │
│  │    └── 更精准的上下游匹配 (供应链效率)                          │
│  └── 学习 (Learning)                                                │
│       ├── 知识溢出 (Knowledge Spillovers)                           │
│       ├── 干中学 (Learning by Doing)                                │
│       └── 创新加速 (Innovation Acceleration)                        │
│                                                                     │
│  第4层: 扩展机制 (Extended Mechanisms)                              │
│  ├── 本地市场效应 (Home Market Effect)                              │
│  │    = 需求大的地方吸引更多企业 → 更多选择 → 更低价格             │
│  ├── 网络效应 (Network Effects)                                     │
│  │    = 每个新加入者都让网络对其他人更有价值                        │
│  ├── 竞争效应 (Competition Effects)                                 │
│  │    = 企业集聚加剧竞争 → 效率提升、价格下降                       │
│  └── 选择效应 (Selection Effects)                                   │
│       = 大城市的激烈竞争筛选出效率更高的企业                        │
│                                                                     │
│  第5层: 度量指标 (Measurement Metrics)                              │
│  ├── 宏观: 城市标度律 β、人口密度、地均GDP                          │
│  ├── 中观: 区位商 LQ、EG指数、HHI、空间基尼系数                     │
│  ├── 微观: 全要素生产率 TFP、工资溢价、企业进入退出率               │
│  └── 感知: 生活便利指数、通勤成本、社交网络密度                     │
│                                                                     │
│  第6层: 数据家族 (Data Family)                                      │
│  ├── Tier 1: 跨城市面板数据 (200城市 × 20年 × 30指标)              │
│  ├── Tier 2: 产业集聚数据 (200城市 × 20产业 × 15指标)              │
│  ├── Tier 3: 企业微观数据 (5000企业 × 30指标)                      │
│  ├── Tier 4: 个体感知数据 (10000居民 × 20指标)                     │
│  └── Tier 5: 城际联系数据 (200城市 × 200城市 × 5指标)              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

经典文献:
  - Marshall (1890) Principles of Economics
  - Hoover (1937) Location Theory and the Shoe and Leather Industries
  - Jacobs (1969) The Economy of Cities
  - Glaeser et al. (1992) Growth in Cities, JPE
  - Ellison & Glaeser (1997) Geographic Concentration in U.S. Industries, JPE
  - Rosenthal & Strange (2004) Evidence on the Nature and Sources of
    Agglomeration Economies, Handbook of Urban and Regional Economics
  - Bettencourt et al. (2007) Growth, Innovation, Scaling, and the Pace
    of Life in Cities, PNAS
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# 第0层: 基础参数与配置
# ============================================================================

@dataclass
class AgglomerationConfig:
    """集聚效益数据家族配置"""
    # 城市层级
    n_cities: int = 200
    n_years: int = 20
    n_industries: int = 20
    n_firms: int = 5000
    n_individuals: int = 10000
    
    # 城市规模参数 (对数正态分布)
    pop_min: float = 3.0    # ln(20万)
    pop_max: float = 7.5    # ln(2000万)
    pop_mean: float = 5.0
    pop_std: float = 1.0
    
    # 标度律参数
    beta_gdp: float = 1.102        # GDP 标度指数
    beta_patent: float = 1.25      # 创新标度指数 (更高)
    beta_wage: float = 1.05        # 工资标度指数 (较低)
    beta_infrastructure: float = 0.85  # 基础设施标度指数 (规模经济)
    
    # 产业结构参数
    n_high_tech: int = 5           # 高科技产业数
    n_traditional: int = 10        # 传统产业数
    n_service: int = 5            # 服务业数
    
    # 随机种子
    seed: int = 42


config = AgglomerationConfig()


# ============================================================================
# 第1层: 核心工具函数
# ============================================================================

def lognormal_population(n: int, mean: float = 5.0, std: float = 1.0,
                         min_val: float = 3.0, max_val: float = 7.5) -> np.ndarray:
    """生成对数正态分布的城市人口 (万人)"""
    raw = np.random.lognormal(mean=mean, sigma=std, size=n)
    raw = np.clip(raw, np.exp(min_val), np.exp(max_val))
    return raw


def urban_scaling(population: np.ndarray, beta: float,
                  base: float = 1.0, noise: float = 0.15) -> np.ndarray:
    """
    城市标度律: Y = Y0 * N^β * exp(ε)
    β > 1: 超线性 (集聚效益)
    β = 1: 线性
    β < 1: 次线性 (规模经济)
    """
    eps = np.random.normal(0, noise, size=len(population))
    return base * (population ** beta) * np.exp(eps)


def location_quotient(industry_share_local: np.ndarray,
                      industry_share_national: np.ndarray) -> np.ndarray:
    """
    区位商 (Location Quotient)
    LQ_{ij} = (E_{ij}/E_i) / (E_j/E)
    LQ > 1: 该地区该产业有比较优势 (专业化)
    LQ < 1: 该地区该产业不占优势
    """
    return industry_share_local / industry_share_national


def herfindahl_index(shares: np.ndarray) -> float:
    """赫芬达尔指数 (HHI): 衡量产业集中度"""
    return np.sum(shares ** 2)


def ellison_glaeser_index(s_i: np.ndarray, x_i: np.ndarray) -> float:
    """
    Ellison-Glaeser 集聚指数
    γ = [G - (1 - Σx_i²)] / [(1 - Σx_i²)(1 - H)]
    其中 G = Σ(s_i - x_i)², H = Σz_k² (企业集中度)
    """
    # 修正: 返回简化的空间基尼
    G = np.sum((s_i - x_i) ** 2)
    H = np.sum((s_i / np.sum(s_i)) ** 2) if np.sum(s_i) > 0 else 0
    denominator = (1 - np.sum(x_i ** 2)) * (1 - H)
    if denominator <= 0:
        return 0
    return (G - (1 - np.sum(x_i ** 2))) / denominator


# ============================================================================
# 第2层: 数据家族生成器
# ============================================================================

class AgglomerationDataFamily:
    """
    集聚效益数据家族生成器
    
    生成5个层级的数据集，构成完整的"数据家族":
    Tier 1: 跨城市面板数据 (宏观)
    Tier 2: 产业集聚数据 (中观)
    Tier 3: 企业微观数据 (微观)
    Tier 4: 个体感知数据 (体验)
    Tier 5: 城际联系数据 (网络)
    """

    def __init__(self, config: AgglomerationConfig = config):
        self.config = config
        self.family = {}  # 存储所有生成的家族数据
        
    # ---- Tier 1: 跨城市面板数据 (宏观) ----
    
    def generate_tier1_city_panel(self) -> pd.DataFrame:
        """
        Tier 1: 跨城市面板数据 (200城市 × 20年)
        
        核心指标:
        - 人口、GDP、人均GDP、地均GDP
        - 就业率、工资水平、专利数
        - 企业数量、企业进入率、企业退出率
        - 基础设施、教育投入、医疗资源
        """
        n = self.config.n_cities
        years = range(2020, 2020 + self.config.n_years)
        
        records = []
        
        # 基础城市特征 (固定)
        city_id = np.arange(n)
        city_name = [f"城市_{i:03d}" for i in range(n)]
        
        # 城市层级分类
        base_pop = lognormal_population(n)
        city_tier = np.zeros(n, dtype=int)
        city_tier[base_pop < np.percentile(base_pop, 25)] = 0  # 小城市
        city_tier[(base_pop >= np.percentile(base_pop, 25)) & 
                  (base_pop < np.percentile(base_pop, 50))] = 1  # 中等城市
        city_tier[(base_pop >= np.percentile(base_pop, 50)) & 
                  (base_pop < np.percentile(base_pop, 75))] = 2  # 大城市
        city_tier[base_pop >= np.percentile(base_pop, 75)] = 3  # 特大城市
        
        tier_names = ['小城市', '中等城市', '大城市', '特大城市']
        
        # 城市基础特征 (固定)
        area = np.random.lognormal(mean=5.0, sigma=0.8, size=n)  # 面积 (km²)
        area = np.clip(area, 50, 50000)
        
        is_coastal = np.random.binomial(1, 0.3, size=n)  # 沿海城市
        is_capital = np.random.binomial(1, 0.05, size=n)  # 省会
        has_port = np.random.binomial(1, 0.15, size=n)  # 港口城市
        
        for t, year in enumerate(years):
            # 人口逐年增长 (1-3%/年)
            growth_rate = np.random.normal(0.02, 0.01, size=n)
            pop = base_pop * (1 + growth_rate) ** t
            pop = np.clip(pop, 5, 5000)
            
            # 城市标度律: GDP ~ N^β  (β=1.102)
            gdp = urban_scaling(pop, beta=self.config.beta_gdp, 
                                base=0.5, noise=0.15)
            
            # 人均GDP
            gdp_per_capita = gdp / pop * 10000  # 万元/人
            
            # 地均GDP
            gdp_per_km2 = gdp / area * 100  # 亿元/千km²
            
            # 工资水平 (标度律, β=1.05)
            wage = urban_scaling(pop, beta=self.config.beta_wage,
                                 base=0.3, noise=0.08) * 1000
            
            # 就业率 (大城市略高, 但有上限)
            employment_rate = 0.90 + 0.05 * (pop / pop.max()) 
            employment_rate = np.clip(employment_rate, 0.80, 0.97)
            employment_rate += np.random.normal(0, 0.02, size=n)
            employment_rate = np.clip(employment_rate, 0.75, 0.98)
            
            # 专利数 (创新标度律, β=1.25, 集聚效应更强)
            patents = urban_scaling(pop, beta=self.config.beta_patent,
                                    base=0.01, noise=0.3).astype(int)
            patents = np.clip(patents, 0, 50000)
            
            # 企业数量
            firms = (pop * np.random.lognormal(mean=-1.0, sigma=0.3, size=n) * 100).astype(int)
            firms = np.clip(firms, 100, 500000)
            
            # 企业进入率 (大城市更高)
            firm_entry_rate = 0.10 + 0.05 * (pop / pop.max()) + np.random.normal(0, 0.02, size=n)
            firm_entry_rate = np.clip(firm_entry_rate, 0.05, 0.25)
            
            # 企业退出率 (大城市竞争激烈, 退出率也高)
            firm_exit_rate = 0.05 + 0.05 * (pop / pop.max()) + np.random.normal(0, 0.02, size=n)
            firm_exit_rate = np.clip(firm_exit_rate, 0.03, 0.20)
            
            # 基础设施指数 (集聚可共享, 标度指数 < 1)
            infrastructure = urban_scaling(pop, beta=self.config.beta_infrastructure,
                                            base=0.2, noise=0.1) * 100
            infrastructure = np.clip(infrastructure, 10, 100)
            
            # 教育投入 (占GDP比例)
            edu_ratio = 0.04 + 0.02 * (1 - pop / pop.max()) + np.random.normal(0, 0.005, size=n)
            edu_ratio = np.clip(edu_ratio, 0.02, 0.08)
            
            # 医疗资源 (每千人医生数)
            doctors_per_1000 = 1.0 + 3.0 * (pop / pop.max()) + np.random.normal(0, 0.5, size=n)
            doctors_per_1000 = np.clip(doctors_per_1000, 0.5, 8.0)
            
            # 居民人均可支配收入
            income = wage * 0.6 * (1 + np.random.normal(0, 0.05, size=n))
            
            for i in range(n):
                records.append({
                    'city_id': city_id[i],
                    'city_name': city_name[i],
                    'year': year,
                    'tier': city_tier[i],
                    'tier_name': tier_names[city_tier[i]],
                    'population_10k': round(pop[i], 2),
                    'area_km2': round(area[i], 1),
                    'is_coastal': is_coastal[i],
                    'is_capital': is_capital[i],
                    'has_port': has_port[i],
                    'gdp_billion': round(gdp[i], 2),
                    'gdp_per_capita_10k': round(gdp_per_capita[i], 2),
                    'gdp_per_km2': round(gdp_per_km2[i], 2),
                    'avg_wage': round(wage[i], 1),
                    'employment_rate': round(employment_rate[i], 3),
                    'patents': int(patents[i]),
                    'n_firms': int(firms[i]),
                    'firm_entry_rate': round(firm_entry_rate[i], 3),
                    'firm_exit_rate': round(firm_exit_rate[i], 3),
                    'infrastructure_index': round(infrastructure[i], 1),
                    'edu_expenditure_ratio': round(edu_ratio[i], 3),
                    'doctors_per_1000': round(doctors_per_1000[i], 2),
                    'disposable_income': round(income[i], 1),
                })
        
        df = pd.DataFrame(records)
        self.family['tier1_city_panel'] = df
        return df
    
    # ---- Tier 2: 产业集聚数据 (中观) ----
    
    def generate_tier2_industry_cluster(self) -> pd.DataFrame:
        """
        Tier 2: 产业集聚数据 (200城市 × 20产业)
        
        核心指标:
        - 各产业就业人数、产值
        - 区位商 (LQ)
        - 产业集中度 (HHI、EG指数)
        - 产业链上下游关联度
        """
        n_cities = self.config.n_cities
        n_industries = self.config.n_industries
        n_ht = self.config.n_high_tech
        n_trad = self.config.n_traditional
        
        city_pop = self.family.get('tier1_city_panel', pd.DataFrame())
        if not city_pop.empty:
            base_pop = city_pop[city_pop['year'] == 2020]['population_10k'].values
        else:
            base_pop = lognormal_population(n_cities)
        
        # 产业分类
        industry_names = (
            [f"高科技_{i+1}" for i in range(n_ht)] +
            [f"传统制造_{i+1}" for i in range(n_trad)] +
            [f"服务产业_{i+1}" for i in range(self.config.n_service)]
        )
        industry_types = (
            ['high_tech'] * n_ht +
            ['traditional'] * n_trad +
            ['service'] * self.config.n_service
        )
        
        # 全国产业份额 (基准)
        national_share = np.random.dirichlet(np.ones(n_industries) * 2)
        
        records = []
        pop_norm = base_pop / base_pop.max()
        
        for i in range(n_cities):
            # 大城市偏向高科技+服务业, 小城市偏向传统制造
            ht_bias = 0.5 + 0.5 * pop_norm[i]
            serv_bias = 0.4 + 0.6 * pop_norm[i]
            trad_bias = 1.0 - 0.3 * pop_norm[i]
            
            # 生成本地产业份额
            raw_weights = np.ones(n_industries)
            for j in range(n_industries):
                if industry_types[j] == 'high_tech':
                    raw_weights[j] = ht_bias * (1 + np.random.randn() * 0.3)
                elif industry_types[j] == 'traditional':
                    raw_weights[j] = trad_bias * (1 + np.random.randn() * 0.2)
                else:
                    raw_weights[j] = serv_bias * (1 + np.random.randn() * 0.25)
            
            raw_weights = np.maximum(raw_weights, 0.01)
            local_share = raw_weights / raw_weights.sum()
            
            # 总就业人数
            total_employment = int(base_pop[i] * np.random.uniform(3000, 8000))
            
            for j in range(n_industries):
                emp_share = local_share[j]
                employment = int(total_employment * emp_share)
                output = employment * np.random.lognormal(mean=2.0, sigma=0.5)
                
                # 区位商
                lq = location_quotient(local_share[j], national_share[j])
                
                # 企业数量 (每个产业)
                n_firms_ind = max(1, int(employment / np.random.uniform(20, 200)))
                
                # 平均工资 (高科技 > 服务 > 传统)
                if industry_types[j] == 'high_tech':
                    wage_mult = np.random.uniform(1.2, 1.8)
                elif industry_types[j] == 'service':
                    wage_mult = np.random.uniform(0.8, 1.2)
                else:
                    wage_mult = np.random.uniform(0.6, 1.0)
                
                avg_wage = 5000 * wage_mult * (1 + 0.3 * pop_norm[i])
                
                records.append({
                    'city_id': i,
                    'industry_id': j,
                    'industry_name': industry_names[j],
                    'industry_type': industry_types[j],
                    'employment': employment,
                    'output_billion': round(output / 1e8, 4),
                    'n_firms': n_firms_ind,
                    'avg_wage': round(avg_wage, 1),
                    'location_quotient': round(lq, 3),
                    'national_share': round(national_share[j], 4),
                    'local_share': round(emp_share, 4),
                })
        
        df = pd.DataFrame(records)
        
        # 计算城市层面的产业集中度
        city_lq = df.groupby('city_id').apply(
            lambda g: {'hhi': herfindahl_index(g['local_share'].values)}
        )
        hhi_vals = city_lq.apply(lambda x: x['hhi']).values
        city_hhi = pd.DataFrame({
            'city_id': range(n_cities),
            'industry_hhi': hhi_vals
        })
        df = df.merge(city_hhi, on='city_id')
        
        self.family['tier2_industry_cluster'] = df
        return df
    
    # ---- Tier 3: 企业微观数据 (微观) ----
    
    def generate_tier3_firm_data(self) -> pd.DataFrame:
        """
        Tier 3: 企业微观数据 (5000家企业)
        
        核心指标:
        - 企业规模、产值、利润
        - 全要素生产率 (TFP)
        - 集聚溢出效应 (同行业其他企业数量的影响)
        - 供应链距离
        """
        n_firms = self.config.n_firms
        n_cities = self.config.n_cities
        n_industries = self.config.n_industries
        
        # 获取城市基础数据
        city_panel = self.family.get('tier1_city_panel', pd.DataFrame())
        if not city_panel.empty:
            base_pop = city_panel[city_panel['year'] == 2020]['population_10k'].values
        else:
            base_pop = lognormal_population(n_cities)
        
        pop_norm = base_pop / base_pop.max()
        
        # 分配企业到城市 (按人口规模加权)
        city_weights = base_pop ** 1.2  # 大城市吸引更多企业
        city_probs = city_weights / city_weights.sum()
        firm_cities = np.random.choice(n_cities, size=n_firms, p=city_probs)
        
        # 分配企业到产业
        industry_probs = np.ones(n_industries) / n_industries
        firm_industries = np.random.choice(n_industries, size=n_firms, p=industry_probs)
        
        records = []
        
        for i in range(n_firms):
            city = firm_cities[i]
            industry = firm_industries[i]
            city_pop = base_pop[city]
            pop_norm_city = pop_norm[city]
            
            # 企业规模 (对数正态, 大部分是中小企业)
            size_class = np.random.choice(['micro', 'small', 'medium', 'large'],
                                          p=[0.3, 0.4, 0.2, 0.1])
            size_mult = {'micro': 0.5, 'small': 1.0, 'medium': 3.0, 'large': 10.0}
            
            employees = int(np.random.lognormal(mean=3.0, sigma=1.0) * size_mult[size_class])
            employees = max(1, employees)
            
            # 集聚溢出: 同城市同行业企业越多, TFP越高
            same_city_industry = np.sum((firm_cities == city) & (firm_industries == industry))
            agglomeration_spillover = np.log(same_city_industry + 2) / np.log(n_firms)  # 归一化
            
            # 全要素生产率 (TFP) - 受集聚溢出、城市规模、企业自身影响
            tfp_base = np.random.lognormal(mean=0.0, sigma=0.3)
            tfp = tfp_base * (1 + 0.2 * agglomeration_spillover) * (1 + 0.1 * pop_norm_city)
            
            # 产值
            revenue = employees * np.random.lognormal(mean=4.0, sigma=0.5) * tfp
            
            # 利润
            profit_margin = np.random.beta(2, 5) * (1 + 0.1 * agglomeration_spillover)
            profit = revenue * profit_margin
            
            # 出口比例 (大城市企业更可能出口)
            export_ratio = 0.1 * pop_norm_city + np.random.uniform(0, 0.2)
            export_ratio = np.clip(export_ratio, 0, 0.8)
            
            # 创新投入 (R&D占收入比例)
            rd_ratio = 0.01 + 0.04 * (industry < self.config.n_high_tech) + \
                       0.02 * pop_norm_city + np.random.uniform(0, 0.02)
            rd_ratio = np.clip(rd_ratio, 0, 0.2)
            
            # 供应链距离指数 (大城市更短)
            supply_chain_distance = np.random.exponential(scale=50 * (1 - 0.3 * pop_norm_city))
            
            # 成立年限
            age = int(np.random.exponential(scale=10) + 1)
            
            records.append({
                'firm_id': i,
                'city_id': city,
                'industry_id': industry,
                'size_class': size_class,
                'employees': employees,
                'revenue_million': round(revenue / 1e6, 2),
                'profit_million': round(profit / 1e6, 2),
                'profit_margin': round(profit_margin, 3),
                'tfp': round(tfp, 3),
                'agglomeration_spillover': round(agglomeration_spillover, 3),
                'export_ratio': round(export_ratio, 3),
                'rd_ratio': round(rd_ratio, 3),
                'supply_chain_distance': round(supply_chain_distance, 1),
                'age': age,
                'city_population': round(city_pop, 2),
            })
        
        df = pd.DataFrame(records)
        self.family['tier3_firm_data'] = df
        return df
    
    # ---- Tier 4: 个体感知数据 (体验) ----
    
    def generate_tier4_individual_survey(self) -> pd.DataFrame:
        """
        Tier 4: 个体感知数据 (10000居民)
        
        核心指标:
        - 收入、工作满意度
        - 通勤时间、通勤满意度
        - 社交网络、孤独感
        - 住房满意度、生活满意度
        - 主观幸福感
        """
        n_ind = self.config.n_individuals
        n_cities = self.config.n_cities
        
        # 获取城市数据
        city_panel = self.family.get('tier1_city_panel', pd.DataFrame())
        if not city_panel.empty:
            base_pop = city_panel[city_panel['year'] == 2020]['population_10k'].values
        else:
            base_pop = lognormal_population(n_cities)
        
        pop_norm = base_pop / base_pop.max()
        
        # 分配个体到城市 (按人口加权)
        city_weights = base_pop ** 1.0
        city_probs = city_weights / city_weights.sum()
        ind_cities = np.random.choice(n_cities, size=n_ind, p=city_probs)
        
        # 年龄分布
        ages = np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+'],
                                 size=n_ind, p=[0.15, 0.30, 0.25, 0.20, 0.10])
        age_map = {'18-25': 22, '26-35': 30, '36-45': 40, '46-55': 50, '55+': 60}
        
        # 教育水平 (大城市更高)
        edu_levels = ['高中及以下', '大专', '本科', '硕士', '博士']
        
        records = []
        
        for i in range(n_ind):
            city = ind_cities[i]
            city_pop = base_pop[city]
            pop_norm_city = pop_norm[city]
            age = ages[i]
            age_val = age_map[age]
            
            # 教育水平 (大城市居民教育水平更高)
            edu_probs = np.array([0.3, 0.2, 0.3, 0.15, 0.05])
            edu_probs = edu_probs * (1 - 0.3 * pop_norm_city) + \
                        np.array([0.05, 0.1, 0.3, 0.35, 0.2]) * pop_norm_city
            edu_probs = edu_probs / edu_probs.sum()
            edu = np.random.choice(edu_levels, p=edu_probs)
            
            # 月收入
            income_base = {'高中及以下': 3000, '大专': 4500, '本科': 7000,
                           '硕士': 10000, '博士': 15000}
            income = income_base[edu] * (1 + 0.5 * pop_norm_city) * \
                     np.random.lognormal(mean=0, sigma=0.2)
            income = round(income, 1)
            
            # 房租
            rent = 500 + 4500 * pop_norm_city + np.random.normal(0, 500)
            rent = max(200, round(rent, 1))
            
            # 通勤时间
            commute = 10 + 45 * pop_norm_city + np.random.normal(0, 5)
            commute = max(5, round(commute, 1))
            
            # 通勤满意度 (1-10)
            commute_satisfaction = 9 - 3 * pop_norm_city + np.random.normal(0, 1)
            commute_satisfaction = np.clip(commute_satisfaction, 1, 10)
            commute_satisfaction = round(commute_satisfaction, 1)
            
            # 社交网络
            wechat_contacts = int(20 + 60 * pop_norm_city + np.random.normal(0, 15))
            wechat_contacts = max(5, wechat_contacts)
            friends_met_weekly = 6 - 3.5 * pop_norm_city + np.random.normal(0, 1)
            friends_met_weekly = max(0.5, round(friends_met_weekly, 1))
            
            # 孤独感 (1-10)
            loneliness = 2.5 + 4.0 * pop_norm_city + np.random.normal(0, 1)
            loneliness = np.clip(loneliness, 1, 10)
            loneliness = round(loneliness, 1)
            
            # 工作满意度
            job_satisfaction = 5 + 2 * pop_norm_city + np.random.normal(0, 1.5)
            job_satisfaction = np.clip(job_satisfaction, 1, 10)
            job_satisfaction = round(job_satisfaction, 1)
            
            # 住房满意度
            housing_satisfaction = 8 - 3 * pop_norm_city + np.random.normal(0, 1.5)
            housing_satisfaction = np.clip(housing_satisfaction, 1, 10)
            housing_satisfaction = round(housing_satisfaction, 1)
            
            # 生活满意度
            life_satisfaction = 6 + 1 * pop_norm_city - 2 * (loneliness / 10) + \
                                np.random.normal(0, 1.5)
            life_satisfaction = np.clip(life_satisfaction, 1, 10)
            life_satisfaction = round(life_satisfaction, 1)
            
            # 主观幸福感
            well_being = (job_satisfaction + housing_satisfaction + 
                         life_satisfaction - 0.5 * loneliness) / 3.5
            well_being = np.clip(well_being, 1, 10)
            well_being = round(well_being, 1)
            
            records.append({
                'individual_id': i,
                'city_id': city,
                'age_group': age,
                'age': age_val,
                'education': edu,
                'monthly_income': income,
                'monthly_rent': rent,
                'commute_minutes': commute,
                'commute_satisfaction': commute_satisfaction,
                'wechat_contacts': wechat_contacts,
                'friends_met_weekly': friends_met_weekly,
                'loneliness_score': loneliness,
                'job_satisfaction': job_satisfaction,
                'housing_satisfaction': housing_satisfaction,
                'life_satisfaction': life_satisfaction,
                'well_being_score': well_being,
                'city_population': round(city_pop, 2),
            })
        
        df = pd.DataFrame(records)
        self.family['tier4_individual_survey'] = df
        return df
    
    # ---- Tier 5: 城际联系数据 (网络) ----
    
    def generate_tier5_intercity_network(self) -> pd.DataFrame:
        """
        Tier 5: 城际联系数据 (200城市 × 200城市)
        
        核心指标:
        - 人口流动 (通勤/移民)
        - 资本流动 (投资额)
        - 信息流动 (百度搜索指数/电话)
        - 供应链联系
        - 协同创新 (合著论文/联合专利)
        """
        n = self.config.n_cities
        
        # 城市坐标 (模拟地理空间)
        lats = np.random.uniform(18, 54, size=n)  # 纬度
        lons = np.random.uniform(73, 135, size=n)  # 经度
        
        # 城市人口
        city_panel = self.family.get('tier1_city_panel', pd.DataFrame())
        if not city_panel.empty:
            pop = city_panel[city_panel['year'] == 2020]['population_10k'].values
        else:
            pop = lognormal_population(n)
        
        records = []
        
        for i in range(n):
            for j in range(i + 1, n):
                # 地理距离 (Haversine)
                dlat = np.radians(lats[i] - lats[j])
                dlon = np.radians(lons[i] - lons[j])
                a = np.sin(dlat/2)**2 + np.cos(np.radians(lats[i])) * \
                    np.cos(np.radians(lats[j])) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                distance = 6371 * c  # km
                
                if distance < 10:
                    continue  # 同一个城市, 跳过
                
                # 引力模型: 人口流动 ∝ (pop_i * pop_j) / distance^2
                gravity = (pop[i] * pop[j]) / (distance ** 1.5)
                gravity = gravity * np.random.lognormal(mean=0, sigma=0.5)
                
                # 人口流动量
                flow_people = int(gravity * np.random.uniform(10, 100))
                if flow_people < 1:
                    flow_people = 0
                
                # 资本流动
                flow_capital = gravity * np.random.uniform(0.1, 1.0) * 1e6
                
                # 信息流动 (逆距离衰减)
                info_flow = np.exp(-distance / 500) * (pop[i] * pop[j]) / 1000
                info_flow = info_flow * np.random.lognormal(mean=0, sigma=0.3)
                
                # 供应链联系
                supply_chain = np.exp(-distance / 300) * np.random.beta(2, 2)
                
                # 协同创新
                co_patents = int(np.exp(-distance / 200) * 
                                 (pop[i] * pop[j]) / 10000 * 
                                 np.random.lognormal(mean=0, sigma=0.5))
                if co_patents < 0:
                    co_patents = 0
                
                records.append({
                    'city_i': i,
                    'city_j': j,
                    'distance_km': round(distance, 1),
                    'flow_people': flow_people,
                    'flow_capital_million': round(flow_capital / 1e6, 2),
                    'info_flow_index': round(info_flow, 2),
                    'supply_chain_strength': round(supply_chain, 3),
                    'co_patents': co_patents,
                    'gravity_score': round(gravity, 2),
                })
        
        df = pd.DataFrame(records)
        self.family['tier5_intercity_network'] = df
        return df
    
    # ---- 全家族生成 ----
    
    def generate_all(self, verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """生成完整数据家族"""
        if verbose:
            print("=" * 70)
            print("  集聚效益数据家族生成器")
            print("=" * 70)
        
        if verbose:
            print("\n[1/5] 生成 Tier 1: 跨城市面板数据...")
        self.generate_tier1_city_panel()
        if verbose:
            print(f"        → {len(self.family['tier1_city_panel']):,} 条记录")
            print(f"        → {self.config.n_cities} 城市 × {self.config.n_years} 年")
            print(f"        → 20+ 指标 (人口/GDP/工资/专利/就业...)")
        
        if verbose:
            print("\n[2/5] 生成 Tier 2: 产业集聚数据...")
        self.generate_tier2_industry_cluster()
        if verbose:
            n_rec = len(self.family['tier2_industry_cluster'])
            print(f"        → {n_rec:,} 条记录")
            print(f"        → {self.config.n_cities} 城市 × {self.config.n_industries} 产业")
        
        if verbose:
            print("\n[3/5] 生成 Tier 3: 企业微观数据...")
        self.generate_tier3_firm_data()
        if verbose:
            print(f"        → {len(self.family['tier3_firm_data']):,} 条记录")
            print(f"        → 涵盖 4 种规模类型 (微型/小型/中型/大型)")
        
        if verbose:
            print("\n[4/5] 生成 Tier 4: 个体感知数据...")
        self.generate_tier4_individual_survey()
        if verbose:
            print(f"        → {len(self.family['tier4_individual_survey']):,} 条记录")
            print(f"        → 5 个年龄组 × 5 个教育水平")
        
        if verbose:
            print("\n[5/5] 生成 Tier 5: 城际联系数据...")
        self.generate_tier5_intercity_network()
        if verbose:
            print(f"        → {len(self.family['tier5_intercity_network']):,} 条记录")
            print(f"        → 基于引力模型的人口/资本/信息流动")
        
        if verbose:
            print("\n" + "=" * 70)
            print("  数据家族生成完成!")
            print("=" * 70)
            total = sum(len(df) for df in self.family.values())
            print(f"  总记录数: {total:,}")
            print(f"  总指标数: 100+")
            print(f"  数据层级: 5 (宏观→中观→微观→个体→网络)")
            print(f"  覆盖城市: {self.config.n_cities}")
            print("=" * 70)
        
        return self.family


# ============================================================================
# 第3层: 数据家族分析工具
# ============================================================================

class AgglomerationAnalyzer:
    """
    集聚效益数据分析器
    
    基于数据家族进行各层级的分析计算
    """
    
    def __init__(self, data_family: Dict[str, pd.DataFrame]):
        self.family = data_family
    
    def compute_urban_scaling(self, metric: str = 'gdp_billion') -> dict:
        """
        计算城市标度律: ln(Y) = ln(Y0) + β * ln(N)
        返回 β 和拟合优度
        """
        df = self.family['tier1_city_panel']
        df = df[df['year'] == 2020].copy()
        
        ln_pop = np.log(df['population_10k'])
        ln_metric = np.log(df[metric])
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(ln_pop, ln_metric)
        
        return {
            'beta': round(slope, 3),
            'intercept': round(intercept, 3),
            'r_squared': round(r_value ** 2, 3),
            'p_value': p_value,
            'is_super_linear': slope > 1,
            'interpretation': (
                f"β = {slope:.3f}: "
                f"{'超线性' if slope > 1 else '线性' if np.isclose(slope, 1, rtol=0.01) else '次线性'}"
                f" → 人口每增长1%, "
                f"{metric}增长 {slope:.1f}%"
            )
        }
    
    def compute_lq_summary(self) -> pd.DataFrame:
        """计算各城市的区位商概览"""
        df = self.family['tier2_industry_cluster']
        # 每种产业类型在每个城市的平均LQ
        lq_summary = df.groupby(['city_id', 'industry_type'])['location_quotient'].mean().reset_index()
        # 特色产业: LQ > 1.25
        lq_summary['is_specialized'] = lq_summary['location_quotient'] > 1.25
        return lq_summary
    
    def compute_agglomeration_benefit(self) -> pd.DataFrame:
        """
        计算集聚效益的综合得分
        排名前20% vs 后20% 城市的差异
        """
        df = self.family['tier1_city_panel']
        df = df[df['year'] == 2020].copy()
        
        # 按人口排序
        df['rank'] = df['population_10k'].rank(pct=True)
        df['group'] = pd.cut(df['rank'], bins=[0, 0.2, 0.8, 1.0],
                             labels=['后20%', '中间60%', '前20%'])
        
        # 分组对比
        metrics = ['gdp_per_capita_10k', 'avg_wage', 'patents', 
                   'n_firms', 'infrastructure_index', 'doctors_per_1000']
        
        comparison = df.groupby('group')[metrics].mean()
        
        # 前20% vs 后20% 的倍数
        if len(comparison) >= 2:
            ratios = comparison.iloc[-1] / comparison.iloc[0]
        else:
            ratios = pd.Series([1.0] * len(metrics), index=metrics)
        
        return comparison, ratios
    
    def compute_wellbeing_curve(self) -> dict:
        """
        计算"集聚-幸福感"曲线
        检验是否存在倒U型关系
        """
        df = self.family['tier4_individual_survey']
        
        # 按城市人口分组
        df['pop_group'] = pd.qcut(df['city_population'], q=10, labels=False)
        
        # 每组幸福感均值
        curve = df.groupby('pop_group')['well_being_score'].mean()
        
        pop_means = df.groupby('pop_group')['city_population'].mean()
        
        return {
            'pop_groups': pop_means.values.tolist(),
            'wellbeing': curve.values.tolist(),
            'peak_population': pop_means.values[curve.argmax()] if len(curve) > 0 else None,
            'peak_wellbeing': curve.max() if len(curve) > 0 else None,
        }
    
    def full_report(self) -> dict:
        """生成完整分析报告"""
        scaling = self.compute_urban_scaling()
        wellbeing = self.compute_wellbeing_curve()
        
        # 集聚效益倍数
        _, ratios = self.compute_agglomeration_benefit()
        
        return {
            'urban_scaling': scaling,
            'agglomeration_multipliers': ratios.to_dict(),
            'wellbeing_curve': wellbeing,
        }


# ============================================================================
# 第4层: 主程序入口
# ============================================================================

if __name__ == '__main__':
    import sys
    import os
    
    output_dir = '/workspace'
    
    # 生成数据家族
    print("=" * 70)
    print("  集聚效益数据家族 (Agglomeration Economy Data Family)")
    print("=" * 70)
    
    generator = AgglomerationDataFamily()
    family = generator.generate_all(verbose=True)
    
    # 分析
    print("\n\n  分析报告")
    print("-" * 70)
    analyzer = AgglomerationAnalyzer(family)
    report = analyzer.full_report()
    
    print(f"\n  城市标度律: β = {report['urban_scaling']['beta']}")
    print(f"  解释: {report['urban_scaling']['interpretation']}")
    print(f"  R² = {report['urban_scaling']['r_squared']}")
    
    print(f"\n  集聚效益 (前20% vs 后20% 城市):")
    for metric, ratio in report['agglomeration_multipliers'].items():
        print(f"    {metric}: {ratio:.2f}x")
    
    wb = report['wellbeing_curve']
    if wb['peak_population']:
        print(f"\n  幸福感峰值: 人口 {wb['peak_population']:.0f} 万时")
        print(f"  幸福感得分: {wb['peak_wellbeing']:.2f}/10")
    
    # 导出CSV
    print(f"\n\n  导出数据家族...")
    tier_names = {
        'tier1_city_panel': 'T1_跨城市面板',
        'tier2_industry_cluster': 'T2_产业集聚',
        'tier3_firm_data': 'T3_企业微观',
        'tier4_individual_survey': 'T4_个体感知',
        'tier5_intercity_network': 'T5_城际联系',
    }
    
    for key, name in tier_names.items():
        path = os.path.join(output_dir, f'{name}.csv')
        family[key].to_csv(path, index=False, encoding='utf-8-sig')
        print(f"    ✓ {name}.csv  ({len(family[key]):,} 条)")
    
    print(f"\n  数据家族文件已导出至: {output_dir}")
    print("=" * 70)