# 时序预测 V2：乡镇街道与自然村数量图谱

## 当前定位

这是本项目的第二套时序预测结果，和 `src/forecast.py` 主线预测的口径不同：

1. V1 预测的是“县级以上行政区划变更记录数量”。
2. V2 预测的是“乡镇街道数量、自然村数量、人口格网演化”，用于补充展示基层行政区划和聚落数量变化。

V2 已整合进主项目；原临时目录后续不再使用。

## 运行命令

重新生成数量图谱：

```bash
PYTHONPATH=src conda run --no-capture-output -n nlpEnv python -m rural_atlas.cli quantity-atlas
```

查看命令帮助：

```bash
PYTHONPATH=src conda run --no-capture-output -n nlpEnv python -m rural_atlas.cli --help
```

## 输入文件

| 文件 | 作用 |
| --- | --- |
| `data/source/rural_time_series/raw/china_provinces_datav_100000_full.json` | 省级边界，用于地图可视化和人口聚合定位。 |
| `data/source/rural_time_series/raw/gaohr/extracted/CN_streets_*.csv` | 2017-2023 年乡镇街道层级 CSV。 |
| `data/source/rural_time_series/raw/stats_yearbook/official_province_admin_2009_2013.csv` | 2009-2013 年统计年鉴省级乡级区划数量。 |
| `data/processed/rural_time_series/ghsl_population_atlas/panel_full_2000_2035.csv` | GHSL 人口格网历史和预测面板，用作外部人口回归因子。 |

## 输出文件

输出目录：

```text
data/processed/rural_time_series/
```

| 文件 | 作用 |
| --- | --- |
| `quantity_atlas/乡镇街道数量_2009_2026.csv` | 省级乡镇街道数量历史和现势估计。 |
| `quantity_atlas/乡镇街道数量_2009_2035.csv` | 省级乡镇街道数量历史、现势估计和预测。 |
| `quantity_atlas/官方年鉴_乡级行政区划数量_2009_2035.csv` | 全国乡级行政区划数量趋势。 |
| `quantity_atlas/自然村数量_2006_2035.csv` | 全国自然村数量趋势。 |
| `quantity_atlas/外部回归因子_人口_2000_2035.csv` | 省级人口外部回归因子。 |
| `quantity_atlas/数量变化图谱.html` | 乡镇街道和自然村数量变化交互图谱。 |
| `quantity_atlas/乡镇街道数量地图.html` | 省级乡镇街道数量交互地图。 |
| `quantity_atlas/数量趋势仪表盘.html` | 数量趋势仪表盘。 |
| `ghsl_population_atlas/atlas.html` | 人口格网时序图谱。 |
| `ghsl_population_atlas/dashboard.html` | 人口格网趋势仪表盘。 |

## 当前结果

| 指标 | 数值 |
| --- | ---: |
| 省级乡镇街道数量面板 | 837 行，2009-2035 |
| 全国乡级行政区划数量序列 | 27 行，2009-2035 |
| 全国自然村数量序列 | 30 行，2006-2035 |
| 人口格网完整面板 | 122688 行，2000-2035 |
| 预测起始年 | 2027 |
| 预测结束年 | 2035 |

## 方法说明

- 省级乡镇街道数量使用历史数量趋势和外部人口回归因子，采用阻尼 Huber 趋势模型。
- 自然村数量使用全国农业普查公开点位：2016 年末约 317 万个，并结合 2006-2016 降幅构造趋势。
- 2024-2026 中没有完整年度观测的位置标为现势估计，不当作真实观测。
- 2027-2035 标为统计预测。
- 原始 GHSL zip/tif 栅格体量很大，没有并入主项目；当前保留的是聚合后的 CSV、图谱 HTML 和来源登记。
