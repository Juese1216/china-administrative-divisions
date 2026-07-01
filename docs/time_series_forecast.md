# 时序预测

## 当前定位

时序预测负责把分类结果按年聚合，统计每年去重后的变更记录总数和 13 类关系标签数量，并预测未来 5 年的变更趋势。

当前实现包含三种模型：

1. Holt-Winters 指数平滑，用于单变量趋势外推。
2. 年份线性回归，用于单变量对照。
3. GDP、人口等外部回归因子的 Poisson 回归，用于多变量时序预测。

## 运行命令

先准备经济指标：

```bash
conda run --no-capture-output -n nlpEnv python src/forecast.py --refresh-economic
```

再运行预测：

```bash
conda run --no-capture-output -n nlpEnv python src/forecast.py
```

如果临时不想使用 GDP、人口等外部回归因子，可以运行：

```bash
conda run --no-capture-output -n nlpEnv python src/forecast.py --disable-exogenous
```

## 输入文件

| 文件 | 作用 |
| --- | --- |
| `data/processed/classification/rule_classification.csv` | 每条变更记录展开后的 13 类关系标签。 |
| `data/source/economic_indicators/china_macro_indicators.csv` | GDP、人口、人均 GDP 等经济指标。 |

## 输出文件

输出目录：

```text
data/processed/time_series_forecast/
```

| 文件 | 作用 |
| --- | --- |
| `annual_total.csv` | 年度变更总数；缺失年份会补 0。 |
| `annual_type_counts_long.csv` | 按年、按类型展开的长表，适合前端图表。 |
| `forecast_total.csv` | 未来 5 年总变更次数预测。 |
| `forecast_by_type.csv` | 未来 5 年各类别变更次数预测。 |
| `model_metrics.csv` | 各模型在历史数据上的 MAE 和 RMSE。 |
| `forecast_overview.csv` | 时序预测总览。 |
| `trend_forecast_total.png` | 总量趋势外推图。 |
| `type_trends_and_forecast.png` | 各类变更趋势外推图。 |

如需外部因子特征、特征权重、高变更年份和可视化 JSON，可运行
`src/forecast.py --debug-output` 生成调试输出。

## 当前结果

| 指标 | 数值 |
| --- | ---: |
| 记录数 | 511 |
| 历史年份范围 | 1999-2026 |
| 历史年份数 | 28 |
| 预测年份范围 | 2027-2031 |
| 类型数 | 5 |
| 历史最高变更年份 | 2000 |
| 历史最高变更数 | 51 |
| 预测期最高年份 | 2027 |
| 预测期最高值 | 3.7155 |

## 年份说明

- `2020` 年有源文件，并且当前 RE 输出中有 13 条变更记录。
- `2022` 年没有民政部源 txt，因此 `annual_total.csv` 中按 0 条补齐。
- `2021`、`2023`、`2024`、`2025`、`2026` 已进入主线 RE、分类和时序统计。
- 预测从 `2027` 年开始，不把 `2021-2026` 当作未来预测年。
