# Web 仪表盘说明

本模块使用 Flask + ECharts 构建交互式 Web 系统，统一展示行政区划浏览、NER、RE、分类、知识图谱和两版时序预测结果。Web 数据源只使用当前项目 `data/processed/` 下的新一轮输出，并统一缓存到 `data/app/dashboard.sqlite`。

## 运行命令

先构建 SQLite 缓存库：

```bash
conda run --no-capture-output -n nlpEnv python web/webdb.py
```

启动 Flask 时也会自动检查 SQLite 是否比任一源 CSV 更旧；如果源 CSV 更新过，会自动重建 SQLite。

再启动 Flask：

```bash
conda run --no-capture-output -n nlpEnv python web/web_app.py
```

默认访问地址：

```text
http://127.0.0.1:5000/
```

如果 `5000` 端口被占用，可以指定端口：

```bash
PORT=5001 conda run --no-capture-output -n nlpEnv python web/web_app.py
```

## 文件作用

| 文件 | 作用 |
| --- | --- |
| `web/webdb.py` | Web 数据库脚本，读取正式 CSV 并写入 `data/app/dashboard.sqlite`，同时生成数据源审计和索引。 |
| `web/web_app.py` | Flask 后端，提供页面路由、JSON API、SQLite 自动更新检查和时序预测 V2 的 HTML 图谱文件访问入口。 |
| `web/templates/base.html` | 公共模板，包含顶部导航、表格分页、通用链接和图表自适应逻辑。 |
| `web/templates/dashboard.html` | 首页总览，展示数据量、年度趋势、分类分布和系统入口。 |
| `web/templates/areas.html` | 行政区划浏览页，支持按省、市、区县、乡镇街道逐级钻取。 |
| `web/templates/area_detail.html` | 区划详情页，展示下级区划、相关变更、相关三元组、NER 实体和局部知识图谱。 |
| `web/templates/graph.html` | 知识图谱页，包含全国母图谱和关键词局部图谱。 |
| `web/templates/ner.html` | NER 展示页，包含实体类型分布、高频实体和实体明细。 |
| `web/templates/relations.html` | RE 展示页，包含关系类型分布和动态变更三元组。 |
| `web/templates/classification.html` | 分类展示页，包含类别统计、弱监督文本分类指标和分类记录。 |
| `web/templates/forecast.html` | 主线时序预测页，展示历史曲线、2027 年之后预测曲线、置信区间和年份明细。 |
| `web/templates/forecast_v2.html` | 时序预测 V2 页面，展示乡镇街道数量、自然村数量和人口外部因子趋势。 |
| `web/static/css/style.css` | 统一视觉样式，包含深色导航、面板、表格、分页、行政区划三栏浏览和响应式布局。 |

## 页面栏目

| 栏目 | 内容 |
| --- | --- |
| 首页 | 展示数据总量、年度趋势、变更类型分布、行政区划层级分布和系统入口。 |
| 行政区划浏览 | 使用行政区划编码表进行省、市、区县、乡镇街道逐级钻取。 |
| 区划详情 | 展示某一区划的下级区划、相关变更记录、相关关系三元组、相关 NER 实体和局部图谱。 |
| 知识图谱 | 默认展示全国母图谱，搜索后展示局部动态图谱；图谱节点可继续进入区划详情。 |
| NER | 支持按实体类型和关键词筛选，展示高频实体、实体类型图和实体明细分页表。 |
| 关系抽取 | 展示动态变更三元组，支持按关系类型、年份、关键词筛选。 |
| 分类结果 | 展示类别分布、弱监督文本分类器指标和分类记录表。 |
| 时序预测 | 展示历史变更曲线、2027 年之后预测曲线和 95% 置信区间，可切换预测目标。 |
| 时序预测 V2 | 整合乡镇街道数量、自然村数量、人口外部回归因子，并链接到 V2 原有交互式 HTML 图谱。 |

## 数据库表

| 表 | 来源 |
| --- | --- |
| `records` | 主线 RE 记录级结果。 |
| `dynamic_triples` | 动态变更三元组。 |
| `static_admin_nodes` | 行政区划编码表生成的静态区划节点。 |
| `static_affiliation_triples` | 静态隶属关系三元组。 |
| `ner_entities` | NER 实体结果。 |
| `class_frequency`、`rule_classification`、`ml_confusion_matrix`、`ml_evaluation` | 分类任务结果。 |
| `annual_total`、`annual_type_counts_long`、`forecast_total`、`forecast_by_type`、`forecast_metrics` | 主线时序预测结果。 |
| `rural_township_province`、`rural_township_national`、`rural_natural_village` | 时序预测 V2 结果。 |
| `web_data_sources` | Web 数据源审计表，记录每个 SQLite 表来自哪个新版 CSV、源文件大小、源文件时间戳和写入行数。 |

## 主要 API

| API | 作用 |
| --- | --- |
| `/api/overview` | 首页概览统计。 |
| `/api/areas/provinces`、`/api/areas/children`、`/api/area/<code>` | 行政区划浏览和区划详情数据。 |
| `/api/entities` | NER 图表和明细。 |
| `/api/relations` | RE 关系类型分布和动态三元组表。 |
| `/api/classifications` | 分类分布、弱监督文本分类指标和分类记录。 |
| `/api/graph/overview`、`/api/graph/search`、`/api/graph/node` | 全国母图谱、局部图谱和节点详情。 |
| `/api/forecast`、`/api/forecast/year/<year>` | 主线历史曲线、预测曲线、置信区间和年份明细。 |
| `/api/forecast/rural` | 时序预测 V2 的全国、省级和自然村趋势数据。 |
| `/api/data-sources` | 查看 Web 当前使用的 CSV 来源、行数、文件大小和时间戳。 |
