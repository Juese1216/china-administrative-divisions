# 文件说明

## 源数据

| 路径 | 作用 |
| --- | --- |
| `data/source/mca_changes/` | 民政部年度县级以上行政区划变更文本。 |
| `data/source/mca_changes/manifest.csv` | 年度 txt 文件清单。 |
| `data/source/admin_codes/行政区划编码表_20260427.xlsx` | 行政区划编码表，包含省市区和乡镇街道两个工作表。 |
| `data/source/economic_indicators/china_macro_indicators.csv` | 中国 GDP、人口等年度经济指标，用作时序预测外部回归因子。 |
| `data/source/rural_time_series/` | 时序预测 V2 的来源登记、边界、乡镇街道 CSV 和统计年鉴材料。 |

## 脚本

| 路径 | 作用 |
| --- | --- |
| `src/paddle_nlp_tokenize.py` | 使用 PaddleNLP 对原始文本做切词和词性标注。 |
| `src/relation_statistics.py` | 用 pandas 统计动词频次和关系候选。 |
| `src/ner.py` | 使用正则、行政区划词表和可选 UIE 做 NER。 |
| `src/relation.py` | 关系抽取脚本，包含关系类别归档、细粒度 RE 和主线 RE。 |
| `src/classify.py` | 分类脚本，运行规则分类和弱监督文本分类器。 |
| `src/graph.py` | 知识图谱脚本，导出 Neo4j 表、树图和时间轴。 |
| `src/forecast.py` | 时序预测脚本，包含经济指标准备、年度聚合、预测和 matplotlib 作图。 |
| `src/rural_atlas/` | 时序预测 V2 的数量图谱、人口格网图谱和预测实现。 |
| `web/webdb.py` | Web 数据库脚本，把正式 CSV 写入 SQLite。 |
| `web/web_app.py` | Flask Web 仪表盘后端，提供页面路由、异步 JSON API 和 SQLite 自动更新检查。 |
| `web/templates/base.html` | Web 公共模板，包含顶部导航、表格分页工具、通用链接和图表自适应逻辑。 |
| `web/templates/dashboard.html` | 首页总览，展示数据量、年度趋势、分类分布和系统入口。 |
| `web/templates/areas.html` | 行政区划浏览页，支持按省、市、区县、乡镇街道逐级钻取。 |
| `web/templates/area_detail.html` | 区划详情页，展示下级区划、相关变更、相关三元组、NER 实体和局部知识图谱。 |
| `web/templates/graph.html` | 知识图谱页，包含全国母图谱和按关键词搜索的局部动态图谱。 |
| `web/templates/ner.html` | NER 展示页，包含实体类型分布、高频实体和分页明细表。 |
| `web/templates/relations.html` | RE 展示页，包含关系类型分布和动态三元组表。 |
| `web/templates/classification.html` | 分类展示页，包含类别统计、弱监督文本分类指标和分类记录表。 |
| `web/templates/forecast.html` | 主线时序预测页，展示历史曲线、2027 年之后预测曲线、置信区间和年份明细。 |
| `web/templates/forecast_v2.html` | 时序预测 V2 页面，展示乡镇街道、自然村和人口外部因子趋势。 |
| `web/static/css/style.css` | Web 统一样式，负责深色导航、面板、表格、分页、行政区划三栏浏览和响应式布局。 |

## 工程文件

| 路径 | 作用 |
| --- | --- |
| `pyproject.toml` | Python 工程元数据、可编辑安装配置和命令入口。 |
| `requirements.txt` | 完整运行依赖，包含 PaddleNLP、Web、预测和图谱相关库。 |
| `Makefile` | 常用安装、检查、运行命令封装。 |
| `LICENSE` | MIT 开源协议文本。 |
| `.gitignore` | GitHub 提交忽略规则，排除生成产物、缓存、日志和虚拟环境。 |
| `tests/test_project_structure.py` | 不依赖第三方库的项目结构、入口和源数据清单检查。 |
| `docs/project_structure.md` | 工程结构与 GitHub 提交说明。 |

## 输出目录

| 路径 | 作用 |
| --- | --- |
| `data/processed/paddle_nlp_tokens/` | PaddleNLP token 表和句子表，供探索统计复用。 |
| `data/processed/relation_statistics/` | 动词频次、关系候选和统计总览。 |
| `data/processed/relation_schema/` | 归档后的关系类别表、触发词表和总览。 |
| `data/processed/ner_rule_uie/` | NER 实体、实体频次、句式线索和总览。 |
| `data/processed/relation_details/` | 细粒度辅助关系三元组、关系类型统计和总览。 |
| `data/processed/relation_extraction/` | 主线 RE 输出，包括记录表、动态三元组、静态隶属三元组。 |
| `data/processed/classification/` | 规则分类、弱监督文本分类评估、混淆矩阵和预测结果。 |
| `data/processed/knowledge_graph/` | Neo4j 导入 CSV、Cypher 脚本、树图和时间轴数据。 |
| `data/processed/time_series_forecast/` | 年度统计、未来预测、模型评估和 matplotlib 趋势图。 |
| `data/processed/rural_time_series/` | 时序预测 V2 输出，包括乡镇街道数量图谱和自然村趋势。 |
| `data/app/dashboard.sqlite` | Web 仪表盘使用的 SQLite 缓存库，包含索引后的结构化查询表和 `web_data_sources` 数据源审计表。 |

## 备份

| 路径 | 作用 |
| --- | --- |
| `backup/` | 旧项目、旧实验流程和不再使用的实现备份。 |
