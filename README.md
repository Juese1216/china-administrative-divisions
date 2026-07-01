# 国家行政区划与历史变更信息管理系统

本项目围绕民政部县级以上行政区划变更文本和行政区划编码表，完成 NER、
RE、变更类型分类、知识图谱导出、时序预测和 Flask + ECharts 可视化系统。

代码已经按课程功能直接整合到 `src/` 顶层脚本中，Web 相关文件统一放在
`web/`，乡镇街道和自然村预测实现保留在 `src/rural_atlas/`。默认运行只生成
Web 展示和系统分析需要的关键 CSV；需要完整中间表时，各入口可追加
`--debug-output`。

## 工程结构

```text
.
├── config/          # 乡镇街道/自然村时序项目配置
├── data/source/     # 可复现源数据和来源登记
├── docs/            # 流程、结果和提交说明
├── scripts/         # 辅助脚本
├── src/             # NLP、RE、分类、图谱和预测主流程
├── tests/           # 工程结构检查
└── web/             # Flask + ECharts 可视化系统
```

`data/processed/`、`data/app/`、`models/`、日志和缓存文件已在 `.gitignore`
中忽略，提交到 GitHub 时默认不会包含这些运行时产物。

## 环境安装

所有 Python 命令默认使用项目 conda 环境：

```bash
conda run -n nlpEnv python ...
```

也可以在当前 Python 环境中安装依赖：

```bash
python -m pip install -r requirements.txt
```

如果要使用 `pyproject.toml` 提供的命令入口，可以执行：

```bash
python -m pip install -e ".[nlp,dev]"
```

提交或交付前建议先跑一次轻量工程检查：

```bash
make smoke
```

## 分步运行

1. 探索性 NLP 统计：PaddleNLP 切词 + pandas 动词/关系候选统计。

```bash
conda run --no-capture-output -n nlpEnv python src/paddle_nlp_tokenize.py
conda run --no-capture-output -n nlpEnv python src/relation_statistics.py
```

2. 命名实体识别（NER）：正则 + 行政区划词表 + 可选 UIE 增强。

```bash
conda run --no-capture-output -n nlpEnv python src/ner.py
```

3. 关系抽取（RE）：关系类别归档、句内关系抽取、记录级三元组整理。

```bash
conda run --no-capture-output -n nlpEnv python src/relation.py
```

4. 分类任务：规则分类 + 全量弱监督文本分类器。

```bash
conda run --no-capture-output -n nlpEnv python src/classify.py
```

5. 知识图谱：Neo4j 导入表、静态树图和动态时间轴。

```bash
conda run --no-capture-output -n nlpEnv python src/graph.py
```

6. 时序预测：pandas 年度聚合 + statsmodels/sklearn 预测 + matplotlib 图表。

```bash
conda run --no-capture-output -n nlpEnv python src/forecast.py
```

7. 时序预测 V2：乡镇街道与自然村数量图谱。

```bash
PYTHONPATH=src conda run --no-capture-output -n nlpEnv python -m rural_atlas.cli quantity-atlas
```

8. 构建 Web 仪表盘 SQLite 数据库。

```bash
conda run --no-capture-output -n nlpEnv python web/webdb.py
```

9. 启动 Flask + ECharts Web 仪表盘。

```bash
conda run --no-capture-output -n nlpEnv python web/web_app.py
```

如本机 `5000` 端口已被占用，可以改用其他端口：

```bash
PORT=5001 conda run --no-capture-output -n nlpEnv python web/web_app.py
```

## 正式脚本

| 功能 | 脚本 |
| --- | --- |
| PaddleNLP 切词 | `src/paddle_nlp_tokenize.py` |
| 关系候选统计 | `src/relation_statistics.py` |
| NER | `src/ner.py` |
| RE | `src/relation.py` |
| 分类 | `src/classify.py` |
| 知识图谱 | `src/graph.py` |
| 时序预测 | `src/forecast.py` |
| 乡镇街道/自然村数量预测 | `src/rural_atlas/` |
| Web 数据库 | `web/webdb.py` |
| Web 后端 | `web/web_app.py` |

## 当前关键结果

- NER：1771 个句子，7716 个实体，零实体句 0。
- RE：511 条变更记录，5445 条动态三元组，42876 条静态隶属三元组。
- 分类：13 类关系标签，407 条记录级训练样本 + 1355 条句子级补充样本参与多标签建模，标签准确率 0.9593，Micro-F1 为 0.9354。
- 知识图谱：46721 个节点，54793 条边。
- 时序预测：1999-2026 年历史统计，预测 2027-2031 年趋势。
- Web 数据库：`data/app/dashboard.sqlite`，17 张业务表。

## 文档入口

- [文件说明](docs/file_index.md)
- [工程结构与 GitHub 提交说明](docs/project_structure.md)
- [NLP 统计](docs/nlp_relation_stats.md)
- [NER 流程](docs/ner_rule_uie_workflow.md)
- [RE 输出](docs/relation_extraction.md)
- [13 类关系例句](docs/relation_examples.md)
- [分类任务](docs/classification.md)
- [知识图谱](docs/knowledge_graph.md)
- [时序预测](docs/time_series_forecast.md)
- [时序预测 V2](docs/rural_time_series_forecast.md)
- [Web 仪表盘](docs/web_dashboard.md)
