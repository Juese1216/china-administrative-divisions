# 工程结构与 GitHub 提交说明

本项目按“数据源 + 可复现脚本 + Web 展示 + 文档”的方式组织，适合直接作为
GitHub 仓库提交。

## 顶层目录

| 路径 | 说明 |
| --- | --- |
| `src/` | NLP、分类、知识图谱、时序预测等核心脚本；`src/rural_atlas/` 是可用 `python -m rural_atlas.cli` 运行的子包。 |
| `web/` | Flask 后端、Jinja2 模板、ECharts 静态资源和 SQLite 构建脚本。 |
| `data/source/` | 原始数据与来源登记，建议随仓库提交，便于复现。 |
| `data/processed/` | 由脚本生成的中间表、图表和模型输入输出，已在 `.gitignore` 中忽略。 |
| `data/app/` | Flask 仪表盘运行时 SQLite 缓存和日志，已忽略。 |
| `config/` | 乡镇街道和自然村时序图谱配置。 |
| `docs/` | 功能说明、流程说明和结果解释。 |
| `scripts/` | 辅助一次性脚本。 |
| `tests/` | 不依赖重型第三方库的工程结构检查。 |

## 推荐提交内容

建议提交：

- `src/`、`web/`、`scripts/`、`config/`、`docs/`
- `data/source/` 中用于复现的源数据和来源登记
- `README.md`、`requirements.txt`、`pyproject.toml`、`Makefile`
- `LICENSE`、`.gitignore`

不建议提交：

- `data/processed/`
- `data/app/`
- `models/`
- `.DS_Store`、日志、缓存、虚拟环境

## GitHub 提交前检查

```bash
make smoke
git status --short
```

首次推送到 GitHub 时：

```bash
git init
git add .
git commit -m "Initial project structure"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```
