# PaddleNLP 切词与关系候选统计

## 本步目标

本步只做 NLP 切词、词性标注和统计分析，用来初步判断行政区划变更文本里可能有哪些关系类型。

这里刻意不处理地名，也不使用行政区划词表修正切词结果。原因是：

- 这一步的目标不是最终 NER，而是先观察高频动词和动词上下文。
- PaddleNLP 对地名的切分即使不完美，也不会影响我们先判断“撤销、设立、划归、更名、迁至”等关系候选。
- 地名、区划代码、省市区层级会放到后面的 NER 步骤中单独处理。

## 运行环境

所有命令默认在项目 conda 环境 `nlpEnv` 中运行。

```bash
conda run -n nlpEnv python ...
```

## 脚本分工

现在本步骤拆成两个互相独立的顶层脚本：

| 模块 | 作用 | 是否调用 PaddleNLP |
| --- | --- | --- |
| `src/paddle_nlp_tokenize.py` | 读取原始 txt，调用 PaddleNLP 进行切词和词性标注，输出 token CSV。 | 是 |
| `src/relation_statistics.py` | 读取上一步生成的 token CSV，用 pandas 统计所有动词的频次和关系候选排序。 | 否 |
| `src/relation.py` | 在 RE 全流程中读取关系统计 CSV 和 NER 句式 cue，归档最终 RE 可用关系类别。 | 否 |

这样做的好处是：如果只是调整统计方法，不需要重新跑 PaddleNLP；如果只想检查 PaddleNLP 切词，也不会混入后面的关系统计逻辑。

## 编译检查

```bash
conda run -n nlpEnv python -m py_compile src/paddle_nlp_tokenize.py src/relation_statistics.py src/relation.py
```

## 第一步：生成 PaddleNLP token CSV

为了避免 Mac 处理时占用过高，默认限制 CPU 线程、批量大小，并在每个 batch 后稍微暂停。

```bash
conda run -n nlpEnv python src/paddle_nlp_tokenize.py \
  --device cpu \
  --cpu-threads 1 \
  --batch-size 8 \
  --sleep-seconds 0.05 \
  --output-dir data/processed/paddle_nlp_tokens
```

这一步输出：

```text
data/processed/paddle_nlp_tokens/
```

主要文件：

| 文件 | 说明 |
| --- | --- |
| `paddle_pos_tokens.csv` | PaddleNLP 输出的 token 级结果，每一行是一个词，包含词性、位置、前后词等信息。 |
| `sentence_records.csv` | 从原始 txt 中切出来的句子记录。 |
| `tokenize_overview.csv` | token 步骤总览统计。 |

## 第二步：关系候选统计

```bash
conda run -n nlpEnv python src/relation_statistics.py \
  --token-csv data/processed/paddle_nlp_tokens/paddle_pos_tokens.csv \
  --output-dir data/processed/relation_statistics
```

这一步不再调用 PaddleNLP，只读取上一步生成的 `paddle_pos_tokens.csv`。

终端默认只预览前 20 行关系候选，完整结果在 CSV 文件中。如果想让 PyCharm 终端打印全部 195 个关系候选，可以运行：

```bash
conda run -n nlpEnv python src/relation_statistics.py \
  --token-csv data/processed/paddle_nlp_tokens/paddle_pos_tokens.csv \
  --output-dir data/processed/relation_statistics \
  --preview-rows 195
```

默认设置下，统计脚本不会提前筛掉低频动词，而是把 PaddleNLP 标出来的所有动词都纳入统计。也就是说：

- `verb_tokens.csv` 保存所有动词 token 明细。
- `verb_frequency.csv` 保存所有不同动词的频次统计。
- `relation_type_candidates.csv` 保存所有不同动词的统计排序结果。
- 低频噪声词也会保留，后面再人工或规则判断它是不是有效关系。

## 第三步：关系类别归档

```bash
conda run --no-capture-output -n nlpEnv python src/relation.py
```

这一步读取：

- `data/processed/relation_statistics/verb_frequency.csv`
- `data/processed/relation_statistics/relation_type_candidates.csv`
- `data/processed/ner_rule_uie/sentence_pattern_summary.csv`

输出目录：

```text
data/processed/relation_schema/
```

主要文件：

| 文件 | 说明 |
| --- | --- |
| `relation_type_archive.csv` | 最终关系类别表，下一步 RE 优先读取这个文件。 |
| `relation_trigger_words.csv` | 每类关系对应的触发词，以及这些触发词是否在 Paddle 动词统计中命中。 |
| `relation_candidate_review.csv` | 对 195 个统计候选的复核归类，区分 core/event/auxiliary/exclude/noise。 |
| `relation_schema.json` | 和 CSV 内容一致的 JSON 版本，方便程序读取。 |
| `relation_schema_archive.md` | 关系类别归档说明。 |
| `relation_schema_overview.csv` | 归档总览统计。 |

## 当前运行结果

本次全量处理了 `data/source/mca_changes/` 下的 27 个年度文本文件。

| 指标 | 数值 |
| --- | ---: |
| 句子数 | 1771 |
| token 数 | 24924 |
| 动词/动词性 token 数 | 3744 |
| 不同动词数 | 195 |
| 地名类 token 数 | 8107 |
| 关系候选数 | 195 |

## 统计输出文件

输出目录：

```text
data/processed/relation_statistics/
```

主要文件：

| 文件 | 说明 |
| --- | --- |
| `verb_frequency.csv` | 所有不同动词的频次统计，后续判断关系类型时优先看这个表。 |
| `relation_type_candidates.csv` | 所有不同动词的关系候选排序表，不代表已经最终筛选。 |
| `relation_statistics_overview.csv` | 本次关系统计的总览统计。 |

如需 token 明细、句子词性统计、动词上下文和共现表，可运行
`src/relation_statistics.py --debug-output` 生成调试输出。

## 当前关系候选观察

从统计结果看，最核心的候选关系集中在：

| 候选关系 | 出现次数 | 说明 |
| --- | ---: | --- |
| 设立 | 731 | 常用于“设立地级市/县/区”。 |
| 撤销 | 478 | 常用于“撤销地区/县级市/行政区”。 |
| 划归 | 184 | 常用于行政区域归属调整。 |
| 调整 | 189 | 常用于行政区划调整说明。 |
| 更名 | 85 | 常用于地名或行政区名称变更。 |
| 直辖 | 77 | 常用于“由某省/自治区直辖”。 |
| 代管 | 50 | 常用于“由某市代管”。 |
| 迁至 | 42 | 常用于政府驻地迁移。 |
| 管辖 | 252 | 常用于管辖关系描述。 |
| 辖 | 131 | 常用于下辖区域描述。 |

也会出现一些辅助性或噪声词，例如 `为`、`驻`、`至`、`公布`、`执行`，甚至 `驻体育南大街` 这类由 PaddleNLP 切词导致的低频噪声。它们会被保留下来，因为当前阶段的目标是完整统计，而不是提前筛选最终关系：

- `为` 常出现在“行政区域为……”结构中。
- `驻` 常出现在“人民政府驻……”结构中。
- `至` 和 `迁至` 与驻地迁移有关。
- `公布` 和 `执行` 多与公告说明有关，不属于核心行政区划变更关系。
- 像 `驻体育南大街` 这样的词说明 PaddleNLP 有时会把“驻 + 地址”连成一个动词 token，后续 NER/RE 阶段再处理。

## 已归档关系类别

当前归档为 13 类，其中 11 类作为核心 RE 关系，1 类作为事件路由，1 类作为辅助范围信息。

| 关系类型 ID | 名称 | 用途 |
| --- | --- | --- |
| `REVOKE_ADMIN` | 撤销建制 | 核心关系 |
| `ESTABLISH_ADMIN` | 设立建制 | 核心关系 |
| `RENAME_ADMIN` | 名称变更 | 核心关系 |
| `MERGE_ADMIN` | 合并建制 | 核心关系 |
| `TRANSFER_ADMIN` | 区域划转 | 核心关系 |
| `JURISDICTION_ADMIN` | 管辖隶属 | 核心关系 |
| `DIRECT_ADMIN` | 省级直辖 | 核心关系 |
| `ENTRUST_ADMIN` | 委托代管 | 核心关系 |
| `GOV_RESIDENCE` | 政府驻地 | 核心关系 |
| `RESIDENCE_TRANSFER` | 驻地迁移 | 核心关系 |
| `AREA_INHERITANCE` | 行政区域承继 | 核心关系 |
| `ADJUSTMENT_EVENT` | 行政区划调整 | 事件路由 |
| `SCOPE_CONSTRAINT` | 范围包含排除 | 辅助信息 |

注意：`为`、`驻`、`至` 这类词不能无条件作为关系。比如 `为` 只有落在“以...行政区域为...行政区域”句式中才归入 `AREA_INHERITANCE`；`至` 通常只是 `迁至` 的切词碎片。
