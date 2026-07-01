# NER：正则 + 词表 + Paddle UIE 增强

## 当前定位

NER 的基础层只负责识别文本里的实体，不单独判断旧名、新名和完整关系。

课程文档里写的“每条变更记录对应的变更前实体、变更后实体、变更类型”，在本项目中作为 NER 与 RE 的联合输出生成：基础 NER 先抽区划名、机构、地址等实体，RE 再结合“撤销、设立、更名、划归”等触发词和句式，把实体整理成记录级的变更前实体、变更后实体和变更类型。

前面的 PaddleNLP 切词和动词频次统计，只用于发现“撤销、设立、划归、更名、迁至”等关系类别；这些统计结果不再用于 NER。

当前 NER 主线改为：

1. 直接读取 `data/source/mca_changes/*.txt`，解析年度、行号、条目编号和句子。
2. 行政区划编码表词表精确匹配。
3. 正则表达式补充历史地名、政府机构、驻地地址。
4. PaddleNLP UIE 只作为可选增强，不在 Mac 上默认运行。
5. 用 pandas 汇总并输出实体表、覆盖表、句式分析表和审计表。

## 输入文件

| 文件 | 作用 |
| --- | --- |
| `data/source/admin_codes/行政区划编码表_20260427.xlsx` | 行政区划代码词表，默认读取“省市区”和“乡镇街道”两张表。 |
| `data/source/mca_changes/*.txt` | 民政部年度行政区划变更原文。NER 默认直接读取这些 txt。 |

兼容模式仍然保留。如果需要读取旧的句子 CSV，可以运行：

```bash
conda run -n nlpEnv python src/ner.py --input-mode sentence-csv
```

## 输出目录

默认输出到：

```text
data/processed/ner_rule_uie/
```

主要文件：

| 文件 | 说明 |
| --- | --- |
| `source_sentence_records.csv` | 从原始 txt 解析出的句子表。 |
| `sentence_pattern_summary.csv` | 每句是否包含撤销、设立、更名、划归、驻地迁移等线索。 |
| `entities.csv` | 最终实体明细表，每一行是一个实体。 |
| `entity_frequency.csv` | 实体出现频次统计。 |
| `ner_overview.csv` | 本次 NER 总览。 |

如需年度覆盖、零实体句、未匹配地名和 UIE 原始输出，可运行
`src/ner.py --debug-output` 生成调试表。

## 与 RE 的联合输出

基础 NER 输出是 `entities.csv` 和 `sentence_pattern_summary.csv`。记录级的旧名、新名、关系标签需要结合触发词和句式判断，因此由 `src/relation.py` 在 RE 阶段生成：

```text
data/processed/relation_extraction/ner_entities_by_record.csv
```

该文件包含：

| 字段 | 含义 |
| --- | --- |
| `record_id` | 变更记录编号。 |
| `record_text` | 变更记录原文。 |
| `变更前实体` | 旧区划或被调整对象。 |
| `变更后实体` | 新区划或调整目标。 |
| `变更类型` | 记录包含的 13 类关系标签集合，例如撤销建制、设立建制、区域划转、政府驻地等。 |
| `变更动作关键词` | 从文本和细粒度关系中整理出的动作词。 |

## 实体类型

| 类型 | 含义 | 示例 |
| --- | --- | --- |
| `ADMIN_AREA` | 行政区划或行政地名 | `云南省`、`畹町市`、`科尔沁区` |
| `GOV_ORG` | 政府机构 | `市人民政府`、`盘县人民政府` |
| `ADDRESS` | 政府驻地或道路门牌 | `科尔沁大街102号` |

## 运行命令

默认安全运行，只跑正则 + 词表，不启用全量 UIE：

```bash
conda run -n nlpEnv python src/ner.py
```

UIE 在 Mac CPU 上可能调用多个核心，所以不建议直接全量运行。
本项目当前 Mac 本地流程不跑 UIE，只保留 `--enable-uie` 开关，等有 Nvidia GPU 环境时再使用。
如果只是想先验证 UIE 增强入口，可以先跑前 3 条：

```bash
conda run -n nlpEnv python src/ner.py --enable-uie --max-sentences 3 --uie-batch-size 2
```

如果之后拿到 Nvidia 显卡机器，再全量启用 UIE：

```bash
conda run -n nlpEnv python src/ner.py --enable-uie --uie-device gpu
```

如果只想使用省市区词表，不加载乡镇街道：

```bash
conda run -n nlpEnv python src/ner.py --no-township
```

## 三种方法各自做什么

| 方法 | 输出 method | 作用 |
| --- | --- | --- |
| 行政区划词表精确匹配 | `lexicon_exact` | 最高置信度，能直接回填行政区划代码。 |
| 正则行政地名 | `regex_admin_area` | 补当前编码表里没有的历史地名。 |
| 正则政府机构 | `regex_government_org` | 抽取 `人民政府`、`民政部`、`国务院` 等机构。 |
| 正则地址 | `regex_address` | 抽取政府驻地、道路门牌。 |
| Paddle UIE | `paddle_uie` | 作为增强来源，补充规则漏掉的实体。 |

## 结果审计

生成 NER CSV 后，运行审计脚本：

```bash
conda run -n nlpEnv python src/ner_audit.py \
  --input-dir data/processed/ner_rule_uie \
  --output-dir data/processed/ner_rule_uie_audit
```

审计输出目录：

```text
data/processed/ner_rule_uie_audit/
```

主要文件：

| 文件 | 说明 |
| --- | --- |
| `ner_audit_summary.csv` | 审计总览，包括实体数、可疑数、错误风险数、复核数。 |
| `suspicious_entities.csv` | 需要复核的实体明细，包含 `severity` 和 `audit_reasons`。 |
| `high_frequency_unmatched_admin.csv` | 高频但没有匹配到当前编码表的历史地名。 |
| `method_samples.csv` | 各抽取方法的样例，便于人工看效果。 |
| `sentence_entity_count_outliers.csv` | 实体数量特别多的句子，便于检查是否过度抽取。 |
| `ambiguous_admin_entities.csv` | 同名多码实体，例如乡镇重名。 |

当前已人工确认的复核规则：

- `市人民政府`、`区人民政府`、`县人民政府` 保留为政府机构泛称，不再作为审计问题。
- `黔江土家族苗族自治县`、`丽江纳西族自治县`、`普洱哈尼族彝族自治县` 保留为历史行政区划旧名，不再作为审计问题。

## 为什么不再用机器学习 NER

之前的人工标注 + BiGRU 训练流程已经归档，当前课程路线按“正则 + 词表 + Paddle UIE 增强”执行。这样更贴近课程要求，也更容易解释每一步为什么能抽到实体。
