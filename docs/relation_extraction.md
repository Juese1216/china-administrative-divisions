# RE：关系抽取与三元组生成

## 当前定位

RE 主线输出两类三元组：

1. 动态变更三元组：从年度变更文本中解析旧区划、新区划、变更动作和生效年份。
2. 静态隶属三元组：从行政区划编码表中生成“区划 -> 隶属于 -> 上级区划”。

`src/relation.py` 已经把关系类别归档、细粒度句内关系抽取、记录级结果整理和静态隶属三元组生成整合到同一个顶层脚本中。

这里同时完成记录级 NER 摘要：基础 NER 负责抽出区划名、机构和地址，RE 负责结合触发词判断哪些是变更前实体、哪些是变更后实体，以及该记录属于哪类变更。因此 `ner_entities_by_record.csv` 是 NER 与 RE 的联合结果，不是单纯的实体明细表。

## 运行命令

```bash
conda run --no-capture-output -n nlpEnv python src/relation.py
```

## 输入文件

| 文件 | 作用 |
| --- | --- |
| `data/processed/ner_rule_uie/entities.csv` | NER 实体明细。 |
| `data/processed/ner_rule_uie/source_sentence_records.csv` | 年度文本切分后的句子表。 |
| `data/processed/relation_details/triples.csv` | 细粒度辅助关系。 |
| `data/source/admin_codes/行政区划编码表_20260427.xlsx` | 静态区划编码表。 |

## 主线输出

输出目录：

```text
data/processed/relation_extraction/
```

| 文件 | 作用 |
| --- | --- |
| `records.csv` | 记录级结果，每条记录包含文本、变更前实体、变更后实体、变更类型。 |
| `ner_entities_by_record.csv` | NER + RE 联合摘要表，包含变更前实体、变更后实体和变更类型。 |
| `dynamic_triples.csv` | 动态变更关系三元组，边属性包含年份和变更类型。 |
| `static_admin_nodes.csv` | 静态区划节点，来自行政区划编码表。 |
| `static_affiliation_triples.csv` | 静态隶属关系三元组。 |
| `overview.csv` | RE 总览统计。 |

## 当前结果

| 指标 | 数值 |
| --- | ---: |
| 记录数 | 511 |
| NER 实体数 | 7716 |
| 动态三元组数 | 5445 |
| 静态区划节点数 | 42877 |
| 静态隶属三元组数 | 42876 |
| 关系类型数 | 13 |

## 说明

- `records.csv` 是后续分类、图谱和网页展示的主表，`type_label` 为一条记录包含的 13 类关系标签集合。
- 主线 RE 已覆盖 1999-2021、2023-2026 年；2022 年没有源 txt，留给时序统计按 0 条补齐。
- `dynamic_triples.csv` 主要表达“旧区划 -> 变更为/划归/隶属于/直辖于 -> 新区划或目标区划”。
- 对街道、镇、乡、村等基层变更，主线 RE 增加了文本补充解析，会处理 `将...划归...管辖`、`X辖...街道/镇/乡`、`驻地由...迁至...` 等句式。
- 公告句如“（某省人民政府 2021 年 1 月 29 日公布）”只作为来源说明剔除，不再删除整条变更记录。
- `static_affiliation_triples.csv` 不从文本猜测，而是直接由行政区划编码表生成。
- `data/processed/relation_details/` 保留 13 类细粒度关系，用于追溯和补充图谱，不作为课程主线口径。
