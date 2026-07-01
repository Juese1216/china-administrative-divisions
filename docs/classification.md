# 分类任务

## 当前定位

分类任务负责把每条行政区划调整记录标注为 13 类细粒度关系类型。由于一条
记录可能同时包含撤销、设立、驻地、区域承继等多个关系，当前实现是
**多标签分类**，不是旧版的 5 类单标签粗分类。

实现包含两层：

1. 从 RE 输出的 `relation_type_ids` 生成 13 类关系弱监督标签。
2. 使用字符级 TF-IDF + One-vs-Rest 逻辑回归训练多标签文本分类器。
3. 默认把训练集记录对应的句子级关系样本并入训练集，用于增加小样本场景下的训练数据。

## 运行命令

```bash
conda run --no-capture-output -n nlpEnv python src/classify.py
```

## 输入文件

| 文件 | 作用 |
| --- | --- |
| `data/processed/relation_extraction/records.csv` | RE 主线记录表，包含每条记录的 `relation_type_ids`。 |

## 输出文件

输出目录：

```text
data/processed/classification/
```

| 文件 | 作用 |
| --- | --- |
| `rule_classification.csv` | 多标签弱监督分类结果；一条记录有多个关系类型时会展开为多行。 |
| `class_frequency.csv` | 13 类关系类型频次和平均置信度。 |
| `ml_classification_predictions.csv` | 多标签文本分类器对全量记录的预测。 |
| `ml_evaluation.csv` | 训练/测试样本数、Exact Match Accuracy、Micro-F1、Macro-F1 等指标。 |
| `ml_confusion_matrix.csv` | 每个关系类型的一对多混淆矩阵统计。 |
| `classification_overview.csv` | 分类任务总览。 |

调试时可运行 `src/classify.py --debug-output` 额外输出训练样本、特征权重和
sklearn 文本评估详情。

## 当前结果

| 指标 | 数值 |
| --- | ---: |
| 记录数 | 511 |
| 关系类型类别数 | 13 |
| 弱监督记录数 | 511 |
| 原始记录级标签分配数 | 2128 |
| 记录级训练样本数 | 407 |
| 句子级补充训练样本数 | 1355 |
| 总训练样本数 | 1762 |
| 测试样本数 | 104 |
| Exact Match Accuracy | 0.6731 |
| 标签准确率 | 0.9593 |
| Micro-F1 | 0.9354 |
| Macro-F1 | 0.7703 |
| 规则与模型完全一致率 | 0.8160 |

说明：多标签任务中，Exact Match Accuracy 要求一条记录的所有关系标签完全预测
一致，因此比单标签准确率更严格。标签准确率按 13 个标签位逐一计算，当前为
0.9593；结果展示时更适合同时给出标签准确率和 Micro-F1。句子级补充样本只来自
训练集记录，测试集仍按记录级样本评估。

## 13 类关系类型

| 类别 | 记录数 |
| --- | ---: |
| 政府驻地 | 439 |
| 设立建制 | 366 |
| 撤销建制 | 338 |
| 行政区域承继 | 292 |
| 行政区划调整事件 | 186 |
| 管辖隶属 | 116 |
| 区域划转 | 115 |
| 省级直辖 | 79 |
| 驻地迁移 | 67 |
| 名称变更 | 52 |
| 委托代管 | 43 |
| 范围包含排除 | 33 |
| 合并建制 | 2 |
