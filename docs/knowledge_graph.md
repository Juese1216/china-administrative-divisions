# 知识图谱

## 当前定位

知识图谱包含静态图谱和动态图谱：

1. 静态图谱：由行政区划编码表生成省、市、县、乡镇街道的层级关系。
2. 动态图谱：由 RE 输出生成旧区划、新区划和变更事件关系，边上保存年份和类型。

## 运行命令

```bash
conda run --no-capture-output -n nlpEnv python src/graph.py
```

## 输入文件

| 文件 | 作用 |
| --- | --- |
| `data/processed/relation_extraction/records.csv` | 记录级结果。 |
| `data/processed/relation_extraction/dynamic_triples.csv` | 动态关系三元组。 |
| `data/processed/relation_extraction/static_admin_nodes.csv` | 静态区划节点。 |
| `data/processed/relation_extraction/static_affiliation_triples.csv` | 静态隶属关系。 |

## 输出文件

输出目录：

```text
data/processed/knowledge_graph/
```

| 文件 | 作用 |
| --- | --- |
| `neo4j_nodes.csv` | Neo4j 节点导入表。 |
| `neo4j_edges.csv` | Neo4j 边导入表。 |
| `neo4j_import.cypher` | Neo4j 导入脚本和查询示例。 |
| `static_tree.json` | 静态区划树，可用于交互式树图。 |
| `timeline.json` | 年度变更时间轴，可用于时间轴动画。 |
| `overview.csv` | 图谱总览统计。 |

## 当前结果

| 指标 | 数值 |
| --- | ---: |
| 节点数 | 46721 |
| 边数 | 54793 |
| 静态区划节点数 | 42877 |
| 静态隶属边数 | 42876 |
| 动态关系边数 | 5445 |
| 记录数 | 511 |

## Neo4j 导入

1. 把 `neo4j_nodes.csv`、`neo4j_edges.csv`、`neo4j_import.cypher` 放到 Neo4j 的 `import` 目录。
2. 在 Neo4j Browser 或 `cypher-shell` 中执行 `neo4j_import.cypher`。
3. 动态关系使用 `DYNAMIC_RELATION` 边，静态隶属使用 `BELONGS_TO` 边。

示例查询：

```cypher
MATCH (a:KGNode)-[r:DYNAMIC_RELATION]->(b:KGNode)
WHERE r.year <= 2010
RETURN a.name, r.relation, b.name, r.year, r.type_label
ORDER BY r.year;
```
