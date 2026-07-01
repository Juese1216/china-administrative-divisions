PYTHON ?= python
PORT ?= 5000

.PHONY: help install install-dev smoke test tokenize relation-stats ner relation classify graph forecast rural-atlas webdb web clean

help:
	@echo "常用命令:"
	@echo "  make install         安装 requirements.txt 里的完整运行依赖"
	@echo "  make install-dev     以可编辑模式安装项目、开发依赖和 PaddleNLP 可选依赖"
	@echo "  make smoke           运行语法编译和工程结构检查"
	@echo "  make webdb           生成 Flask 仪表盘 SQLite 数据库"
	@echo "  make web PORT=5001   启动 Flask Web 仪表盘"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -e ".[dev,nlp]"

smoke:
	$(PYTHON) -m compileall -q src web scripts
	$(PYTHON) tests/test_project_structure.py

test:
	$(PYTHON) tests/test_project_structure.py

tokenize:
	$(PYTHON) src/paddle_nlp_tokenize.py

relation-stats:
	$(PYTHON) src/relation_statistics.py

ner:
	$(PYTHON) src/ner.py

relation:
	$(PYTHON) src/relation.py

classify:
	$(PYTHON) src/classify.py

graph:
	$(PYTHON) src/graph.py

forecast:
	$(PYTHON) src/forecast.py

rural-atlas:
	PYTHONPATH=src $(PYTHON) -m rural_atlas.cli quantity-atlas

webdb:
	$(PYTHON) web/webdb.py

web:
	PORT=$(PORT) $(PYTHON) web/web_app.py

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name ".DS_Store" -type f -delete
