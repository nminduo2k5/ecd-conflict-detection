.PHONY: install test lint bench rq1 rq2 rq3 rq4 clean
install:
	pip install -e ".[dev]"
test:
	pytest -q
lint:
	ruff check src tests experiments benchmark
bench:
	ecd gen --n 1000 --out data/processed/synthetic_1k
	ecd gen --n 10000 --out data/processed/synthetic_10k --seed 1
rq1:
	cd experiments && python run_rq1_embedding.py
rq2:
	cd experiments && python run_rq2_pruning.py
rq3:
	cd experiments && python run_rq3_incremental.py
rq4:
	cd experiments && python run_rq4_scalability.py
clean:
	rm -rf results/* data/processed/* .pytest_cache
