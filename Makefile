.PHONY: lock install run

REQ_DIRS := requirements

run:
	PYTHONPATH=. python3 experiments/run_experiment.py

lock:
	@set -e; \
	for d in $(REQ_DIRS); do \
		echo "Locking $$d ..."; \
		pip-compile $$d/requirements.in       -o $$d/requirements.txt --generate-hashes; \
	done

install:
	pip install -r requirements/requirements.txt