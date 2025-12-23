.PHONY: lock install

# Directories with requirement specs
REQ_DIRS := requirements

lock:
	@set -e; \
	for d in $(REQ_DIRS); do \
		echo "Locking $$d ..."; \
		pip-compile $$d/requirements.in       -o $$d/requirements.txt --generate-hashes; \
	done

install:
	pip install -r requirements/requirements.txt