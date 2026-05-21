.PHONY: install install-dev install-attic test lint clean bundle docker-build hosted-demo help

help:
	@echo "Digital Registrar — make targets"
	@echo "  install         — install just the core (registrar-pipeline, registrar-schemas, registrar-eval)"
	@echo "  install-dev     — editable install of core + 3 apps + dev tools"
	@echo "  install-attic   — install the research scaffolding (benchmarks, ablations, baselines, obfuscator)"
	@echo "  test            — run pytest across core + app test suites"
	@echo "  lint            — ruff check"
	@echo "  bundle          — build PyInstaller bundles for the 3 apps"
	@echo "  docker-build    — build Docker images for the 3 apps"
	@echo "  clean           — remove caches / build artifacts"

# Production install (end users): just the core package
install:
	pip install vendor/tnmhelper-0.1.0-py3-none-any.whl
	pip install .

# Dev install (cloners): editable mode for everything + dev tooling
install-dev:
	pip install vendor/tnmhelper-0.1.0-py3-none-any.whl
	pip install -e ".[dev]"
	pip install -e apps/infer-gui
	pip install -e apps/schema-editor
	pip install -e apps/annotator

# Optional: install the research stuff
install-attic:
	pip install -e attic/

test:
	pytest -q tests apps/infer-gui/tests apps/schema-editor/tests apps/annotator/tests

lint:
	ruff check src apps tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build dist .pytest_cache .ruff_cache .mypy_cache

# Native bundles via PyInstaller (must be run on each target OS)
bundle:
	cd packaging && pyinstaller pyinstaller/infer_gui.spec
	cd packaging && pyinstaller pyinstaller/schema_editor.spec
	cd packaging && pyinstaller pyinstaller/annotator.spec

# Docker images
docker-build:
	docker build -f packaging/docker/infer-gui.Dockerfile -t digitalregistrar/gui .
	docker build -f packaging/docker/schema-editor.Dockerfile -t digitalregistrar/schema-editor .
	docker build -f packaging/docker/annotator.Dockerfile -t digitalregistrar/annotator .
