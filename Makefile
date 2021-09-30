.PHONY: all lint

all_tests: lint unittest performancetest

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  lint			to run flake8 on all Python files"
	@echo "  unittest		to run unit tests on nimare"
	@echo "  performancetest		to run performance tests"

lint:
	@flake8 nimare

unittest:
	@py.test -m "not performance and not performance_smoke" --cov-append --cov-report term-missing --cov=nimare nimare

performancetest:
	@py.test -m "performance" --cov-append --cov-report term-missing --cov=nimare nimare

performancesmoketest:
	@py.test -m "performance_smoke" --cov-append --cov-report term-missing --cov=nimare nimare
