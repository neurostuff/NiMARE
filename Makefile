.PHONY: all lint

all_tests: lint unittest performancetest

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  lint							to run flake8 on all Python files"
	@echo "  unittest						to run unit tests on nimare"
	@echo "  test_performance_estimators	to run performance tests on meta estimators"
	@echo "  test_performance_correctors	to run performance tests on correctors"
	@echo "  test_performance_smoke			to run performance smoke tests"

lint:
	@flake8 nimare

unittest:
	@py.test -m "not performance_estimators and not performance_correctors and not performance_smoke" --cov-append --cov-report term-missing --cov=nimare nimare

test_performance_estimators:
	@py.test -m "performance_estimators" --cov-append --cov-report term-missing --cov=nimare nimare

test_performance_correctors:
	@py.test -m "performance_correctors" --cov-append --cov-report term-missing --cov=nimare nimare

test_performance_smoke:
	@py.test -m "performance_smoke" --cov-append --cov-report term-missing --cov=nimare nimare
