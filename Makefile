.PHONY: test workbench-export workbench-export-flat workbench-install workbench-build workbench-dev

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v

workbench-export:
	PYTHONPATH=src python3 scripts/export_workbench_scenario.py --memory hierarchical

workbench-export-flat:
	PYTHONPATH=src python3 scripts/export_workbench_scenario.py --memory flat

workbench-install:
	npm --prefix workbench-ui install

workbench-build: workbench-export
	npm --prefix workbench-ui run build

workbench-dev: workbench-export
	npm --prefix workbench-ui run dev
