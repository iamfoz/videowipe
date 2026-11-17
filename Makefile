check:
	pytest

fact-baseline:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/eval_clean_detection.py input/detext_examples --manifest input/detext_examples/fact_baseline.json --mask-dir input/detext_examples/mask --output result/fact-baseline/report.json --artifact-dir result/fact-baseline/previews

fact-baseline-formal:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/eval_clean_detection.py input/detext_examples --manifest input/detext_examples/fact_baseline.json --mask-dir input/detext_examples/mask --output result/fact-baseline/report.json --artifact-dir result/fact-baseline/previews --require-clean-git

decision-baseline:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/eval_clean_detection.py input/detext_examples --decision-manifest input/detext_examples/decision_baseline.json --output result/decision-baseline/report.json

decision-baseline-formal:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python scripts/eval_clean_detection.py input/detext_examples --decision-manifest input/detext_examples/decision_baseline.json --output result/decision-baseline/report.json --require-clean-git

i18n-extract:
	python scripts/i18n_sync.py extract

i18n-report:
	python scripts/i18n_sync.py report

i18n-sync:
	python scripts/i18n_sync.py update

i18n-check:
	python scripts/i18n_sync.py check

LOCALE ?= en
i18n-translate:
	python scripts/i18n_sync.py translate --locale $(LOCALE)
