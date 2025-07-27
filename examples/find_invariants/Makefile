.PHONY: clean clean-ignored delete-exp test full-test replay-test-exp test-exp

TO_CLEAN := \
	-name '__pycache__' -o \
	-name '.pytest_cache' -o \
	-name '.ruff_cache' -o \
	-name '.DS_Store'

clean-ignored:
	find . \( $(TO_CLEAN) \) -exec rm -rf {} +

clean: clean-ignored

test-exp:
	python experiments/test_abduction_experiment.py run --max_workers=2
	python experiments/test_baseline_experiment.py run --max_workers=1

replay-test-exp:
	python experiments/test_abduction_experiment.py replay
	python experiments/test_baseline_experiment.py replay

delete-test-exp:
	rm -rf experiments/test-output

# Test demonstrations, in a way that does not require Why3 to be installed.
test:
	python -m delphyne check-demo --strategy-dirs=[.] --modules=[abduct_and_branch] --file=abduct_and_branch.demo.yaml --ensure-no-error
# Contains implicit answers...
# python -m delphyne check-demo --strategy-dirs=[.] --modules=[abduct_and_saturate] --file=abduct_and_saturate.demo.yaml --ensure-no-error

full-test: test test-exp replay-test-exp
	pytest tests