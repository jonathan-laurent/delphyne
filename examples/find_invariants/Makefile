.PHONY: clean-ignored clean
.PHONY: test full-test test-exp execute-commands FORCE
.PHONY: delete-test-exp delete-command-cache delete-full-test-cache

TO_CLEAN := \
	-name '__pycache__' -o \
	-name '.pytest_cache' -o \
	-name '.ruff_cache' -o \
	-name '.DS_Store'


clean-ignored:
	find . \( $(TO_CLEAN) \) -exec rm -rf {} +

clean: clean-ignored


# Test demonstrations, in a way that does not require Why3 to be installed.
test:
	delphyne check abduct_and_branch.demo.yaml --ensure-no-error
# Other demos contain implicit answers...

full-test: execute-commands test-exp test
	pytest tests

test-exp:
	python experiments/test_abduction_experiment.py run --max_workers=2
	@echo "\n"
	python experiments/test_baseline_experiment.py run --max_workers=1
	@echo "\n"
	python experiments/test_abduction_experiment.py replay
	@echo "\n"
	python experiments/test_baseline_experiment.py replay
	@echo "\n"
	python experiments/test_abduction_experiment.py force-summary --add-timing
	@echo "\n"


# delphyne run commands/test_abduct_branch.exec.yaml --cache --no-output --ensure-no-erro
CMD_FILES := $(filter-out commands/test_abduct_branch_difficult.exec.yaml,$(wildcard commands/*.exec.yaml))

execute-commands: $(CMD_FILES)

commands/%.exec.yaml: FORCE
	delphyne run $@ --cache --filter=[values] --no-header --ensure-no-error
	@echo "\n"

FORCE:


delete-test-exp:
	rm -rf experiments/test-output

delete-command-cache:
	rm -rf commands/cache

delete-full-test-cache: delete-test-exp delete-command-cache