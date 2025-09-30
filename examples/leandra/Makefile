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


# Test demonstrations, in a way that does not require lean-interact to be installed.
test:
	@echo "TODO"

full-test: execute-commands test-exp test
	pytest tests

test-exp:
	@echo "TODO"


CMD_FILES := $(wildcard commands/*.exec.yaml)

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