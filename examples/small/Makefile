.PHONY: full-test execute-commands FORCE delete-full-test-cache

full-test: execute-commands

CMD_FILES := $(wildcard *.exec.yaml)

# Execute all command files in the directory.
execute-commands: $(CMD_FILES)

%.exec.yaml: FORCE
	delphyne run $@ --cache --filter=[values] --no-header --ensure-no-error
	@echo "\n"

FORCE:

delete-full-test-cache:
	rm -rf cache