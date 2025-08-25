.PHONY: full-test execute-commands FORCE

CMD_FILES := $(wildcard *.exec.yaml)

FORCE:

# Execute all command files in the directory.
execute-commands: $(CMD_FILES)

%.exec.yaml: FORCE
	delphyne run $@ --cache --filter=[values] --no-header --ensure-no-error
	@echo "\n"

full-test: execute-commands