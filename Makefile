.PHONY: install pyright test full-test
.PHONY: clean-ignored clean full-clean
.PHONY: delete-test-cache delete-full-test-cache
.PHONY: schemas demos-stub feedback-stub stubs doc-logo readme repomix
.PHONY: deploy-doc-release deploy-doc-dev prepare-release release
.PHONY: cloc count-doc-words


RELEASE_SCRIPT := python scripts/prepare_release.py

TO_CLEAN := \
	-name '__pycache__' -o \
	-name '*.pyc' -o \
	-name '*.pyo' -o \
	-name '*.egg-info' -o \
	-name '.pytest_cache' -o \
	-name '_build' -o \
	-name '.ruff_cache' -o \
	-name '.DS_Store' -o \
	-name 'repomix-output.xml'

SCHEMAS_FOLDER := vscode-ui/resources
STUBS_FOLDER := vscode-ui/src/stubs


# Install Delphyne (and its dependencies) in editable mode.
install:
	pip install -e ".[dev]"


# Perform typechecking for the whole codebase and examples.
pyright:
	@echo "Checking main project"
	pyright
	@echo "\n\nChecking find_invariants"
	pyright examples/find_invariants
	@echo "\n\nChecking mini_eqns"
	pyright examples/mini_eqns
	@echo "\n\nChecking small"
	pyright examples/small


# Run a quick, minimal test suite. These tests should not require additional
# dependencies on top of those specified in Delphyne's pyproject.toml.
test:
	pytest tests
	make -C examples/find_invariants test
	delphyne run tests/commands/run_make_sum_using_demo.exec.yaml --update


# Run a longer test suite. This might require additional dependencies, as
# specified by individual example projects.
full-test: test
	make -C examples/find_invariants full-test
	make -C examples/mini_eqns full-test
	make -C examples/small full-test


# Clean files ignored by git.
clean-ignored:
	find . \( $(TO_CLEAN) \) -exec rm -rf {} +


# Clean all files that are cheap to regenerate.
clean: clean-ignored
	rm -rf build
	rm -rf site
	rm -rf tests/cmd_out
	make -C vscode-ui clean
	make -C examples/libraries/why3py clean
	make -C examples/find_invariants clean
	make -C examples/mini_eqns clean


# Perform a complete cleaning.
full-clean: clean
	make -C examples/libraries/why3py full-clean


# Delete the request cache of the test suite. Using `make test` will regenerate
# the cache, although doing so can take time and require API keys for a number
# of LLM providers.
delete-test-cache:
	rm -rf tests/cache
	rm -rf tests/output


delete-full-test-cache:
	make -C examples/find_invariants/ delete-full-test-cache
	make -C examples/mini_eqns/ delete-full-test-cache
	make -C examples/small/ delete-full-test-cache


# Generate the demo file schema.
# This should only be executed after a change was made to the `Demo` type.
schemas:
	mkdir -p $(SCHEMAS_FOLDER)
	python -m delphyne.server.generate_schemas demo_file > \
	    $(SCHEMAS_FOLDER)/demo-schema.json
	python -m delphyne.server.generate_schemas config_file > \
	    $(SCHEMAS_FOLDER)/config-schema.json


# Generate stubs by using GPT-4 to translate Python types into TypeScript.
# This should only be executed after a change is made to
# the `Demo` or `DemoFeedback` types
stubs: demos-stub feedback-stub

demos-stub:
	python -m delphyne.server.generate_stubs demos > $(STUBS_FOLDER)/demos.ts

feedback-stub:
	python -m delphyne.server.generate_stubs feedback > $(STUBS_FOLDER)/feedback.ts


# Generate white logos from the black logos (for dark mode themes).
LOGOS_DIR := docs/assets/logos
BLACK_LOGOS := $(wildcard $(LOGOS_DIR)/black/*.png)
WHITE_LOGOS := $(subst /black/,/white/,$(BLACK_LOGOS))
GRAY_LOGOS := $(subst /black/,/gray/,$(BLACK_LOGOS))
$(LOGOS_DIR)/white/%.png: $(LOGOS_DIR)/black/%.png
	convert $< -fill black -colorize 100% -channel RGB -negate +channel $@
$(LOGOS_DIR)/gray/%.png: $(LOGOS_DIR)/black/%.png
	convert $< -fill '#666d77' -colorize 100% $@
doc-logo: $(WHITE_LOGOS) $(GRAY_LOGOS)
	cp $(LOGOS_DIR)/gray/mini.png vscode-ui/media/logo/delphyne.png


# Generate README.md from docs/index.md
readme:
	python scripts/generate_readme.py > README.md


# Folders and files to ignore by repomix.
# Use commas after each item except the last.
REPOMIX_IGNORE = \
	examples/find_invariants/experiments/analysis/,\
	examples/find_invariants/experiments/test-output/,\
	examples/find_invariants/benchmarks/,\
	examples/libraries/,\
	scripts/prepare_release.py
#
# Not excluded so far:
# tests/cache/
#
# Generate a single file summarizing the repo, to be passed to LLMs for context.
repomix:
	repomix --ignore "$(REPOMIX_IGNORE)"


# Build and deploy the documentation for the latest stable release.
# Warning: this should only be used if the documentation on the current commit
# is valid for the latest stable release.
deploy-doc-release:
	git fetch origin gh-pages
	mike deploy 0.11 latest --update-aliases --push


# Build and deploy the documentation for the dev version
deploy-doc-dev:
	git fetch origin gh-pages
	mike deploy dev --push


# Prepare a new release.
#
# To make a new release, follow the following steps:
#     1. Bump the version number in `pyproject.toml`
#     2. Run `make prepare-release`
#	  3. Check that the changes are ok using `git diff`
#     4. Commit the changes (with "Bump version" message).
#     5. Finalize and push the release using `make release`
prepare-release:
	${RELEASE_SCRIPT} prepare `${RELEASE_SCRIPT} current-version`
# 	@$(MAKE) full-test


# Finalize and push a release (see `prepare-release`).
release:
	@test -z "$$(git status --porcelain)" || (echo "Uncommitted changes found" && exit 1)
	@$(MAKE) deploy-doc-release
	git tag v`${RELEASE_SCRIPT} current-version`
	git push --tags


# Count the number of lines of code
cloc:
	cloc . --exclude-dir=node_modules,out,.vscode-test --include-lang=python,typescript


# Estimate the size of the documentation. A page is traditionally defined as 250
# words (double spaced) or 500 words (single spaced).
count-doc-words:
	@echo "Number of words in mkdocs website (excluding references):"
	find docs -name '*.md' -exec cat {} + | wc -w
	@echo "Number of words in Python docstrings:"
	rg --multiline --multiline-dotall '"""(.*?)"""' -o src | wc -w