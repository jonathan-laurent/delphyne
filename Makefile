.PHONY: install-dev-deps clean clean-ignored test full-clean all examples schemas stubs install doc-logo cloc deploy-doc

TO_CLEAN := \
	-name '__pycache__' -o \
	-name '*.pyc' -o \
	-name '*.pyo' -o \
	-name '*.egg-info' -o \
	-name '.pytest_cache' -o \
	-name '_build' -o \
	-name '.ruff_cache' -o \
	-name '.DS_Store'

SCHEMAS_FOLDER := vscode-ui/resources
STUBS_FOLDER := vscode-ui/src/stubs


install:
	pip install -e .


install-dev-deps:
	pip install pyright ruff
	pip install mkdocs mkdocstrings[python] mkdocs-autolinks-plugin mkdocs-material mkdocs-glightbox


test:
	pytest tests
	make -C examples/find_invariants test


clean-ignored:
	find . \( $(TO_CLEAN) \) -exec rm -rf {} +


clean: clean-ignored
	rm -rf build
	rm -rf site
	rm -rf tests/cache
	rm -rf tests/cmd_out
	make -C vscode-ui clean
	make -C examples/libraries/why3py clean
	make -C examples/why3 clean
	make -C examples/mini_eqns clean


full-clean: clean
	make -C examples/libraries/why3py full-clean
	make -C examples/why3 full-clean


# Generate the demo file schema.
# This should only be executed after a change was made to the `Demo` type.
schemas:
	mkdir -p $(SCHEMAS_FOLDER)
	python -m delphyne.server.generate_schemas demo_file > \
	    $(SCHEMAS_FOLDER)/demo-schema.json


# Generate stubs by using GPT-4 to translate Python types into TypeScript.
# This should only be executed after a change is made to
# the `Demo` or `DemoFeedback` types
stubs:
	python -m delphyne.server.generate_stubs demos > $(STUBS_FOLDER)/demos.ts
	python -m delphyne.server.generate_stubs feedback > $(STUBS_FOLDER)/feedback.ts


all: install
	make -C vscode-ui install


examples: all
	make -C examples/libraries/why3py install


LOGOS_DIR := docs/assets/logos
BLACK_LOGOS := $(wildcard $(LOGOS_DIR)/black/*.png)
WHITE_LOGOS := $(subst /black/,/white/,$(BLACK_LOGOS))
$(LOGOS_DIR)/white/%.png: $(LOGOS_DIR)/black/%.png
	convert $< -fill black -colorize 100% -channel RGB -negate +channel $@
doc-logo: $(WHITE_LOGOS)


deploy-doc:
	mkdocs gh-deploy --force


# Count the number of lines of code
cloc:
	cloc . --exclude-dir=node_modules,out,.vscode-test --include-lang=python,typescript
