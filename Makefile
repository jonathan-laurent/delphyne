.PHONY: clean clean-ignored test full-clean all examples schemas stubs install doc-logo

TO_CLEAN := \
	-name '__pycache__' -o \
	-name '*.pyc' -o \
	-name '*.pyo' -o \
	-name '*.egg-info' -o \
	-name '.pytest_cache' -o \
	-name '_build' -o \
	-name '.DS_Store'

SCHEMAS_FOLDER := vscode-ui/resources
STUBS_FOLDER := vscode-ui/src/stubs


install:
	pip install -e .


test:
	pytest tests


clean-ignored:
	find . \( $(TO_CLEAN) \) -exec rm -rf {} +


clean: clean-ignored
	rm -rf build
	rm -rf tests/cache
	make -C vscode-ui clean
	make -C examples/libraries/why3py clean
	make -C examples/why3 clean


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


doc-logo:
	convert docs/assets/logos/delphyne.png \
	    -channel RGB -negate +channel \
		docs/assets/logos/delphyne-white.png
