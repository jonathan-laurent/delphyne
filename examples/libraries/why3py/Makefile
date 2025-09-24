.PHONY: clean check full-clean install uninstall

clean:
	make -f ../../../Makefile clean-ignored

check:
	pyright

test:
	pytest tests

full-clean: clean
	rm -f src/why3py/bin/core.so
	rm -rf .eggs

install:
	pip install -e .

uninstall:
	pip uninstall why3py