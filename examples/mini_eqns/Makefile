.PHONY: clean full-test

clean:
	make -f ../../Makefile clean-ignored

full-test:
	delphyne run commands/test_baseline_simple.exec.yaml --cache --filter=[values] --no-header --ensure-no-error