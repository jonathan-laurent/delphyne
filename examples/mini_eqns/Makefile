.PHONY: clean full-test delete-full-test-cache

clean:
	make -f ../../Makefile clean-ignored

full-test:
	delphyne run commands/test_baseline_simple.exec.yaml --cache --filter=[values] --no-header --ensure-no-error

delete-full-test-cache:
	rm -rf commands/cache