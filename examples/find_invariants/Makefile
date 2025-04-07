.PHONY: clean

clean:
	make -f ../../Makefile clean-ignored

test:
	python -m  delphyne check-demo --strategy-dirs=[.] --modules=[find_invariants] --file=find_invariants.demo.yaml --ensure-no-error