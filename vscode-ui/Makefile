.PHONY: clean install extension check

clean:
	rm -rf out
	rm -f *.vsix
	rm -rf .vscode-test

full-clean: clean
	rm -rf node_modules

check:
	npx tsc --noEmit

install:
	npm install

extension:
	npx vsce package