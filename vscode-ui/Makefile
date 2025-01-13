.PHONY: clean install extension

clean:
	rm -rf out
	rm -f *.vsix
	rm -rf .vscode-test

full-clean: clean
	rm -rf node_modules

install:
	npm install

extension:
	npx vsce package