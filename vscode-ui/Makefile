.PHONY: clean install extension

clean:
	rm -rf node_modules out
	rm -f *.vsix
	rm -rf .vscode-test

install:
	npm install

extension:
	npm install -g vsce
	vsce package