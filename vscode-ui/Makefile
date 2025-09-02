.PHONY: clean install-deps extension check

# Install NPM dependencies.
# A Pyton installation is required because of `node-gyp`
install-deps:
	npm install

# Build the extension package, as a *.vsix archive
extension:
	npx vsce package

# Typecheck the whole extension codebase
check:
	npx tsc --noEmit

# Clean temporary files that are not costly to generate
clean:
	rm -rf out
	rm -f *.vsix
	rm -rf .vscode-test

# Clean all nonversioned files
full-clean: clean
	rm -rf node_modules