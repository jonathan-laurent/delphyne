find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$|\.pytest_cache|_build)" | xargs rm -rf
rm -f src/why3py/bin/core.so