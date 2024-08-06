find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$|/\.egg-info$|/\.pytest_cache$|/_build$|/\.DS_Store$)" | xargs rm -rf
rm -rf build
rm -rf tests/cache
(cd examples/libraries/why3py ; sh clean.sh)
(cd examples/why3 ; sh clean.sh)