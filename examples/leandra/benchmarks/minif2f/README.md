# MiniF2F Benchmark Suite

MiniF2F benchmark suite, adapted from:
https://github.com/yangky11/miniF2F-lean4

## Build Instructions

To build the project, run the following commands:

```sh
lake exe cache get
lake build
```

This may take a while since it requires downloading the Mathlib cache.

To update the Lean version:

- Remove `.lake` and `lake-manifest.json` (`make full-clean`)
- Update `lean-toolchain` with the correct version
- Update `lakefile.toml` to set the matching Mathlib version number
- Rebuild the project