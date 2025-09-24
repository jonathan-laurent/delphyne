# A Python Interface to Why3

## Installation

Installing this library can be a bit tricky since it is implemented in OCaml and wrapped in Python. First, you must build the OCaml library. If you don't have ocaml installed, you should install the Opam package manager first: 

```
bash -c "sh <(curl -fsSL https://opam.ocaml.org/install.sh)"
opam init
```

We recommend creating an opam switch with compiler version 5.1.1 since more recent versions were not compatible with `metaquot` last time we checked:

```
opam switch create delphyne 5.1.1
```

Then, you should install the `python-libgen` dependency, which is not on Opam, along with all other dependencies:

```
opam pin add -y python-libgen https://github.com/jonathan-laurent/python-libgen.git
```

You can finally build the OCaml library by running the following within the `ocaml` subdirectory:

```sh
opam install . --deps-only
dune build
```

Once this is done, you can build and install the Python library by running the following at the root of the why3py repository:

```sh
pip install -e .
```

In particular, running this script should create a `src/why3py/bin.core.so` file. Finally, once this is done, you have to install and configure the Alt-Ergo prover used by `why3py`:

```
opam install alt-ergo
why3 config detect
```

If everything worked, you should be able to run the tests successfully:

```
pytest tests
```

You can also see all the generated obligations for an example by running:

```
pytest -k "prove" -rP --color=yes
```