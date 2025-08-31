# Demonstration Language

Delphyne includes a demonstration language for writing and maintaining few-shot prompting examples, in the form of coherent scenarios of navigating search trees. The demonstration language is amenable to a test-driven development workflow. This workflow is supported by a dedicated VSCode extension, which is described in the [next chapter](./extension.md).

## Demonstration Files

Demonstrations can be written in demonstration files with a `.demo.yaml` extension. A demonstration file features a list of demonstrations ([`Demo`][delphyne.core.demos.Demo]). Each demonstration can be separately evaluated. Many short examples can be found in the demonstation file from the Delphyne's test suite:


??? info "Source for `tests/example_strategies.demo.yaml`"

    ```yaml
    --8<-- "tests/example_strategies.demo.yaml"
    ```

!!! tip "On Reading Demonstration Files"
    Demonstration files are much easier to read and understand using Delphyne's [VSCode extension](./extension.md). [Standard shortcuts](https://code.visualstudio.com/docs/editing/codebasics#_folding) can be used to fold and unfold sections. The additional ++cmd+d+cmd+k++ shortcut can be used to automatically fold all large sections. Demonstrations can be evaluated and the path followed by each test inspected in the extension's [Tree View](./extension.md#navigating-trees).


A demonstration is either a standalone _query demonstration_[^standalone-ex] or a _strategy demonstration_. A _query demonstration_ describes a query instance along with one or several associated answers. A _strategy demonstration_ bundles multiple query demonstrations with unit tests that describe tree navigation scenarios.

[^standalone-ex]: An example of a standalone query demonstration is `MakeSum_demo` in `tests/example_strategies.demo.yaml`.

!!! warning
    It is possible to specify few shot examples using one _standalone_ query demonstration per example _and nothing else_. However, doing so is not recommended. Indeed, such demonstrations are harder to write since tooling cannot be leveraged to generate query descriptions automatically. More importantly, they are harder to read and maintain because individual examples are presented without proper context. Strategy demonstrations allow grounding examples in concrete scenarios, while enforcing this relationship through unit tests.

Strategy demonstrations have the following shape:

```yaml
- demonstration: ...    # optional demonstration name
  strategy: ...         # name of a strategy function decorated with @strategy
  args: ...             # dictionary of arguments to pass to this strategy
  tests:
    - ...
    - ...
  queries:
    - query: ...       # Query name
      args: ...        # Query arguments
      answers:
        - label: ...   # Optional label (to be referenced in tests)
          example: ... # Whether to use as an example (optional boolean) 
          tags: ...    # Optional set of tags
          answer: |
            ...
        - ...
```

The Delphyne VSCode extension automatically checks the syntactic well-formedness of demonstrations (in addition of allowing their evaluation). For explanations on specific fields, see the [API Reference][delphyne.core.demos.StrategyDemo]. Tests are expressed using a custom DSL that we describe below.

## Demonstration Tests
    
??? info "Extract from `examples/find_invariants/abduct_and_branch.demo.yaml`"

    ```yaml
    - strategy: prove_program_via_abduction_and_branching
      args:
        prog: ... # (1)!
      tests:
        - run | success
        - run 'partial' | success
        # Demonstrating `EvaluateProofState`
        - at EvaluateProofState#1 'partial' | answer eval
        - at EvaluateProofState#2 'partial propose_same' | answer eval
        # Demonstrating `IsProposalNovel`
        - at iterate#1 | go cands | go next(next(nil){'partial'}[1]) | save second_attempt
        - load second_attempt | at IsProposalNovel 'blacklisted' | answer cands
        - load second_attempt | at IsProposalNovel 'not_blacklisted' | answer cands
      queries: ... # (2)!
    ```

    1. See details in original file.
    2. See details in original file.