# The Demonstration Language

Delphyne includes a demonstration language for writing and maintaining few-shot prompting examples, in the form of coherent scenarios of navigating search trees. The demonstration language is amenable to a test-driven development workflow. This workflow is supported by a dedicated VSCode extension, which is described in the [next chapter](./extension.md).

## Demonstration Files

Demonstrations can be written in demonstration files with a `.demo.yaml` extension. A demonstration file features a list of demonstrations ([`Demo`][delphyne.core.demos.Demo]). Each demonstration can be separately evaluated. Many short examples can be found in the demonstration file from Delphyne's test suite:


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

The Delphyne VSCode extension automatically checks the syntactic well-formedness of demonstrations (in addition to allowing their evaluation). For explanations on specific fields, see the [API Reference][delphyne.core.demos.StrategyDemo]. Tests are expressed using a custom DSL that we describe below.

## Demonstration Tests
    
Evaluating a demonstration consists in evaluating all its tests in sequence. Each test describes a path through the tree, starting from the root. The Delphyne VSCode extension allows visualizing this path. A test can _succeed_, _fail_, or be _stuck_. A test is said to be _stuck_ if it cannot terminate due to a missing query answer. In this case  (and as demonstrated in the [Overview](./overview.md#adding-demonstrations)), the extension allows locating such a query and adding it to the demonstration.

Each test is composed of a sequence of instructions separated by `|`. The most common sequence by far is `run | success`, which we describe next.

### Walking through the Tree

Starting at the current node, the `run` instruction uses answers from the `queries` section to walk through the tree until either a leaf node is reached or an answer is missing (in which case the test is declared as stuck). Each node type (e.g. [`Branch`][delphyne.Branch]) defines a [navigation function][delphyne.Navigation] that describes how the node should be traversed.

??? info "Navigation Functions"

    A node's navigation function returns a generator that yields local spaces, receives corresponding elements and ultimately returns an action. This is best understood through examples:

    ??? example "Example: Navigation function for `Branch` nodes"
        ```python
        @dataclass(frozen=True)
        class Branch(dp.Node):
            cands: OpaqueSpace[Any, Any]

            @override
            def navigate(self):
                return (yield self.cands)
        ```

    ??? example "Example: Navigation function for `Join` nodes"
        ```python
        @dataclass(frozen=True)
        class Join(dp.Node):
            subs: Sequence[dp.EmbeddedTree[Any, Any, Any]]

            @override
            def navigate(self):
                ret: list[Any] = []
                for sub in self.subs:
                    ret.append((yield sub))
                return tuple(ret)
        ```

Whenever `run` needs to select an element from a space defined by a query, it looks for this query in the demonstration’s `queries` section and picks the _first_ provided answer. If no answer is found, it gets _stuck_ at the current node. When `run` encounters a space defined by a tree, it recursively navigates this tree. The `run` command stops when a leaf is reached. It is often composed with the `success` command, which ensures that the current node is a success leaf.

<!-- (In contrast to policies, the demonstration interpreter can break within opaque spaces.) -->

Nothing more than `run | success` is needed to demonstrate taking a direct path to a solution. The more advanced instructions we discuss next are useful to describe more complex scenarios.

### Advanced Tests

This section describes more advanced test instructions. In addition to the examples from the test suite, demonstrations with advanced tests are featured in the `find_invariants` example:

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

#### Exploring alternative paths with hints

The `run` function can be passed a sequence of answer labels as *hints*, specifying alternate paths through the tree. Whenever a query is encountered, it checks if an answer is available whose label matches the first provided hint. If so, this answer is used and the hint is consumed. For example, instruction `run 'foo bar'` can be interpreted as:

> Walk through the tree, using answer `foo` whenever applicable and then `bar`.[^unused-hint]

[^unused-hint]: A warning is issued if the `run` command reaches a leaf node while unused hints remain.

This design allows describing paths concisely, by only specifying the few places in which they differ from a default path. This works well for demonstrations, which typically describe *shallow* traces centered around a successful scenario, with side explorations (e.g., showing how a bad decision leads to a low value score, or demonstrating how redundant candidates can be removed at a particular step).

#### Stopping at particular nodes

The `at` instruction works like `run`, except that it takes as an additional argument a [node selector][delphyne.core.demos.NodeSelector] that specifies a node at which the walk must stop. The simplest form of node selector consists in a [tag][delphyne.Tag] to match. For example, instruction `at EvalProg 'wrong'` behaves similarly to `run 'wrong'`, except that it stops when encountering a node tagged with `EvalProg`. By default, all *spaces* are tagged with the name of the [associated query][delphyne.AbstractQuery.default_tags] or [strategy][delphyne.StrategyInstance.default_tags], and each node inherits the tags of its [primary space][delphyne.Node.primary_space] if it has one. Custom space tags can be added using the [`SpaceBuilder.tagged`][delphyne.SpaceBuilder.tagged] method[^tagged-ex]. Finally, the `#n` operator can be used to match the $n^{th}$ instance of a tag. For example, `at PickPositiveInteger#2` stops at the _second_ encountered node tagged with `PickPositiveInteger`[^at-op-ex].

[^tagged-ex]: See `tests/example_strategies.py:dual_number_generation` for an example.
[^at-op-ex]: See `tests/example_strategies.demo.yaml:test_generate_pairs`.

!!! warning
    Importantly, `at` can only stop *within* the same tree that it started in, and *not* inside a nested tree. In order to stop at a node tagged with `bar` _within_ a space tagged with `foo`, you can use `at foo/bar`. This design choice is mandated by *modularity*: individual strategies can be made responsible for setting unambiguous tags for nodes that they control, but cannot ensure the absence of clashing tags in *other* strategies.

#### Entering nested spaces

<!-- TODO: find a more canonical example -->

The `go` instruction allows entering a tree nested within the current node. For example, if the current node is a `Conjecture` node (defined in `tests/example_strategies.py`), `go cands` enters the tree that defines the `cands` space or errors if `cands` is defined by a query. This instruction can be shortened as `go`, since `cands` is the primary space of `Conjecture` nodes.

More interestingly, suppose the demonstration already explores two paths within `cands` that reach different success leaves and thus correspond to two different candidates. Each of these paths can be described through a sequence of *hints*: the first candidate is identified by `''` (i.e. default path) and the second by `'foo'` (i.e. use answer `'foo'` when appropriate). Then, instruction `go aggregate([cands{''}, cands{'foo'}])` can be used to enter the strategy tree comparing those two candidates. It can be shortened to `go aggregate(['', 'foo'])` since `cands` is a primary space.

In general, any element of a local space can be referenced via a (possibly empty) sequence of hints. For spaces defined by queries, at most one hint is expected that indicates which answer to use. For spaces defined by trees, a sequence of hints is expected that leads to a success leaf by calling `run` recursively.

The `answer` instruction is similar to `go`. It takes a space selector as an argument but expects to find a query instead of a tree when entering this space. It succeeds if the corresponding query is answered in the demonstration and fails otherwise.
