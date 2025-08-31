# Strategy Language

Following the previous [Overview](./overview.md) chapter, we now provide more details on Delphyne's _strategy language_. We provide a quick overview of the most useful techniques and concepts and refer to the [Reference](../reference/strategies/trees.md) for more details and explanations (follow the hypertext links).

## Strategies and Modularity

A _strategy_ is a program with unresolved choice points, which can be reified into a search tree. Delphyne allows writing strategies as Python [generators](https://wiki.python.org/moin/Generators) that yield internal tree nodes and receive associated actions in return. The [`Strategy`][delphyne.Strategy] type has three type parameters, corresponding to the strategy's _signature_ (i.e., the type of nodes it can emit), its associated _inner policy type_ and its _return type_. Strategy functions are typically defined via the [`strategy`][delphyne.strategy] decorator, which creates functions that return [`StrategyInstance`][delphyne.StrategyInstance] values, wrapping the underlying generator while adding some metadata and convenience methods (e.g., [`using`][delphyne.StrategyInstance.using]).

```py
@strategy  # Example from the previous chapter
def find_param_value(
    expr: str) -> Strategy[Branch | Fail, FindParamValueIP, int]: ...
```

Branching nodes can be yielded via the [`branch`][delphyne.branch] function:
<!-- , whose type is worth examining (we ignore some optional arguments): -->

```py
def branch[P, T](cands: Opaque[P, T]) -> Strategy[Branch, P, T]: ...
```

Crucially, `branch` does not take a query as an argument but an **opaque space** ([`Opaque`][delphyne.Opaque]). An opaque space can be seen as a mapping from the ambient inner policy to an iterator of values (or, more precisely, a [search stream](./policies.md)). Opaque spaces can be produced from queries or strategy instances, by providing a mapping from the ambient inner policy to a prompting policy or a policy respectively:

```py
class Query[T]:
    def using[Pout](self,
        get_policy: Callable[[Pout], PromptingPolicy]) -> Opaque[Pout, T]: ...

class StrategyInstance[N: Node, P, T]:
    def using(self,
        get_policy: Callable[[Pout], Policy[N, P]]) -> Opaque[Pout, T]: ...
```

Importantly, search policies such as [`dfs`][delphyne.dfs] are unaware of whether an opaque space originates from a query or a strategy. This guarantees modularity and allows queries to be transparently refined into dedicated strategies whenever more guidance is desired. Opaque spaces also allow queries with different signatures (i.e., yielding different kinds of tree nodes) to be composed, while being associated independent search policies.

!!! info "Strategy Inlining"
    It is also possible to _inline_ a strategy call within another strategy, provided that both strategies share the same signature and inner policy type. This can be done using the [`inline`][delphyne.StrategyInstance.inline] method, which _unwraps_ a [`StrategyInstance`][delphyne.StrategyInstance] value and gives access to the underlying generator.

For an example of a strategy that branches over results of a sub-strategy, see the `prove_program_via_abduction_and_branching` strategy from the `find_invariants` example. In particular, see how doing so results in nested inner policy records.

??? info "Source for `examples/find_invariants/abduct_and_branch.py`"
    ```python
    --8<-- "examples/find_invariants/abduct_and_branch.py"
    ```

## Queries

## Trees and Reification

## Adding new Effects




<!--
Strategies:
    Coroutines, branch, using, opaque spaces composition, ...
Queries:
    Parsers, generic parsers, 
Trees and Reification:
    Tree datastructure, warnings about purity, implementation details...
Trees:
    Adding effects.
-->