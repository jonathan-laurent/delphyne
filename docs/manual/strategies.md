# The Strategy Language

Following the previous [Overview](./overview.md) chapter, we now provide more details on Delphyne's _strategy language_. We provide a quick overview of the most useful techniques and concepts and refer to the [Reference](../reference/strategies/trees.md) for more details and explanations (follow the hypertext links).

## Defining Strategies

A _strategy_ is a program with unresolved choice points, which can be reified into a search tree. Delphyne allows writing strategies as Python [generators](https://wiki.python.org/moin/Generators) that yield internal tree nodes and receive associated actions in return. The [`Strategy`][delphyne.Strategy] type has three type parameters, corresponding to the strategy's _signature_ (i.e., the type of nodes it can emit), its associated _inner policy type_ and its _return type_. Strategy functions are typically defined via the [`strategy`][delphyne.strategy] decorator, which creates functions that return [`StrategyInstance`][delphyne.StrategyInstance] values, wrapping the underlying generator while adding some metadata and convenience methods (e.g., [`using`][delphyne.StrategyInstance.using]).

```py
@strategy # (1)!
def find_param_value(
    expr: str) -> Strategy[Branch | Fail, FindParamValueIP, int]: ...
```

1. Example from the [previous chapter](./overview.md).

Query functions can have arguments of arbitrary type (including functions). When launching strategies from [commands](./extension.md#commands) and in the presence of type annotations, arguments are automatically unserialized using Pydantic. Thus, it is _useful_ for _top-level strategies_ to have serializable argument types that are properly annotated.

Branching nodes can be yielded via the [`branch`][delphyne.branch] function:
<!-- , whose type is worth examining (we ignore some optional arguments): -->

```py
def branch[P, T](cands: Opaque[P, T]) -> Strategy[Branch, P, T]: ...
```

Crucially, `branch` does not take a query as an argument but an **opaque space** ([`Opaque`][delphyne.Opaque]). An opaque space can be seen as a mapping from the ambient inner policy to an iterator of values (more precisely, a [search stream](./policies.md)). Opaque spaces can be produced from queries or strategy instances, by mapping the ambient inner policy to a prompting policy or a policy respectively:

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

### Queries

New query types can be defined by subclassing the [`Query`][delphyne.Query] class. We refer to the associated [API documentation][delphyne.Query] for details. Some highlights:

- Queries can be associated _system prompts_ and _instance prompts_. Prompts can be defined inline or in separate [Jinja](https://jinja.palletsprojects.com/en/stable/) files (see `find_invariants` example).
- Query types can have fields of any type as long as they can be serialized and unserialized using Pydantic (this includes custom dataclasses).
- Prompts can be parameterized, and parameters instantiated on the policy side (e.g., `params` argument of [`few_shot`][delphyne.few_shot]). This is useful for testing prompt variations, specializing specific prompt fragments for particular, etc...
- It is possible to define several [_answer modes_][delphyne.AnswerMode] for a query, each mode being associated a distinct parser (see `tests/example_strategies.py:GetFavoriteDish` for an example).
- Queries support [structured output](https://platform.openai.com/docs/guides/structured-outputs) and [tool calls](https://platform.openai.com/docs/guides/function-calling).


## Trees and Reification

Strategies can be reified (i.e. compiled) into trees using the [`reify`][delphyne.reify] function (see [reference documentation][delphyne.reify] for details and caveats). The [`Tree`][delphyne.Tree] class is defined as follows:

```py
@dataclass(frozen=True)
class Tree[N: Node, P, T]:
    node: N | Success[T]
    child: Callable[[Value], Tree[N, P, T]]
    ref: GlobalNodePath # (1)!
```

1. To ignore on first reading. See [documentation on references](../reference/strategies/traces.md).

A tree contains either a node ([`Node`][delphyne.Node]) or a success leaf ([`Node`][delphyne.Success]). When applicable, children trees are indexed by _actions_ ([`Value`][delphyne.core.refs.Value]). Actions result from combining elements of local spaces ([`Space`][delphyne.Space]). For example, if `node` has type [`Branch`][delphyne.Branch], then an action corresponds to a branching candidate.

### Adding New Effects {#adding-effects}

New types of effects beyond [`Branch`][delphyne.Branch] and [`Fail`][delphyne.Fail] can be added easily, by subclassing [`Node`][delphyne.Node]. Here are a number of additional effects defined in the Delphyne standard library:

- [`Join`][delphyne.Join]: allows evaluating a sequence of independent sub-strategies, possibly in parallel.
- [`Compute`][delphyne.Compute]: allows performing expensive and possibly non-replicable/nondeterministic computations in strategies (see details in [How-To Guide](../how-to-guides.md#compute)).
- [`Value`][delphyne.stdlib.Value]: allows adding value information into strategy trees. Such information can be leveraged by search policies (e.g. [`best_first_search`][delphyne.best_first_search]).
- [`Flag`][delphyne.Flag]: allows providing a finite number of alternative implementations for sub-tasks, to be selected either offline or at runtime.
- [`Message`][delphyne.Message]: allows decorating trees with debugging messages.

Node types are dataclasses whose fields can be of several kinds:

- **Nonparametric local spaces** ([`Space`][delphyne.Space]), the main types of which are _opaque spaces_ ([`OpaqueSpace`][delphyne.OpaqueSpace]) and _embedded trees_ ([`EmbeddedTree`][delphyne.EmbeddedTree]).
- **Parametric local spaces**, which are functions from local [values][delphyne.core.refs.Value] (i.e. assembly of elements from local spaces) to local spaces.
- **Data fields** that contain policy metadata, debugging information, etc...

More details are available in the [API Reference][delphyne.Node]. For examples of defining new effects, you can refer to the source code of the aforementioned effects in the Delphyne standard library (in the `delphyne.stdlib.nodes` module).