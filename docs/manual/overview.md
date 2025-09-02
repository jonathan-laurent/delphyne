# Overview

Let us illustrate the Delphyne framework using a simple example that is also featured in the [Getting Started](../index.md) section. The task being solved in this example is, given a mathematical expression featuring real variables $x$ and $n$, to find an integer value for $n$ for which the expression is nonnegative for all values of $x$. We first propose a more verbose solution that better illustrates Delphyne's fundamental abstractions, and then demonstrate a shorter one. All code for these examples is available in `examples/small/find_param_value.py` and `examples/small/find_param_value_universal.py`

## Writing a Strategy {#writing-a-strategy}

Delphyne lets you combine prompting with traditional programming to solve problems. The first step is defining a _strategy_. A strategy provides a high-level sketch for solving a problem. It consists in a program with unresolved choice points, which are determined by LLMs at runtime. For example, here is an example of a strategy for the problem defined above (finding a value for integer parameter $n$ that makes a mathematical expression nonnegative for all $x$):

```python linenums="1"
import sympy as sp
from typing import assert_never
import delphyne as dp # (1)!
from delphyne import Branch, Fail, Strategy, strategy

@strategy
def find_param_value(
    expr: str
) -> Strategy[Branch | Fail, FindParamValueIP, int]: # (2)!
    x, n = sp.Symbol("x", real=True), sp.Symbol("n")
    symbs = {"x": x, "n": n}
    try:
        n_val = yield from dp.branch(
            FindParamValue(expr)
              .using(lambda p: p.guess, FindParamValueIP)) # (3)!
        expr_sp = sp.parse_expr(expr, symbs).subs({n: n_val})
        equiv = yield from dp.branch(
            RewriteExpr(str(expr_sp))
              .using(lambda p: p.prove, FindParamValueIP))
        equiv_sp = sp.parse_expr(equiv, symbs)
        equivalent = (expr_sp - equiv_sp).simplify() == 0
        yield from dp.ensure(equivalent, "not_equivalent")
        yield from dp.ensure(equiv_sp.is_nonnegative, "not_nonneg")
        return n_val
    except Exception as e: # (4)!
        assert_never((yield from dp.fail("sympy_error", message=str(e))))
```

1. We recommend importing Delphyne under alias `dp`, which is a convention we use in most examples. Most symbols from Delphyne's core or standard library can be accessed in this way. 
2. This return type can be ignored on first reading. The first type parameter passed to `Strategy` is the strategy's _signature_ (the list of _effects_ it can trigger or, equivalently, the types of tree nodes that can appear in the induced tree), the second parameter is the strategy's associated _inner policy type_ (see explanations in [Writing a Policy](#writing-a-policy)), and the third parameter is the type of returned values (here, an integer denoting the value of `n`).
3. The `using` method can be ignored on first reading. It takes as an argument a mapping from the ambient inner policy (of type `FindParamValueIP`, as specified in the strategy's return type) to a prompting policy for handling the `FindParamValue` query).
4. If a SymPy expression fails to parse or `simplify` raises an exception, the whole strategy fails.

The strategy above proceeds in three steps. First, it prompts an LLM to conjecture a value for $n$ (Lines 13-15). Then, it substitutes $n$ with the provided value and asks an LLM for a _proof_ that the resulting expression is positive for all $x$, in the form of an equivalent expression for which this fact obvious, in the sense that it can be derived from simple [interval arithmetic](https://en.wikipedia.org/wiki/Interval_arithmetic) (Lines 16-19). For example, expression $x^2 - 2x + 3$ can be rewritten into $(x - 1)^2 + 2.$ Finally, [SymPy](https://www.sympy.org/) is used to check the validity of this proof (Lines 20-23) and the value of $n$ is returned if the proof is valid. The two aforementioned LLM queries are defined as follows:


```py
@dataclass
class FindParamValue(dp.Query[int]): # (1)!
    """
    Given a sympy expression featuring a real variable `x` and an
    integer parameter `n`, find an integer value for `n` such that the
    expression is non-negative for all real `x`. Terminate your answer
    with a code block delimited by triple backquotes containing an integer.
    """ # (2)!

    expr: str
    __parser__ = dp.last_code_block.yaml


@dataclass
class RewriteExpr(dp.Query[str]):
    """
    Given a sympy expression featuring variable `x`, rewrite it into an
    equivalent form that makes it clear that the expression is
    nonnegative for all real values of `x`. Terminate your answer with a
    code block delimited by triple backquotes. This block must contain a
    new sympy expression, or nothing if no rewriting could be found.
    """

    expr: str
    __parser__ = dp.last_code_block.trim
```

1. The type argument passed to [`Query`][delphyne.Query] is the _answer type_ for this query. Custom datatypes are also allowed as long as they are serializable using Pydantic.
2. Short system prompts can be specified in docstrings but longer prompts are often more conveniently specified in separate [Jinja](https://jinja.palletsprojects.com/en/stable/) files. Custom _instance prompts_ can also be specified. By default, a YAML dumping of all query fields is used.

LLM requests are represented by _queries_. Queries are stratified by type. A _query type_ can be defined by inheriting the [`Query`][delphyne.Query] class and is defined by:

1. A name (e.g., `FindParamValue`).
2. An answer type (e.g., `int`).
3. A _system prompt_ that describes the underlying family of tasks (i.e., the docstring).
4. A series of fields describing specific task instances.
5. A parser that specifies how to parse LLM answers into the answer type.

!!! info "Few-Shot Prompting"
    As we discuss later, organizing queries into types is especially useful for implementing _few-shot_ prompting. Indeed, to answer a query of a given type, one can build a prompt that concatenates (1) the _system prompt_ associated with the query's type, (2) a sequence of _examples_ that each consist in a query of the same type along with the expected answer, and (3) the query to be answered.

Going back to the above [definition](#writing-a-strategy) of the `find_param_value` strategy, one can branch over possible answers to a query using the [`branch`][delphyne.branch] operator. In addition, one can ensure that a property holds using the [`ensure`][delphyne.ensure] operator, or fail straight using [`fail`][delphyne.fail]. Strategies can be [_reified_][delphyne.reify] (i.e., compiled) into search trees that feature __success leaves__ (corresponding to reaching a `return` statement), __failure leaves__ (corresponding to failing an `ensure` statement or reaching a `fail` statement), and internal __branching nodes__ (corresponding to reaching a `branch` statement). How such trees must be navigated at runtime (using LLM guidance) is defined by separate _policies_, which is the topic of the [next section](#writing-a-policy).

??? info "Difference between `ensure` and `assert`"
    Strategies can also feature Python assertions (`assert` statements) but these serve a different purpose from `ensure`. An `assert` statement failing means that an _internal_ error happened, either indicating a bug in the strategy code or invalid toplevel inputs. In contrast, an `ensure` statement can fail following incorrect LLM answers (and thereby induce a failure leaf), as a normal part of performing search.

??? info "Modularity and Extensibility"
    As we explain in the [next chapter](./strategies.md), the [`branch`][delphyne.branch] operator can be used to branch over answers of a query but __also__ over results of another strategy. In this way, any query in a strategy can be later refined into a dedicated strategy if more guidance is needed. Allowing such composition is one of the key language design challenges solved by Delphyne. In addition, the strategy language can be easily extended with new _effects_ (i.e., new types of tree nodes). Examples from the Delphyne standard library include: [`value`][delphyne.value], [`join`][delphyne.join], [`compute`][delphyne.compute], [`message`][delphyne.message], [`get_flag`][delphyne.get_flag]...

??? tip "Leveraging Typing"
    Strategies can be precisely typed, and their types statically checked using [Pyright](https://github.com/microsoft/pyright) in _strict mode_.

## Writing a Policy

Crucially, Delphyne allows fully separating the definition of search spaces (induced by _strategies_) from the algorithms used to navigate them (i.e., _policies_). This separation serves multiple practical purposes:

1. Many different policies can be implemented for any given strategy without modifying it, realizing different tradeoffs in terms of latency, budget consumption, reliability...
2. Policies can be tuned _independently_ without changing strategy code. In addition, [demonstrations](#adding-demonstrations) are query-agnostic and so tuning policies is guaranteed not to break demonstrations either (see [next section](#adding-demonstrations)).

In practice, policies tend to require more tuning and faster iteration cycles than strategies, since they often feature search hyperparameters for which good values are hard to guess _a priori_. Thus, they tend to evolve on a shorter time-scale and keeping them independent is valuable.

Below, we show a possible (parametric) policy for the `find_param_value` strategy.

```py linenums="1"
@dp.ensure_compatible(find_param_value) # (1)!
def serial_policy(
    model_name: dp.StandardModelName = "gpt-5-mini",
    proof_retries: int = 1
) -> dp.Policy[Branch | Fail, FindParamValueIP]:
    model = dp.standard_model(model_name)
    return dp.dfs() & FindParamValueIP(
        guess=dp.few_shot(model),
        prove=dp.take(proof_retries + 1) @ dp.few_shot(model))
```

1. This decorator is only used statically to have the type checker verify that the policy being defined has a type compatible with the `find_param_value` strategy.

As shown on Line 7, a policy ([`Policy`][delphyne.Policy]) consists in two components that are paired using the `&` operator. The first one is a search policy ([`SearchPolicy`][delphyne.SearchPolicy]), which consists in a search algorithm used to navigate the tree defined by the strategy, and which must be capable of handling `Branch` and `Fail` nodes in this case. Here, we use _depth-first search_, which is implemented in the [`dfs`][delphyne.dfs] standard library function. In addition, for every query issued by our strategy, we need to provide a suitable _prompting policy_ ([`PromptingPolicy`][delphyne.PromptingPolicy]) to describe how the query must be answered by LLMs. These prompting policies are gathered in an record of type `FindParamValueIP`, which is called the __inner policy type__ associated with `find_param_value` and which is defined as follows:

```py
@dataclass
class FindParamValueIP: # (1)!
    guess: dp.PromptingPolicy
    prove: dp.PromptingPolicy
```

1. The `IP` suffix stands for _inner policy_.

In general, for every strategy, an associated inner policy type is defined that precisely indicates what information must be provided to build associated policies (in addition to the toplevel search algorithm). Although the inner policy type is fairly simple in this case, it can get more complicated when (1) a strategy branches over results of another sub-strategy (in which case a [`Policy`][delphyne.Policy] must be provided) or (2) strategies involve loops or recursion (in which case prompting policies and sub-policies may depend on extra parameters such as the iteration number or recursion depth). In the definition of [`find_param_value`](#writing-a-strategy), the [`using`][delphyne.Query.using] method is called on queries to provide each time a mapping from the ambient inner policy (of type `FindParamValueIP`) to the prompting policy to use.

??? tip "A More Concise Alternative for Handling Inner Policies"
    Defining inner policy types and specifying functions from those to prompting policies or sub-policies via [`using`][delphyne.Query.using] allows a maximal degree of flexibility, static type safety and editor support. However, one might find it verbose. Thus, inner policies can also be specified via simple Python dictionaries (see [`IPDict`][delphyne.IPDict]), as we [soon](#a-more-concise-version) demonstrate.

The `serial_policy` policy defined above uses depth-first search ([`dfs`][delphyne.dfs]) to find solutions: every time a branching node is reached in the search tree, branching candidates are lazily enumerated and explored in order. As specified in the inner policy, branching candidates are generated by repeatedly sampling LLM answers, using the [`few_shot`][delphyne.few_shot] prompting policy (we discuss how few-shot examples can be added in the [next section](#adding-demonstrations)). The branching factor for producing proofs is set to `proof_retries + 1`, while the branching factor for producing candidates for $n$ is set to infinity (LLM answers are not deduplicated by default). As mentioned earlier, alternative policies can be defined. For example, the following policy repeatedly performs a parallel variant of depth-first search ([`par_dfs`][delphyne.par_dfs]), where multiple LLM completions are requested at every branching nodes and children are explored in parallel.

```python
@dp.ensure_compatible(find_param_value)
def parallel_policy(
    model_name: dp.StandardModelName = "gpt-5-mini",
    par_find: int = 2,
    par_rewrite: int = 2
) -> dp.Policy[Branch | Fail, FindParamValueIP]:
    model = dp.standard_model(model_name)
    return dp.loop() @ dp.par_dfs() & FindParamValueIP(
        guess=dp.few_shot(model, max_requests=1, num_completions=par_find),
        prove=dp.few_shot(model, max_requests=1, num_completions=par_rewrite))
```

In general, a wide variety of policies can be assembled using standard basic blocks such as [`dfs`][delphyne.dfs], [`par_dfs`][delphyne.par_dfs], [`few_shot`][delphyne.few_shot] and [`take`][delphyne.take] (many others are available, e.g., [`best_first_search`][delphyne.best_first_search]). However, adding new building blocks or search algorithms is easy. For example, [`par_dfs`][delphyne.par_dfs] can be simply redefined as follows:

```python
@search_policy
def par_dfs[P, T](
    tree: Tree[Branch | Fail, P, T],
    env: PolicyEnv,
    policy: P,
) -> StreamGen[T]:
    match tree.node:
        case Success(x):
            yield Solution(x)
        case Fail():
            pass
        case Branch(cands):
            cands = yield from cands.stream(env, policy).all()
            yield from Stream.parallel([
                par_dfs()(tree.child(a.tracked), env, policy)
                for a in cands]).gen()
```

The manual [chapter](./policies.md) on policies provides details on how new policy components can be defined, using [search stream combinators](../reference/stdlib/streams.md).

Once a policy is specified, we can run our strategy as follows:

```py
budget = dp.BudgetLimit({dp.NUM_REQUESTS: 2}) # (1)!
res, _ = (
    find_param_value("2*x**2 - 4*x + n")
    .run_toplevel(dp.PolicyEnv(demonstration_files=[]), serial_policy()) # (2)!
    .collect(budget=budget, num_generated=1))
print(res[0].tracked.value)  # e.g. 2
```

1. We define a budget limit for search, in terms of number of LLM requests. Other metrics are available: number of input/output tokens, API spending in dollars...
2. In addition to a policy, we also provide a _global policy environment_ ([`PolicyEnv`][delphyne.PolicyEnv]), which can be used to specify demonstration files (for few-shot prompting), prompt templates, LLM request caches...

Here, `gpt-5-mini` (the default model specified by `serial_policy`) is used to answer queries via zero-shot prompting. However, LLMs often work better when provided examples of answering similar queries. Delphyne features a domain-specific language for writing and maintaining such examples in the form of _demonstrations_.

## Adding Demonstrations

Although queries can sometimes be successfully answered via _zero-shot_ prompting, LLMs typically work better when examples of answering queries of the same type are provided. Such examples become essential parts of an LLM-enabled program, which must be kept in sync as it evolves. Delphyne offers a dedicated language for writing and maintaining examples. In this language, related examples are bundled with unit tests into coherent _demonstrations_. Demonstrations showcase concrete scenarios of navigating search trees and can be written interactively, using a tool-assisted, test-driven workflow.

Let us add a demonstration for the `find_param_value` strategy. We do so by adding the following to a demonstration file (with extension `.demo.yaml`):

```yaml
- strategy: find_param_value
  args:
    expr: "x**2 - 2*x + n"
  tests:
    - run | success
  queries: []
```

The test in this snippet indicates that the goal is to demonstrate how to successfully solve a specific instance of our strategy, using a set of query/answer pairs provided in the `queries` section. This section is initially empty. Thus, evaluating the demonstration above (using the `Evaluate Demonstration` code action from the [VSCode extension](./extension.md)) results in a warning:

![](../assets/screenshot/overview-extension-example/dark-large.png#only-dark)
![](../assets/screenshot/overview-extension-example/light-large.png#only-light)

Here, the extension indicates that the demonstation's unique test is _stuck_, due to a missing query answer. Using the extension's tree view, the user can visualize _where_ in the tree it is stuck and add the missing query to the demonstration by clicking on the `+` icon:

```yaml
- strategy: find_param_value
  args:
    expr: "x**2 - 2*x + n"
  tests:
    - run | success
  queries: 
    - query: FindParamValue
      args:
        expr: x**2 - 2*x + n
      answers: []
```

The user can then add an answer to the query, either manually or by querying an LLM and editing the result. After this, the demonstration can be evaluated again and the process repeats, until all tests pass. In this case, the final demonstration is:


```yaml
- strategy: find_param_value
  args:
    expr: "x**2 - 2*x + n"
  tests:
    - run | success
  queries: 
    - query: FindParamValue
      args:
        expr: x**2 - 2*x + n
      answers:
        - answer: | # (1)!
            ```
            1
            ```
    - query: RewriteExpr
      args:
        expr: x**2 - 2*x + 1
      answers:
        - answer: |
            ```
            (x - 1)**2
            ```
```

1. As specified by the [parser](#writing-a-strategy) of `FindParamValue`, answers must end with a code block featuring a YAML object. A chain of thoughts or an explanation can be provided outside the code block, although we do not do so here.

Once such a demonstration is written, the associated tests guarantee that it stays relevant and consistent as the whole program evolves. The Delphyne language server provides a wide range of diagnostics for demonstrations, from detecting unreachable or unparseable answers to indicating runtime strategy exceptions.

The demonstration language allows expressing more advanced tests. In particular, it allows specifying negative examples and describing scenarios of recovering from bad decisions during search. More details are provided in the [Demonstrations](./demos.md) chapter of the manual.

!!! note "Demonstrations are Policy-Agnostic"
    Crucially, demonstrations are **policy-agnostic**. In fact, a common practice is to start writing demonstrations after a strategy is defined and _before_ associated policies are specified. Modifying policies is guaranteed never to break a demonstration, leveraging the full separation between _strategies_ and _policies_ at the heart of Delphyne's design.

    In order to walk through search trees without depending on a specific policy, the `run` test instruction leverages local [navigation functions][delphyne.Navigation] that are defined for every type of tree node (including custom nodes added by users). For example, upon encountering a branching node of type [`Branch`][delphyne.Branch], `run` examines whether branching occurs over query answers or over the results of another strategy. In the first case, an answer to the featured query is fetched from the `queries` section of the demonstration. In the second case, `run` executes recursively.

## A More Concise Version

<!-- trading some static type safety and flexibility in exchange for concision.  -->

There are two ways in which the strategy above can be shortened (while preserving its logic). First, one can use _inner policy dictionaries_ instead of manually defining and manipulating inner policy types such as `FindParamValueIP` (see [`IPDict`][delphyne.IPDict] for details). Second, Delphyne offers an experimental feature that saves users from manually defining queries. Indeed, the [`guess`][delphyne.guess] operator can be used to ask LLMs for an object of a given type, given some context that is automatically extracted from the stack frame at the call site.

Using these two features, the `find_param_value` strategy can be rewritten as follows:

```py
import sympy as sp
from typing import assert_never
import delphyne as dp 
from delphyne import Branch, Fail, IPDict, Strategy, strategy


@strategy
def find_param_value(expr: str) -> Strategy[Branch | Fail, IPDict, int]:
    """
    Find an integer `n` that makes a given math expression nonnegative
    for all real `x`. Prove that the resulting expression is nonnegative
    by rewriting it into an equivalent form.
    """
    x, n = sp.Symbol("x", real=True), sp.Symbol("n")
    symbs = {"x": x, "n": n}
    try:
        n_val = yield from dp.guess(int, using=[expr])
        expr_sp = sp.parse_expr(expr, symbs).subs({n: n_val})
        equiv = yield from dp.guess(str, using=[str(expr_sp)])
        equiv_sp = sp.parse_expr(equiv, symbs)
        equivalent = (expr_sp - equiv_sp).simplify() == 0
        yield from dp.ensure(equivalent, "not_equivalent")
        yield from dp.ensure(equiv_sp.is_nonnegative, "not_nonneg")
        return n_val
    except Exception as e:
        assert_never((yield from dp.fail("sympy_error", message=str(e))))


def serial_policy():
    model = dp.standard_model("gpt-5-mini")
    return dp.dfs() & {
        "n_val": dp.few_shot(model),
        "equiv": dp.take(2) @ dp.few_shot(model),
    }
```

Behind the scenes, [`guess`][delphyne.guess] issues instances of the [`UniversalQuery`][delphyne.UniversalQuery] query from Delphyne's standard library. We show a concrete example of a generated prompt below:

??? info "Prompt Details"

    Here is an instance of the exact prompt that is generated for the second instance of `guess` that assigns a value to `equiv`.

    **System Prompt:**

    I am executing a program that contains nondeterministic assignments along with assertions (e.g., in the form of `ensure` and `fail` statements). I am stuck at one of these nondeterministic assignments and your goal is to generate an assigned value, in such a way that the program can go on and not fail any assertion.

    More specifically, I'll give you three pieces of information:

    - A nondeterministic program.
    - The name of the variable that is being assigned at the program location where I am currently stuck.
    - Some values for a number of local variables.

    Your job is to generate a correct value to assign. The expected type of this value is indicated inside the nondeterministic assignment operator.

    Terminate your answer with a code block (delimited by triple backquotes) that contains a YAML object of the requested type. Do not wrap this YAML value into an object with a field named like the assigned variable.

    **Instance Prompt:**

    ~~~md
    Program:

    ```
    @strategy
    def find_param_value(expr: str) -> Strategy[Branch | Fail, IPDict, int]:
        """
        Find an integer `n` that makes a given math expression nonnegative
        for all real `x`. Prove that the resulting expression is nonnegative
        by rewriting it into an equivalent form.
        """
        x, n = sp.Symbol("x", real=True), sp.Symbol("n")
        symbs = {"x": x, "n": n}
        try:
            n_val = yield from dp.guess(int, using=[expr])
            expr_sp = sp.parse_expr(expr, symbs).subs({n: n_val})
            equiv = yield from dp.guess(str, using=[str(expr_sp)])
            equiv_sp = sp.parse_expr(equiv, symbs)
            equivalent = (expr_sp - equiv_sp).simplify() == 0
            yield from dp.ensure(equivalent, "not_equivalent")
            yield from dp.ensure(equiv_sp.is_nonnegative, "not_nonneg")
            return n_val
        except Exception as e:
            assert_never((yield from dp.fail("sympy_error", message=str(e))))
    ```

    Variable being currently assigned: equiv

    Selected local variables:

    ```yaml
    str(expr_sp): x**2 - 2*x + 1
    ```

    Type of value to generate (as a YAML object): <class 'str'>
    ~~~