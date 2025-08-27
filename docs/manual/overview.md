# Overview

Let us illustrate the Delphyne framework using a simple example that is also featured in the [Getting Started](../index.md) section. The task being solved in this example is, given a mathematical expression featuring real variables $x$ and $n$, to find an integer value for $n$ for which the expression is nonnegative for all values of $x$. We first propose a more verbose solution that better illustrates Delphyne's fundamental abstractions, and then demonstrate a shorter one. All code for these examples is available in `examples/small/find_param_value.py` and `examples/small/find_param_value_universal.py`

## Writing a Strategy {#writing-a-strategy #find_param_value}

Delphyne lets you combine prompting with traditional programming to solve problems. The first step is defining a _strategy_. A strategy provides a high-level sketch for solving a problem. It consists in a program with unresolved choice points, which are determined by LLMs at runtime. For example, here is an example of a strategy for the problem defined above (finding a value for integer parameter $n$ that makes a mathematical expression nonnegative for all $x$):

```python linenums="1"
import sympy as sp
from typing import assert_never
import delphyne as dp
from delphyne import Branch, Fail, Strategy, strategy

@strategy
def find_param_value(
    expr: str
) -> Strategy[Branch | Fail, FindParamValueIP, int]: # (1)!
    x, n = sp.Symbol("x", real=True), sp.Symbol("n")
    symbs = {"x": x, "n": n}
    try:
        n_val = yield from dp.branch(
            FindParamValue(expr)
              .using(lambda p: p.find, FindParamValueIP)) # (2)!
        expr_sp = sp.parse_expr(expr, symbs).subs({n: n_val})
        equiv = yield from dp.branch(
            RewriteExpr(str(expr_sp))
              .using(lambda p: p.rewrite, FindParamValueIP))
        equiv_sp = sp.parse_expr(equiv, symbs)
        equivalent = (expr_sp - equiv_sp).simplify() == 0
        yield from dp.ensure(equivalent, "not_equivalent")
        yield from dp.ensure(equiv_sp.is_nonnegative, "not_nonneg")
        return n_val
    except Exception as e: # (3)!
        assert_never((yield from dp.fail("sympy_error", message=str(e))))
```

1. This return type can be ignored on first reading. The first type parameter passed to `Strategy` is the strategy's _signature_ (the list of _effects_ it can trigger or, equivalently, the types of tree nodes that can appear in the induced tree), the second parameter is the strategy's associated _inner policy type_ (see explanations in [Writing a Policy](#writing-a-policy)), and the third parameter is the type of returned values (here, an integer denoting the value of `n`).
2. The `using` method can be ignored on first reading. It takes as an argument a mapping from the ambient inner policy (of type `FindParamValueIP`, as specified in the strategy's return type) to a prompting policy for handling the `FindParamValue` query).
3. If a SymPy expression fails to parse or `simplify` raises an exception, the whole strategy fails.

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

Going back to the above [definition](#find_param_value) of the `find_param_value` strategy, one can branch over possible answers to a query using the [`branch`][delphyne.branch] operator. In addition, one can ensure that a property holds using the [`ensure`][delphyne.ensure] operator, or fail straight using [`fail`][delphyne.fail]. Strategies can be [_reified_](delphyne.reify) (i.e., compiled) into search trees that feature __success leaves__ (corresponding to reaching a `return` statement), __failure leaves__ (corresponding to failing an `ensure` statement or reaching a `fail` statement), and internal __branching nodes__ (corresponding to reaching a `branch` statement). How such trees must be navigated at runtime (using LLM guidance) is defined by separate _policies_, which is the topic of the [next section](#writing-a-policy).

??? info "Difference between `ensure` and `assert`"
    Strategies can also feature Python assertions (`assert` statements) but these serve a different purpose from `ensure`. An `assert` statement failing means that an _internal_ error happened, either indicating a bug in the strategy code or invalid toplevel inputs. In contrast, an `ensure` statement can fail following incorrect LLM answers (and thereby induce a failure leaf), as a normal part of performing search.

??? info "Modularity and Extensibility"
    As we explain in the [next chapter](./strategies.md), the [`branch`][delphyne.branch] operator can be used to branch over answers of a query but __also__ over results of another strategy. In this way, any query in a strategy can be later refined into a dedicated strategy if more guidance is needed. Allowing such composition is one of the key language design challenges solved by Delphyne. In addition, the strategy language can be easily extended with new _effects_ (i.e., new types of tree nodes). Examples from the Delphyne standard library include: [`value`][delphyne.value], [`join`][delphyne.join], [`compute`][delphyne.compute], [`message`][delphyne.message], [`get_flag`][delphyne.get_flag]...

??? tip "Leveraging Typing"
    Strategies can be precisely typed, and their types statically checked using [Pyright](https://github.com/microsoft/pyright) in _strict mode_.

## Writing a Policy

## Adding Demonstrations

<!-- ::: examples.small.find_param_value.find_param_value
    options:
      show_root_heading: false -->

