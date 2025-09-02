# The Policy Language

<!-- TODO: where is caching documented precisely? -->

Delphyne offers a [layered API](https://www.fast.ai/posts/2020-02-13-fastai-A-Layered-API-for-Deep-Learning.html) for defining policies. At a high level, policies can be specified by assembling standard components. At a lower level, atomic components can be defined using search stream combinators. We summarize the key concepts and abstractions underlying Delphyne's policy language below. Links are provided to the [Reference](../reference/policies/definitions.md) for details.

## Key Concepts

- [**Policy**][delphyne.Policy]: a policy is a pair of a _search policy_ and of an _inner policy_. Combining a _strategy_ and a _policy_ results in a _search stream_.
- [**Search Policy**][delphyne.SearchPolicy]: a search policy is a function that takes a tree, a _global policy environment_ and an _inner policy_ as arguments and returns a search stream.
- **Inner Policy**: an object that associates some prompting policies and policies to all query and sub-strategy instances inside a strategy. Inner policies are typically instances of custom dataclasses but Python dictionaries can also be used, trading flexibility and static type safety in exchange for concision (see [`IPDict`][delphyne.IPDict]).
- [**Prompting Policy**][delphyne.PromptingPolicy]: a function that maps a query along with a _global policy environment_ to a _search stream_ (e.g. [`few_shot`][delphyne.few_shot]).
- [**Global Policy Environment**][delphyne.PolicyEnv]: global environment accessible to all policies, which allows: fetching prompts and examples, caching LLM requests, generating traces and logs...
- [**Search Stream**][delphyne.Stream]: a (possibly infinite) iterator of yielded solutions and resource management messages (requesting authorization for spending some resource _budget_ or declaring actual resource consumption). Search streams follow the [search stream protocol][delphyne.core.streams] and can be assembled using [standard combinators][delphyne.Stream].
- [**Budget**][delphyne.Budget]: a finite-support function from resource consumption metrics (e.g. [number of requests][delphyne.NUM_REQUESTS], [LLM API spending in dollars][delphyne.DOLLAR_PRICE]) to real values.
- [**Stream Transformer**][delphyne.StreamTransformer]: a function that maps a search stream into another search stream (e.g. [`with_budget`][delphyne.with_budget] and [`take`][delphyne.take]). Stream transformers can be composed with search policies, prompting policies and other stream transformers using `@`.
- [**Tree Transformer**][delphyne.ContextualTreeTransformer]: a function that maps a tree into another one, possibly with a different signature (e.g. [`elim_compute`][delphyne.elim_compute], [`elim_messages`][delphyne.elim_messages]). Can be composed with search policies using `@` to modify their accepted signature.

## Defining New Search Policies

For examples of defining new search policies, we recommend inspecting the source code of [`dfs`][delphyne.dfs], [`par_dfs`][delphyne.par_dfs] and [`best_first_search`][delphyne.best_first_search].