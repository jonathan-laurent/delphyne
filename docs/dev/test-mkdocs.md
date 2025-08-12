# Testing MkDocs

<!-- Takeaway: do not forget `options:` -->

## Resources on Writing Good Docstrings

- [Pydantic Example](https://github.com/pydantic/pydantic/blob/b0175de473823f6f6927b9ecdc8998059727a086/pydantic/type_adapter.py#L68)

## Tricks

- In docstrings, both ``[`AttachedQuery`][delphyne.core.AttachedQuery]`` and ` [delphyne.core.AttachedQuery][]` will work but the former is better style.
- The `"rewrap.autoWrap.enabled": true` option is pretty good for what I want to do. Also, always select before rewrapping and we should be good.
- Start documenting each field in uppercase.

## API Documentation for Query

The [`Query`][delphyne.stdlib.Query] class is the base class for all queries in Delphyne, providing convenient features and automatic type inference. Does it work to have partial paths ([`Query`][delphyne.Query])?

## Documenting Type Aliases

::: delphyne.Stream

::: delphyne.AnswerPrefix

::: delphyne.Tag

::: delphyne.IPDict

## Documenting a Class

::: delphyne.Query
      
## Documenting a Class without Members

::: delphyne.Query
    options:
      members: false

## Documenting a Class With Selected Members

::: delphyne.Query
    options:
      members: [parse_answer, query_config]
      summary: true

## Documenting a Dataclass

::: delphyne.core.AttachedQuery

## Documenting a Function

::: delphyne.few_shot
    options:
      show_root_heading: false

## Documenting a whole module

::: delphyne.stdlib.queries
    options:
      show_source: false
