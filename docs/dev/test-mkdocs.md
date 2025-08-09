# Testing MkDocs

<!-- Takeaway: do not forget `options:` -->

## Resources on Writing Good Docstrings

- [Pydantic Example](https://github.com/pydantic/pydantic/blob/b0175de473823f6f6927b9ecdc8998059727a086/pydantic/type_adapter.py#L68)

## Tricks

- In docstrings, both ``[`AttachedQuery`][delphyne.core.AttachedQuery]`` and ` [delphyne.core.AttachedQuery][]` will work but the former is better style.
- The `"rewrap.autoWrap.enabled": true` option is pretty good for what I want to do. Also, always select before rewrapping and we should be good.
- Start documenting each field in uppercase.

## API Documentation for Query

The [`Query`][delphyne.stdlib.Query] class is the base class for all queries in Delphyne, providing convenient features and automatic type inference.

## Documenting a Class

::: delphyne.Query
    options:
      show_root_heading: true
      heading_level: 3

## Documenting another Class

::: delphyne.core.AttachedQuery
    options:
      heading_level: 3

## Documenting a Function

::: delphyne.few_shot
    options:
      show_root_heading: false
      heading_level: 3

## Documenting a whole module

::: delphyne.stdlib.queries
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3