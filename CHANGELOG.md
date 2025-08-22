# Changelog

## Version 0.7.0 (2025-08-22)

- **Breaking:** overhaul of parsers in the standard library. Parsers are now simpler and more composable. In particular, it is now possible to transform parsers by mapping a function to their results or adding validators. Some (partial) upgrading instructions:
  - Replace `raw_yaml` by `get_text.yaml`.
  - Replace `yaml_from_last_code_block` by `last_code_block.yaml`.
  - Replace the `"structured"` parser spec by `structured`.
  - Replace the `"structured"` parser spec by `structured`.
  - Look at the new signature for `Query.__parser__` and at the new methods `Query.parser` and `Query.parser_for`, which replace `Query.query_config`.

## Version 0.6.1 (2025-08-19)

First released version with a full API Reference documentation. From this version on, Delphyne adheres to semantic versioning.