# Leandra: A Simple and Flexible Proving Agent for Lean 4

## Setup

To use a local Loogle server, set the `LOOGLE_URL` environment variable.
For example:

```sh
export LOOGLE_URL="http://localhost:8088"
```

## Future Improvements

- Spend more resources when proving something is really worth it. For example, if all holes except one have been filled, maybe more effort should be put into proving it.
- If some goals could not be filled, then we give feedback based on the name of those that could not be proved.