# Code2Inv Benchmark

This folder contains the [Code2Inv 2017](https://github.com/PL-ML/code2inv) competition problems for invariant generation, translated into WhyML.

## Selected Problems

An interesting problem that we often showcase for demonstrations is the following:

```
use int.Int

let main () diverges =
  let ref x = 1 in
  let ref y = 0 in
  while y < 100000 do
    x <- x + y;
    y <- y + 1
  done;
  assert { x >= y }
```

To solve this problem, one must find the following invariants, which are typically found in order.

```
x >= y
x >= 1
y >= 0
```

## Demonstration Problems

Here are some problems used for demonstrations:

```
use int.Int

let example1 () diverges =
  let ref x = any int in
  let ref y = 3 in
  let ref z = 0 in
  assume { x < 0 };
  while x < 10 do
    x <- x + 1;
    z <- z - 1;
    if z < 0 then y <- y + 1
  done;
  assert { x <> y }
```

```
use int.Int

let main () diverges =
  let n = any int in
  let ref x = 0 in
  while x < n do
    x <- x + 1;
  done;
  assume { n > 0 };
  assert { x = n }
```

```
use int.Int

let main () diverges =
  let ref x = 0 in
  let ref y = any int in
  let ref z = any int in
  while x < 500 do
    x <- x + 1;
    if z <= y then
      y <- z
  done;
  assert { z >= y }
```