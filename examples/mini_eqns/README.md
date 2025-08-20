# Proving Math Equalities with Delphyne

> [!WARNING]
> This example is still under development.

This folder contains an example of using Delphyne to generate machine-checkable proofs of simple trigonometric equalities. It is inspired by the _Equations_ environment from the [HyperTree Proof Search paper](https://arxiv.org/pdf/2205.11491).


## Benchmark

We list here all the trigonometric inequalities from Table 9 of the HyperTree Proof Search paper (linked above).

```
sin(pi/2 + x) = cos(x)
cos(pi/2 - x) = sin(x)
cos(pi + x) = -cos(x)
sin(pi - x) = sin(x)
cos(pi/3) = sin(pi/6)
cos(pi/4) = sin(pi/4)
cos(pi/6) = sin(pi/3)
cos(2*pi + x) = cos(x)
sin(2*pi + x) = sin(x)

cos(x)**2 + sin(x)**2 = 1
cos(x) = cos(x/2)**2 - sin(x/2)**2
sin(x + y) - sin(x - y) = 2*cos(x)*sin(y)
cos(x - y) + cos(x + y) = 2*cos(x)*cos(y)
sin(2*x) = 2*sin(x)*cos(x)
cos(2*x) = 1 - 2*sin(x)**2
cos(2*x) = 2*cos(x)**2 - 1
sin(x) = 2*sin(x/2)*cos(x/2)
cos(x + y)*cos(x - y) = cos(x)**2 - sin(y)**2
sin(x + y)*sin(y - x) = cos(x)**2 - cos(y)**2
sin(x)**3 = (3*sin(x) - sin(3*x))/4
sin(3*x) = 3*sin(x) - 4*sin(x)**3
sin(4*x) = cos(x)*(4*sin(x) - 8*sin(x)**3)
```

Used as examples:

```
cos(pi/3) = sin(pi/6)
2*sin(x/2)*cos(x/2) = sin(x)
```