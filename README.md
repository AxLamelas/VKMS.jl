# Variable-Knot Monotonic Spline (VKMS.jl) 
[![DOI](https://zenodo.org/badge/777892627.svg)](https://zenodo.org/doi/10.5281/zenodo.10882186)

This package implements the algorithm described in [citation] to fit non-parametrically a function observed through a non-linear functional.

A basic model is provided `KnotModel` which consists of a variable length group, a slope `m` and an y-intercept `b`. The way the model is calculated can customized by overwriting `model_function_factory(m::KnotModel)`
