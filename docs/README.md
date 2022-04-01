# Running Documentation

If running for first time run julia using docs as the project, for local build of docs

```shell
julia --project=docs/
```

Instantiate a new environment and add Documenter and development package as dependencies

```julia
pkg> instantiate
pkg> add Documenter
pkg> dev .
julia> include("docs/make.jl")
```

On subsequent occassions, it is sufficient to activate the docs environent, and run docs/make.jl

```julia
pkg> activate docs
julia> include("docs/make.jl")
pkg> activate . # to revert to project environment
```
