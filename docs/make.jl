# Make source available
push!(LOAD_PATH, "../src/")

using Documenter
using BetaModels

makedocs(
    sitename = "BetaModels.jl Documentation",
    pages = ["Home" => "index.md"],
    format = Documenter.HTML(prettyurls = false),
)
deploydocs(repo = "github.com/jatotterdell/BetaModels.jl.git", devbranch = "main")
