using Documenter, SimSearchManifoldLearning

makedocs(;
    modules=[SimSearchManifoldLearning],
    authors="Eric S. Tellez",
    repo="https://github.com/sadit/SimSearchManifoldLearning.jl/blob/{commit}{path}#L{line}",
    sitename="SimSearchManifoldLearning.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://sadit.github.io/SimSearchManifoldLearning.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "UMAP" => "apiumap.md",
        "API ManifoldLearning" => "apiml.md"
    ],
    warnonly = true
)

deploydocs(;
    repo="github.com/sadit/SimSearchManifoldLearning.jl",
    devbranch=nothing,
    branch = "gh-pages",
    versions = ["stable" => "v^", "v#.#.#", "dev" => "dev"]
)
