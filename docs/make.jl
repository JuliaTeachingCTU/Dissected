using Documenter, Literate

# Literate.markdown("src/retentative/retentative.jl", "src/retentative"; flavor=Literate.CommonMarkFlavor())
# Literate.markdown("src/retentative/retnet_heads.jl", "src/retentative"; flavor=Literate.CommonMarkFlavor())
# Literate.markdown("src/retentative/retentative_optimization.jl", "src/retentative"; flavor=Literate.CommonMarkFlavor())
# Literate.markdown("src/retentative/retnet_differentiation.jl", "src/retentative"; flavor=Literate.CommonMarkFlavor())
# Literate.markdown("src/ir/ad.jl", "src/ir"; flavor=Literate.CommonMarkFlavor())
Literate.markdown("src/relax/relax.jl", "src/relax"; flavor=Literate.CommonMarkFlavor())


makedocs(
	pages = [
		"index.md",
        # "Retentative neural networks" => [
        # "introduction" => "retentative/retentative.md",
        # "adding heads" => "retentative/retnet_heads.md",
        # "optimizing memory" => "retentative/retentative_optimization.md",
        # "adding gradients" => "retentative/retnet_differentiation.md",
        # ],
        "Backpropagation through void" => [ "relax/relax.md"]
    ],
	sitename="Dissecting",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        mathengine = MathJax3(),
    ),
)

