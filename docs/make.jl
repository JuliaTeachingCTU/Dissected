using Documenter, Literate

Literate.markdown("src/retentative/retentative.jl", "src/retentative"; flavor=Literate.CommonMarkFlavor())


makedocs(
	pages = [
		"index.md",
        "Retentative neural networks" => "retentative/retentative.md",
    ],
	sitename="Dissecting",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    )
)

