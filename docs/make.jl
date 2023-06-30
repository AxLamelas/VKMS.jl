using Documenter
using VLEvolution

makedocs(
    sitename = "VLEvolution",
    format = Documenter.HTML(),
    modules = [VLEvolution]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
