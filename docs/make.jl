using Documenter
using VKMS

makedocs(
    sitename = "VKMS",
    format = Documenter.HTML(),
    modules = [VKMS]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
