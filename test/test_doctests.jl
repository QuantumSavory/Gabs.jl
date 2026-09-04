@testitem "Doctests" tags=[:doctests] begin
    using Documenter
    using Gabs
    using StaticArrays # Preload the extension before Documenter captures REPL output.

    DocMeta.setdocmeta!(Gabs, :DocTestSetup, :(using Gabs); recursive=true)
    doctest(Gabs)
end
