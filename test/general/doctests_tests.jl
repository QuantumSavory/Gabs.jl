using Test
using Documenter
using Gabs
using StaticArrays # Preload the extension before Documenter captures REPL output.

@testset "Doctests" begin
    DocMeta.setdocmeta!(Gabs, :DocTestSetup, :(using Gabs); recursive=true)
    doctest(Gabs)
end
