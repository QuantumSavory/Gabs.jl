using ParallelTestRunner

const JET_PROJECT = normpath(joinpath(@__DIR__, "projects", "jet"))
const JET_TEST_PATH = joinpath(@__DIR__, "jet_tests.jl")

args = isempty(ARGS) ? ["general"] : ARGS
jet_only = length(args) == 1 && startswith(only(args), "jet")
if isempty(ARGS)
    @info "No test arguments provided; defaulting to `general` tests."
end
if jet_only
    @info "Routing to direct JET test execution." args project=JET_PROJECT
    using Pkg

    Pkg.activate(JET_PROJECT)
    Pkg.instantiate()
else
    @info "Routing to ParallelTestRunner." args
end

testsuite = find_tests(@__DIR__)
filter!(testsuite) do (name, _)
    endswith(name, "_tests")
end

if !(VERSION >= v"1.10") || get(ENV, "QUANTUMSAVORY_DOWNGRADE_TEST", "") == "true"
    delete!(testsuite, "general/doctests_tests")
end

if jet_only
    # Run JET directly rather than via ParallelTestRunner because
    # JET does not like being loaded after a Pkg.activate change.
    include(JET_TEST_PATH)
else
    using Pkg
    Pkg.precompile()
    using Gabs
    runtests(Gabs, args; testsuite)
end
