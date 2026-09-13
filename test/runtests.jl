using TestItemRunner

const JET_PROJECT = normpath(joinpath(@__DIR__, "projects", "jet"))
const test_args = isempty(ARGS) ? ["general"] : ARGS
const JET_flag = length(test_args) == 1 && startswith(only(test_args), "jet")

CUDA_flag = false
if Sys.iswindows()
    @info "Skipping GPU tests -- only executed on *NIX platforms."
else
    CUDA_flag = get(ENV, "CUDA_TEST", "") == "true"
    CUDA_flag && @info "Running with CUDA tests."
    if !CUDA_flag
        @info "Skipping GPU tests -- must be explicitly enabled."
        @info "Environment must set CUDA_TEST=true."
    end
end

if JET_flag
    @info "Activating the dedicated JET test environment." project=JET_PROJECT
    using Pkg

    Pkg.activate(JET_PROJECT)
    Pkg.instantiate()
    include("jet_tests.jl")
else
    if CUDA_flag
        using Pkg
        Pkg.add("CUDA")
    end

    using Gabs

    testfilter = ti -> begin
        exclude = Symbol[:jet]
        if CUDA_flag
            return :cuda in ti.tags
        else
            push!(exclude, :cuda)
        end
        if !(VERSION >= v"1.10") || get(ENV, "QUANTUMSAVORY_DOWNGRADE_TEST", "") == "true"
            push!(exclude, :doctests)
            push!(exclude, :aqua)
        end

        return all(!in(exclude), ti.tags)
    end

    println("Starting tests with $(Threads.nthreads()) threads out of `Sys.CPU_THREADS = $(Sys.CPU_THREADS)`...")

    @run_package_tests filter=testfilter
end
