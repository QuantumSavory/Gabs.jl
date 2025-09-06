using TestItemRunner
using Gabs

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

using Pkg
CUDA_flag && Pkg.add("CUDA")

testfilter = ti -> begin
    exclude = Symbol[:jet]
    if get(ENV, "JET_TEST", "") == "true"
        return :jet in ti.tags
    else
        push!(exclude, :jet)
    end
    if CUDA_flag
        return :cuda in ti.tags
    else
        push!(exclude, :cuda)
    end
    if !(VERSION >= v"1.10")
        push!(exclude, :doctests)
        push!(exclude, :aqua)                                 
    end
    return all(!in(exclude), ti.tags)
end

println("Starting tests with $(Threads.nthreads()) threads out of `Sys.CPU_THREADS = $(Sys.CPU_THREADS)`...")

@run_package_tests filter=testfilter

if get(ENV,"JET_TEST","")=="true"
    @run_package_tests filter=(ti -> :jet in ti.tags)
end