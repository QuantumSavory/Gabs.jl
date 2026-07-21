using TestItemRunner

if ARGS == ["jet"]
    using Pkg

    Pkg.activate(joinpath(@__DIR__, "projects", "jet"))
    Pkg.instantiate()
    include("jet_tests.jl")
else
    using Gabs

    testfilter = ti -> begin
        exclude = Symbol[:jet]
        if !(VERSION >= v"1.10")
            push!(exclude, :doctests)
            push!(exclude, :aqua)
        end

        return all(!in(exclude), ti.tags)
    end

    println("Starting tests with $(Threads.nthreads()) threads out of `Sys.CPU_THREADS = $(Sys.CPU_THREADS)`...")

    @run_package_tests filter=testfilter
end
