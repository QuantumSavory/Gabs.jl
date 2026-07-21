using Gabs
using JET
using Test

@testset "JET checks" begin
    rep = JET.report_package(Gabs; target_modules=(Gabs,))
    @show rep

    reports = JET.get_reports(rep)
    # Baseline from Julia 1.12.6 and JET 0.11.6; lower this as reports are fixed.
    @test length(reports) <= 15
    @test_broken isempty(reports)
end
