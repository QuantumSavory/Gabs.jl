@testitem "Measurements" begin
    using Gabs
    using StaticArrays

    @testset "generaldyne" begin    
        vac = vacuumstate()
        vacs = vac ⊗ vac ⊗ vac ⊗ vac
        gd1 = Generaldyne(vacs, vac ⊗ vac, [2, 4])
        out1 = outcome(gd1)
        @test isequal(out1, vac ⊗ vac)

        coh = coherentstate(1.0+im)
        cohs = coh ⊗ vac ⊗ coh ⊗ vac
        epr = eprstate(1.0, 3.0)
        gd2 = Generaldyne(cohs, epr, [1, 4])
        out2 = outcome(gd2)
        @test isequal(out2, vac ⊗ coh)

        state = GaussianState(Vector{Float64}(collect(1:4)), Matrix{Float64}(reshape(collect(1:16), (4,4))))
        meas = GaussianState(Vector{Float64}(collect(1:2)), Matrix{Float64}(reshape(collect(1:4), (2,2))))
        gd3 = Generaldyne(state, meas, [2])
        out3 = outcome(gd3)
        xA, xB = [1.0, 2.0], [3.0, 4.0]
        VA, VB, VAB = [1.0 5.0; 2.0 6.0], [11.0 15.0; 12.0 16.0], [9.0 13.0; 10.0 14.0]
        out3_mean = xA .+ VAB*((inv(VB .+ meas.covar))*(meas.mean .- xB))
        out3_covar = VA .- VAB*((inv(VB .+ meas.covar))*transpose(VAB))
        @test isapprox(out3, GaussianState(out3_mean, out3_covar))

        sstatic = vacuumstate(SVector{2}, SMatrix{2,2})
        statestatic = sstatic ⊗ sstatic ⊗ sstatic ⊗ sstatic
        gdstatic = Generaldyne(statestatic, sstatic, [2])
        outstatic = outcome(gdstatic)
        @test isequal(outstatic, sstatic ⊗ sstatic ⊗ sstatic)
    end
end