@testitem "Automatic differentiation" begin
    using Gabs
    using DifferentiationInterface
    import ForwardDiff
    import FiniteDiff


    @testset "derivatives" begin

        n = rand(1:5)
        xs = rand(Float64)

        function f1(x::R) where {R<:Real}
            basis = QuadPairBasis(n)
            state = vacuumstate(basis)
            op = attenuator(basis, x, 2)
            newstate = op * state
            return abs(1 - purity(newstate))^2
        end

        findiff_f1 = derivative(f1, AutoFiniteDiff(), xs)
        fordiff_f1 = derivative(f1, AutoForwardDiff(), xs)

        @test isapprox(findiff_f1, fordiff_f1, atol = 1e-4)

    end

    @testset "gradients" begin

        n = rand(1:5)
        xs = rand(n)

        function f1(x::AbstractVector{<:Real})
            basis = QuadPairBasis(length(x))
            state = squeezedstate(basis, x, x)
            op = amplifier(basis, x, ones(length(x)))
            newstate = op * state
            return abs(1 - purity(newstate))^2
        end

        findiff_f1 = gradient(f1, AutoFiniteDiff(), xs)
        fordiff_f1 = gradient(f1, AutoForwardDiff(), xs)

        @test isapprox(findiff_f1, fordiff_f1, atol = 1e-4)

    end
end
        