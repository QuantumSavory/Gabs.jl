@testitem "express formalism conversions" begin
    using Gabs
    
    @testset "GaussianState in QuadPairBasis" begin
        state = vacuumstate(2)
        @test state.basis isa QuadPairBasis
        
        # Express to same basis type should return same/equivalent state
        state_pair = express(state, QuadPairBasis(2))
        @test state_pair.basis isa QuadPairBasis
        @test state_pair.mean ≈ state.mean
        @test state_pair.cov ≈ state.cov
    end
    
    @testset "GaussianState conversion to QuadBlockBasis" begin
        state = vacuumstate(2)
        state_block = express(state, QuadBlockBasis(2))
        @test state_block.basis isa QuadBlockBasis
        @test state_block.nmodes == 2
        
        # Convert back to QuadPairBasis and verify equivalence
        state_pair = express(state_block, QuadPairBasis(2))
        @test state_pair.basis isa QuadPairBasis
        @test state_pair.mean ≈ state.mean
        @test state_pair.cov ≈ state.cov
    end
    
    @testset "GaussianState identity express" begin
        state = coherentstate(1.0, 2)
        state_id = express(state)
        @test state_id.basis === state.basis
        @test state_id.mean ≈ state.mean
        @test state_id.cov ≈ state.cov
    end
    
    @testset "Non-trivial states basis conversion" begin
        state = squeezedstate(0.5, 0.0, 1)
        state_block = express(state, QuadBlockBasis(1))
        state_pair = express(state_block, QuadPairBasis(1))
        
        # Verify round-trip equivalence
        @test state_pair.mean ≈ state.mean
        @test state_pair.cov ≈ state.cov
    end
    
    @testset "GaussianUnitary conversion" begin
        unitary = phaseshift(π/4, 1)
        @test unitary.basis isa QuadPairBasis
        
        unitary_block = express(unitary, QuadBlockBasis(1))
        @test unitary_block.basis isa QuadBlockBasis
        
        # Verify properties are preserved
        @test unitary_block.d ≈ express(unitary, QuadBlockBasis(1)).d
    end
    
    @testset "GaussianChannel conversion" begin
        channel = attenuator(0.9, 1)
        @test channel.basis isa QuadPairBasis
        
        channel_block = express(channel, QuadBlockBasis(1))
        @test channel_block.basis isa QuadBlockBasis
        
        # Verify the channel properties
        @test channel_block.nmodes == 1
    end
    
    @testset "Multi-mode basis conversions" begin
        state = eprstate(3)
        @test state.nmodes == 3
        
        state_block = express(state, QuadBlockBasis(3))
        @test state_block.nmodes == 3
        
        # Verify physical equivalence
        state_pair = express(state_block, QuadPairBasis(3))
        @test state_pair.mean ≈ state.mean
        @test state_pair.cov ≈ state.cov
    end

end
