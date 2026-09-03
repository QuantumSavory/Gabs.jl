@testitem "express formalism conversions" begin
    using Gabs
    
    @testset "GaussianState in QuadPairBasis" begin
        state = vacuumstate(QuadPairBasis(2))
        @test state.basis isa QuadPairBasis
        
        # Express to same basis type should return same/equivalent state
        state_pair = express(state, QuadPairBasis(2))
        @test state_pair.basis isa QuadPairBasis
        @test state_pair.mean ≈ state.mean
        @test state_pair.covar ≈ state.covar
    end
    
    @testset "GaussianState conversion to QuadBlockBasis" begin
        state = vacuumstate(QuadPairBasis(2))
        state_block = express(state, QuadBlockBasis(2))
        @test state_block.basis isa QuadBlockBasis
        @test nmodes(state_block.basis) == 2
        
        # Convert back to QuadPairBasis and verify equivalence
        state_pair = express(state_block, QuadPairBasis(2))
        @test state_pair.basis isa QuadPairBasis
        @test state_pair.mean ≈ state.mean
        @test state_pair.covar ≈ state.covar
    end
    
    @testset "GaussianState identity express" begin
        state = coherentstate(QuadPairBasis(2), 1.0)
        state_id = express(state)
        @test state_id.basis === state.basis
        @test state_id.mean ≈ state.mean
        @test state_id.covar ≈ state.covar
    end
    
    @testset "Non-trivial states basis conversion" begin
        state = squeezedstate(QuadPairBasis(1), 0.5, 0.0)
        state_block = express(state, QuadBlockBasis(1))
        state_pair = express(state_block, QuadPairBasis(1))
        
        # Verify round-trip equivalence
        @test state_pair.mean ≈ state.mean
        @test state_pair.covar ≈ state.covar
    end
    
    @testset "GaussianUnitary conversion" begin
        unitary = phaseshift(QuadPairBasis(1), π/4)
        @test unitary.basis isa QuadPairBasis
        
        unitary_block = express(unitary, QuadBlockBasis(1))
        @test unitary_block.basis isa QuadBlockBasis
        
        # Verify properties are preserved
        @test unitary_block.disp ≈ express(unitary, QuadBlockBasis(1)).disp
    end
    
    @testset "GaussianChannel conversion" begin
        channel = attenuator(QuadPairBasis(1), 0.9, 1)
        @test channel.basis isa QuadPairBasis
        
        channel_block = express(channel, QuadBlockBasis(1))
        @test channel_block.basis isa QuadBlockBasis
        
        # Verify the channel properties
        @test nmodes(channel_block.basis) == 1
    end
    
    @testset "Multi-mode basis conversions" begin
        state = eprstate(QuadPairBasis(2), 0.5, 0.0)
        @test nmodes(state.basis) == 2
        
        state_block = express(state, QuadBlockBasis(2))
        @test nmodes(state_block.basis) == 2
        
        # Verify physical equivalence
        state_pair = express(state_block, QuadPairBasis(2))
        @test state_pair.mean ≈ state.mean
        @test state_pair.covar ≈ state.covar
    end

end
