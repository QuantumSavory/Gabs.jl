module QuantumOpticsBaseExt

using LinearAlgebra: I, normalize!
using Gabs
using QuantumInterface: QuantumOpticsRepr, UseAsState
import QuantumInterface: express
using QuantumOpticsBase

function _block_moments(state::Gabs.GaussianState{<:Gabs.QuadBlockBasis}, ::Type{T}) where {T}
    return Vector{T}(state.mean), Matrix{T}(state.covar)
end

function _block_moments(state::Gabs.GaussianState{<:Gabs.QuadPairBasis}, ::Type{T}) where {T}
    nmodes = Int(Gabs.nmodes(state))
    mean = Vector{T}(undef, 2 * nmodes)
    covar = Matrix{T}(undef, 2 * nmodes, 2 * nmodes)

    @inbounds for i in Base.OneTo(nmodes)
        mean[i] = state.mean[2 * i - 1]
        mean[nmodes + i] = state.mean[2 * i]

        for j in Base.OneTo(nmodes)
            covar[j, i] = state.covar[2 * j - 1, 2 * i - 1]
            covar[nmodes + j, i] = state.covar[2 * j, 2 * i - 1]
            covar[j, nmodes + i] = state.covar[2 * j - 1, 2 * i]
            covar[nmodes + j, nmodes + i] = state.covar[2 * j, 2 * i]
        end
    end

    return mean, covar
end

function _fock_basis(cutoff::Integer, nmodes::Integer)
    mode_basis = FockBasis(cutoff)
    return isone(nmodes) ? mode_basis : tensor(ntuple(_ -> mode_basis, nmodes)...)
end

function _vacuum_ket(cutoff::Integer, nmodes::Integer)
    mode_basis = FockBasis(cutoff)
    vacuum = fockstate(mode_basis, 0)
    return isone(nmodes) ? vacuum : tensor(ntuple(_ -> vacuum, nmodes)...)
end

function _creation_ops(cutoff::Integer, nmodes::Integer)
    mode_basis = FockBasis(cutoff)
    if isone(nmodes)
        return [create(mode_basis)]
    end

    basis = _fock_basis(cutoff, nmodes)
    return [embed(basis, i, create(mode_basis)) for i in 1:nmodes]
end

function _creation_generator(create_ops, pair_matrix, linear_terms)
    generator = zero(dense(create_ops[1] * create_ops[1]))

    @inbounds for j in eachindex(create_ops), i in eachindex(create_ops)
        generator += (pair_matrix[i, j] / 2) * dense(create_ops[i] * create_ops[j])
    end

    @inbounds for i in eachindex(create_ops)
        generator += linear_terms[i] * dense(create_ops[i])
    end

    return generator
end

function express(state::Gabs.GaussianState, repr::QuantumOpticsRepr, ::UseAsState)
    T = float(real(promote_type(eltype(state.mean), eltype(state.covar), typeof(state.ħ))))
    C = Complex{T}
    tol = sqrt(eps(T)) * max(Int(Gabs.nmodes(state)), 1)

    isapprox(Gabs.purity(state), one(T); atol=tol, rtol=tol) ||
        throw(ArgumentError("Only pure Gaussian states can be expressed as `QuantumOpticsBase.Ket`s."))

    nmodes = Int(Gabs.nmodes(state))
    mean, covar = _block_moments(state, T)

    q = mean[1:nmodes]
    p = mean[nmodes + 1:2 * nmodes]
    Vqq = covar[1:nmodes, 1:nmodes]
    Vqp = covar[1:nmodes, nmodes + 1:2 * nmodes]
    Vpq = covar[nmodes + 1:2 * nmodes, 1:nmodes]
    Vpp = covar[nmodes + 1:2 * nmodes, nmodes + 1:2 * nmodes]

    α = complex.(q, p) ./ sqrt(2 * T(state.ħ))
    identity_matrix = Matrix{C}(I, nmodes, nmodes)
    pair_matrix = complex.(Vqq - Vpp, Vqp + Vpq) ./ (2 * T(state.ħ))
    occupations = (complex.(Vqq + Vpp, Vqp - Vpq) .- T(state.ħ) * identity_matrix) ./
        (2 * T(state.ħ))

    pair_matrix = pair_matrix / (occupations + identity_matrix)
    pair_matrix = (pair_matrix + transpose(pair_matrix)) / 2
    linear_terms = α - pair_matrix * conj.(α)

    create_ops = _creation_ops(repr.cutoff, nmodes)
    generator = _creation_generator(create_ops, pair_matrix, linear_terms)
    ket = exp(generator) * _vacuum_ket(repr.cutoff, nmodes)
    return normalize!(ket)
end

end
