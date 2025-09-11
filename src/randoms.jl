"""
    randstate([Tm=Vector{Float64}, Tc=Matrix{Float64},] basis::SymplecticBasis; pure=false)

Calculate a random Gaussian state in symplectic representation defined by `basis`.
"""
function randstate(::Type{Tm}, ::Type{Tc}, basis::SymplecticBasis{N}; pure = false, ħ = 2) where {Tm,Tc,N<:Int}
    mean, covar = _randstate(basis; pure = pure, ħ = ħ)
    return GaussianState(basis, Tm(mean), Tc(covar), ħ = ħ)
end
randstate(::Type{T}, basis::SymplecticBasis{N}; pure = false, ħ = 2) where {T,N<:Int} = randstate(T,T,basis; pure = pure, ħ = ħ)
function randstate(basis::SymplecticBasis{N}; pure = false, ħ = 2) where {N<:Int}
    mean, covar = _randstate(basis; pure = pure, ħ = ħ)
    return GaussianState(basis, mean, covar, ħ = ħ)
end
function _randstate(basis::QuadPairBasis{N}; pure = false, ħ = 2) where {N<:Int}
    nmodes = basis.nmodes
    mean = randn(2*nmodes)
    covar = zeros(2*nmodes, 2*nmodes)
    symp = randsymplectic(basis)
    # generate pure Gaussian state
    if pure
        mul!(covar, symp, symp')
        covar .*= (ħ/2)
        return mean, covar
    end
    # create buffer for matrix multiplication
    buf = zeros(2*nmodes, 2*nmodes)
    # William decomposition for mixed Gaussian states
    sympeigs = (ħ/2)*(abs.(rand(nmodes)) .+ 1.0)
    will = Diagonal(repeat(sympeigs, inner = 2))
    mul!(covar, symp, mul!(buf, will, symp'))
    return mean, covar
end
function _randstate(basis::QuadBlockBasis{N}; pure = false, ħ = 2) where {N<:Int}
    nmodes = basis.nmodes
    mean = randn(2*nmodes)
    covar = zeros(2*nmodes, 2*nmodes)
    symp = randsymplectic(basis)
    # generate pure Gaussian state
    if pure
        mul!(covar, symp, symp')
        covar .*= (ħ/2)
        return mean, covar
    end
    # create buffer for matrix multiplication
    buf = zeros(2*nmodes, 2*nmodes)
    # William decomposition for mixed Gaussian states
    sympeigs = (ħ/2)*(abs.(rand(nmodes)) .+ 1.0)
    will = Diagonal(repeat(sympeigs, outer = 2))
    mul!(covar, symp, mul!(buf, will, symp'))
    return mean, covar
end

"""
    randunitary([Td=Vector{Float64}, Ts=Matrix{Float64},] basis::SymplecticBasis; passive=false)

Calculate a random Gaussian unitary operator in symplectic representation defined by `basis`.
"""
function randunitary(::Type{Td}, ::Type{Ts}, basis::SymplecticBasis{N}; passive = false, ħ = 2) where {Td,Ts,N<:Int}
    disp, symp = _randunitary(basis, passive = passive)
    return GaussianUnitary(basis, Td(disp), Ts(symp), ħ = ħ)
end
randunitary(::Type{T}, basis::SymplecticBasis{N}; passive = false, ħ = 2) where {T,N<:Int} = randunitary(T,T,basis; passive = passive, ħ = ħ)
function randunitary(basis::SymplecticBasis{N}; passive = false, ħ = 2) where {N<:Int}
    disp, symp = _randunitary(basis, passive = passive)
    return GaussianUnitary(basis, disp, symp, ħ = ħ)
end
function _randunitary(basis::SymplecticBasis{N}; passive = false) where {N<:Int}
    nmodes = basis.nmodes
    disp = randn(2*nmodes)
    symp = randsymplectic(basis, passive = passive)
    return disp, symp
end

"""
    randchannel([Td=Vector{Float64}, Tt=Matrix{Float64},] basis::SymplecticBasis)

Calculate a random Gaussian channel in symplectic representation defined by `basis`.
"""
function randchannel(::Type{Td}, ::Type{Tt}, basis::SymplecticBasis{N}; ħ = 2) where {Td,Tt,N<:Int}
    disp, transform, noise = _randchannel(basis)
    return GaussianChannel(basis, Td(disp), Tt(transform), Tt(noise), ħ = ħ)
end
randchannel(::Type{T}, basis::SymplecticBasis{N}; ħ = 2) where {T,N<:Int} = randchannel(T,T,basis; ħ = ħ)
function randchannel(basis::SymplecticBasis{N}; ħ = 2) where {N<:Int}
    disp, transform, noise = _randchannel(basis)
    return GaussianChannel(basis, disp, transform, noise, ħ = ħ)
end
function _randchannel(basis::SymplecticBasis{N}) where {N<:Int}
    nmodes = basis.nmodes
    disp = randn(2*nmodes)
    # generate symplectic matrix describing the evolution of the system with N modes
    # and environment with 2N modes
    symp = randsymplectic(typeof(basis)(3*nmodes))
    transform, B = symp[1:2*nmodes, 1:2*nmodes], @view(symp[1:2*nmodes, 2*nmodes+1:6*nmodes])
    # generate noise matrix from evolution of environment
    noise = zeros(2*nmodes, 2*nmodes)
    mul!(noise, B, B')
    return disp, transform, noise
end

"""
    randsymplectic([T=Matrix{Float64},] basis::SymplecticBasis, passive=false)

Calculate a random symplectic matrix in symplectic representation defined by `basis`.
"""
function randsymplectic(::Type{T}, basis::SymplecticBasis{N}; passive = false) where {T, N<:Int} 
    symp = randsymplectic(basis, passive = passive)
    return T(symp)
end
randsymplectic(basis::SymplecticBasis{N}; passive = false) where {N<:Int} = _randsymplectic(basis, passive = passive)
function _randsymplectic(basis::QuadPairBasis{N}; passive = false) where {N<:Int}
    nmodes = basis.nmodes
    # generate random orthogonal symplectic matrix
    O = _rand_orthogonal_symplectic(basis)
    if passive
        return O
    end
    # instead generate random active symplectic matrix
    O′ = _rand_orthogonal_symplectic(basis)
    # direct sum of symplectic matrices for single-mode squeeze transformations
    rs = rand(nmodes)
    squeezes = zeros(2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes)
        val = rs[i]
        squeezes[2*i-1, 2*i-1] = val
        squeezes[2*i, 2*i] = 1/val
    end
    return O * squeezes * O′
end
function _randsymplectic(basis::QuadBlockBasis{N}; passive = false) where {N<:Int}
    nmodes = basis.nmodes
    # generate random orthogonal symplectic matrix
    O = _rand_orthogonal_symplectic(basis)
    if passive
        return O
    end
    # instead generate random active symplectic matrix
    O′ = _rand_orthogonal_symplectic(basis)
    # direct sum of symplectic matrices for single-mode squeeze transformations
    rs = rand(nmodes)
    squeezes = zeros(2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes)
        val = rs[i]
        squeezes[i, i] = val
        squeezes[i+nmodes, i+nmodes] = 1/val
    end
    return O * squeezes * O′
end


# Generates random orthogonal symplectic matrix by blocking real
# and imaginary parts of a random unitary matrix
function _rand_orthogonal_symplectic(basis::QuadPairBasis{N}) where {N<:Int}
    nmodes = basis.nmodes
    U = _rand_unitary(basis)
    O = zeros(2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes), j in Base.OneTo(nmodes)
        val = U[i,j]
        O[2*i-1,2*j-1] = real(val)
        O[2*i, 2*j-1] = -imag(val)
        O[2*i-1, 2*j] = imag(val)
        O[2*i, 2*j] = real(val)
    end
    return O
end
function _rand_orthogonal_symplectic(basis::QuadBlockBasis{N}) where {N<:Int}
    nmodes = basis.nmodes
    U = _rand_unitary(basis)
    O = zeros(2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes), j in Base.OneTo(nmodes)
        val = U[i,j]
        O[i,j] = real(val)
        O[i+nmodes, j] = -imag(val)
        O[i, j+nmodes] = imag(val)
        O[i+nmodes, j+nmodes] = real(val)
    end
    return O
end
# Generates unitary matrix randomly distributed over Haar measure;
# see https://arxiv.org/abs/math-ph/0609050 for algorithm.
# This approach is faster and creates less allocations than rand(Haar(2), nmodes) from RandomMatrices.jl
function _rand_unitary(basis::SymplecticBasis{N}) where {N<:Int}
    nmodes = basis.nmodes
    M = rand(ComplexF64, nmodes, nmodes) ./ sqrt(2.0)
    q, r = qr(M)
    d = Diagonal([r[i, i] / abs(r[i, i]) for i in Base.OneTo(nmodes)])
    return q * d
end