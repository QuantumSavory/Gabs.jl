##
# Predefined Gaussian channels
##

function displace(::Type{Td}, ::Type{Tt}, basis::SymplecticBasis{N}, alpha::A, noise::M) where {Td,Tt,N<:Int,A,M}
    disp, transform = _displace(basis, alpha)
    return GaussianChannel(basis, Td(disp), Tt(transform), Tt(noise))
end
displace(::Type{T}, basis::SymplecticBasis{N}, alpha::A, noise::M) where {T,N<:Int,A,M} = displace(T, T, basis, alpha, noise)
function displace(basis::SymplecticBasis{N}, alpha::A, noise::M) where {N<:Int,A,M}
    disp, transform = _displace(basis, alpha)
    return GaussianChannel(basis, disp, transform, noise)
end

function squeeze(::Type{Td}, ::Type{Tt}, basis::SymplecticBasis{N}, r::R, theta::R, noise::M) where {Td,Tt,N<:Int,R,M}
    disp, transform = _squeeze(basis, r, theta)
    return GaussianChannel(basis, Td(disp), Tt(transform), Tt(noise))
end
squeeze(::Type{T}, basis::SymplecticBasis{N}, r::R, theta::R, noise::M) where {T,N<:Int,R,M} = squeeze(T, T, basis, r, theta, noise)
function squeeze(basis::SymplecticBasis{N}, r::R, theta::R, noise::M) where {N<:Int,R,M}
    disp, transform = _squeeze(basis, r, theta)
    return GaussianChannel(basis, disp, transform, noise)
end

function twosqueeze(::Type{Td}, ::Type{Tt}, basis::SymplecticBasis{N}, r::R, theta::R, noise::M) where {Td,Tt,N<:Int,R,M}
    disp, transform = _twosqueeze(basis, r, theta)
    return GaussianChannel(basis, Td(disp), Tt(transform), Tt(noise))
end
twosqueeze(::Type{T}, basis::SymplecticBasis{N}, r::R, theta::R, noise::M) where {T,N<:Int,R,M} = twosqueeze(T, T, basis, r, theta, noise)
function twosqueeze(basis::SymplecticBasis{N}, r::R, theta::R, noise::M) where {N<:Int,R,M}
    disp, transform = _twosqueeze(basis, r, theta)
    return GaussianChannel(basis, disp, transform, noise)
end

function phaseshift(::Type{Td}, ::Type{Tt}, basis::SymplecticBasis{N}, theta::R, noise::M) where {Td,Tt,N<:Int,R,M}
    disp, transform = _phaseshift(basis, theta)
    return GaussianChannel(basis, Td(disp), Tt(transform), Tt(noise))
end
phaseshift(::Type{T}, basis::SymplecticBasis{N}, theta::R, noise::M) where {T,N<:Int,R,M} = phaseshift(T, T, basis, theta, noise)
function phaseshift(basis::SymplecticBasis{N}, theta::R, noise::M) where {N<:Int,R,M}
    disp, transform = _phaseshift(basis, theta)
    return GaussianChannel(basis, disp, transform, noise)
end

function beamsplitter(::Type{Td}, ::Type{Tt}, basis::SymplecticBasis{N}, transmit::R, noise::M) where {Td,Tt,N<:Int,R,M}
    disp, transform = _beamsplitter(basis, transmit)
    return GaussianChannel(basis, Td(disp), Tt(transform), Tt(noise))
end
beamsplitter(::Type{T}, basis::SymplecticBasis{N}, transmit::R, noise::M) where {T,N<:Int,R,M} = beamsplitter(T, T, basis, transmit, noise)
function beamsplitter(basis::SymplecticBasis{N}, transmit::R, noise::M) where {N<:Int,R,M}
    disp, transform = _beamsplitter(basis, transmit)
    return GaussianChannel(basis, disp, transform, noise)
end

"""
    attenuator([Td=Vector{Float64}, Tt=Matrix{Float64},] basis::SymplecticBasis, theta<:Real, n<:Int)

Gaussian channel describing the coupling of an input
single mode Gaussian state and its environment via a beam splitter operation. The channel is paramatrized
by beam splitter rotation angle `theta` and thermal noise `n`.

## Mathematical description of an attenuator channel

An attenuator channel, `E(θ, nₜₕ)`, where `θ` is
the beam splitter rotation parameter and `nₜₕ ≥ 1` is the thermal noise parameter,
is characterized by the zero displacement vector, transformation matrix `cos(θ)I`,
and noise matrix `nₜₕsin²(θ)I`.

## Example

```jldoctest
julia> attenuator(QuadPairBasis(1), pi/6, 3)
GaussianChannel for 1 mode.
  symplectic basis: QuadPairBasis
displacement: 2-element Vector{Float64}:
 0.0
 0.0
transform: 2×2 Matrix{Float64}:
 0.866025  0.0
 0.0       0.866025
noise: 2×2 Matrix{Float64}:
 0.75  0.0
 0.0   0.75
```
"""
function attenuator(::Type{Td}, ::Type{Tt}, basis::SymplecticBasis{N}, theta::R, n::M; ħ = 2) where {Td,Tt,N<:Int,R,M}
    disp, transform, noise = _attenuator(basis, theta, n)
    return GaussianChannel(basis, Td(disp), Tt(transform), Tt(noise); ħ = ħ)
end
attenuator(::Type{T}, basis::SymplecticBasis{N}, theta::R, n::M; ħ = 2) where {T,N<:Int,R,M} = attenuator(T, T, basis, theta, n; ħ = ħ)
function attenuator(basis::SymplecticBasis{N}, theta::R, n::M; ħ = 2) where {N<:Int,R,M}
    disp, transform, noise = _attenuator(basis, theta, n)
    return GaussianChannel(basis, disp, transform, noise; ħ = ħ)
end
function _attenuator(basis::Union{QuadPairBasis{N},QuadBlockBasis{N}}, theta::R, n::M) where {N<:Int,R,M}
    nmodes = basis.nmodes
    disp = zeros(R, 2*nmodes)
    transform = Matrix{R}(cos(theta) * I, 2*nmodes, 2*nmodes)
    noise = Matrix{R}((sin(theta))^2 * n * I, 2*nmodes, 2*nmodes)
    return disp, transform, noise
end
function _attenuator(basis::QuadPairBasis{N}, theta::R, n::M) where {N<:Int,R<:Vector,M<:Vector}
    nmodes = basis.nmodes
    Rt = eltype(R)
    disp = zeros(Rt, 2*nmodes)
    transform = zeros(Rt, 2*nmodes, 2*nmodes)
    noise = zeros(Rt, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes)
        ct, st = cos(theta[i]), sin(theta[i])
        ni = n[i]

        transform[2*i-1, 2*i-1] = ct
        transform[2*i, 2*i] = ct

        noise[2*i-1, 2*i-1] = st^2 * ni
        noise[2*i, 2*i] = st^2 * ni
    end
    return disp, transform, noise
end
function _attenuator(basis::QuadBlockBasis{N}, theta::R, n::M) where {N<:Int,R<:Vector,M<:Vector}
    nmodes = basis.nmodes
    Rt = eltype(R)
    disp = zeros(Rt, 2*nmodes)
    transform = zeros(Rt, 2*nmodes, 2*nmodes)
    noise = zeros(Rt, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes)
        ct, st = cos(theta[i]), sin(theta[i])
        ni = n[i]

        transform[i, i] = ct
        transform[i+nmodes, i+nmodes] = ct

        noise[i, i] = st^2 * ni
        noise[i+nmodes, i+nmodes] = st^2 * ni
    end
    return disp, transform, noise
end

"""
    amplifier([Td=Vector{Float64}, Tt=Matrix{Float64},] basis::SymplecticBasis, r<:Real, n<:Int)

Gaussian channel describing the interaction of an input
single mode Gaussian state and its environment via a two-mode squeezing operation. The channel is paramatrized
by squeezing amplitude parameter `r` and thermal noise `n`.

## Mathematical description of an amplifier channel

An amplifier channel, `A(r, nₜₕ)`, where `r` is
the squeezing amplitude parameter and `nₜₕ ≥ 1` is the thermal noise parameter,
is characterized by the zero displacement vector, transformation matrix `cosh(r)I`,
and noise matrix `nₜₕsinh²(r)I`.

## Example

```jldoctest
julia> amplifier(QuadPairBasis(1), 2.0, 3)
GaussianChannel for 1 mode.
  symplectic basis: QuadPairBasis
displacement: 2-element Vector{Float64}:
 0.0
 0.0
transform: 2×2 Matrix{Float64}:
 3.7622  0.0
 0.0     3.7622
noise: 2×2 Matrix{Float64}:
 39.4623   0.0
  0.0     39.4623
```
"""
function amplifier(::Type{Td}, ::Type{Tt}, basis::SymplecticBasis{N}, r::R, n::M; ħ = 2) where {Td,Tt,N<:Int,R,M}
    disp, transform, noise = _amplifier(basis, r, n)
    return GaussianChannel(basis, Td(disp), Tt(transform), Tt(noise); ħ = ħ)
end
amplifier(::Type{T}, basis::SymplecticBasis{N}, r::R, n::M; ħ = 2) where {T,N<:Int,R,M} = amplifier(T, T, basis, r, n; ħ = ħ)
function amplifier(basis::SymplecticBasis{N}, r::R, n::M; ħ = 2) where {N<:Int,R,M}
    disp, transform, noise = _amplifier(basis, r, n)
    return GaussianChannel(basis, disp, transform, noise; ħ = ħ)
end
function _amplifier(basis::Union{QuadPairBasis{N},QuadBlockBasis{N}}, r::R, n::M) where {N<:Int,R,M}
    nmodes = basis.nmodes
    disp = zeros(R, 2*nmodes) 
    transform = Matrix{R}(cosh(r) * I, 2*nmodes, 2*nmodes)
    noise = Matrix{R}((sinh(r))^2 * n * I, 2*nmodes, 2*nmodes)
    return disp, transform, noise
end
function _amplifier(basis::QuadPairBasis{N}, r::R, n::M) where {N<:Int,R<:Vector,M<:Vector}
    nmodes = basis.nmodes
    Rt = eltype(R)
    disp = zeros(Rt, 2*nmodes) 
    transform = zeros(Rt, 2*nmodes, 2*nmodes)
    noise = zeros(Rt, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes)
        cr, sr = cosh(r[i]), sinh(r[i])
        ni = n[i]

        transform[2*i-1, 2*i-1] = cr
        transform[2*i, 2*i] = cr

        noise[2*i-1, 2*i-1] = sr^2 * ni
        noise[2*i, 2*i] = sr^2 * ni
    end
    return disp, transform, noise
end
function _amplifier(basis::QuadBlockBasis{N}, r::R, n::M) where {N<:Int,R<:Vector,M<:Vector}
    nmodes = basis.nmodes
    Rt = eltype(R)
    disp = zeros(Rt, 2*nmodes) 
    transform = zeros(Rt, 2*nmodes, 2*nmodes)
    noise = zeros(Rt, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes)
        cr, sr = cosh(r[i]), sinh(r[i])
        ni = n[i]

        transform[i, i] = cr
        transform[i+nmodes, i+nmodes] = cr

        noise[i, i] = sr^2 * ni
        noise[i+nmodes, i+nmodes] = sr^2 * ni
    end
    return disp, transform, noise
end

##
# Predefined operations on Gaussian channels
##

function tensor(::Type{Td}, ::Type{Tt}, op1::GaussianChannel, op2::GaussianChannel) where {Td,Tt}
    typeof(op1.basis) == typeof(op2.basis) || throw(ArgumentError(SYMPLECTIC_ERROR))
    op1.ħ == op2.ħ || throw(ArgumentError(HBAR_ERROR))
    disp, transform, noise = _tensor(op1, op2)
    return GaussianChannel(op1.basis ⊕ op2.basis, Td(disp), Tt(transform), Tt(noise); ħ = op1.ħ)
end
tensor(::Type{T}, op1::GaussianChannel, op2::GaussianChannel) where {T} = tensor(T, T, op1, op2)
function tensor(op1::GaussianChannel, op2::GaussianChannel)
    typeof(op1.basis) == typeof(op2.basis) || throw(ArgumentError(SYMPLECTIC_ERROR))
    op1.ħ == op2.ħ || throw(ArgumentError(HBAR_ERROR))
    disp, transform, noise = _tensor(op1, op2)
    return GaussianChannel(op1.basis ⊕ op2.basis, disp, transform, noise; ħ = op1.ħ)
end
function _tensor(op1::GaussianChannel{B1,D1,T1}, op2::GaussianChannel{B2,D2,T2}) where {B1<:QuadPairBasis,B2<:QuadPairBasis,D1,D2,T1,T2}
    return _directsumchannel(op1, op2, _tensorperm(op1.basis, op2.basis))
end
function _tensor(op1::GaussianChannel{B1,D1,T1}, op2::GaussianChannel{B2,D2,T2}) where {B1<:QuadBlockBasis,B2<:QuadBlockBasis,D1,D2,T1,T2}
    return _directsumchannel(op1, op2, _tensorperm(op1.basis, op2.basis))
end

# As `_directsummoments`, but a channel carries a noise matrix alongside its
# transform and both are relabelled the same way.
function _directsumchannel(op1::GaussianChannel, op2::GaussianChannel, perm)
    disp1, trans1, noise1 = op1.disp, op1.transform, op1.noise
    disp2, trans2, noise2 = op2.disp, op2.transform, op2.noise
    n1, n2 = length(disp1), length(disp2)
    d1, d2 = _codevice(disp1, disp2)
    disp′ = vcat(d1, d2)
    transform′ = _blockdiag(trans1, trans2, n1, n2)
    noise′ = _blockdiag(noise1, noise2, n1, n2)
    if perm !== nothing
        disp′ = disp′[perm]
        transform′ = transform′[perm, perm]
        noise′ = noise′[perm, perm]
    end
    disp′′ = _promote_output_vector(typeof(disp1), typeof(disp2), disp′)
    transform′′ = _promote_output_matrix(typeof(trans1), typeof(trans2), transform′)
    noise′′ = _promote_output_matrix(typeof(noise1), typeof(noise2), noise′)
    return disp′′, transform′′, noise′′
end

##
# Embedding Gaussian channels
##

"""
    embed(basis::SymplecticBasis, idx::Int, op::GaussianChannel)
    embed(basis::SymplecticBasis, indices::Vector{<:Int}, op::GaussianChannel)

Embed a smaller Gaussian channel into a larger symplectic basis. The channel
acts on the modes specified by `idx`/`indices` and leaves other modes unchanged.
"""
function embed(
    basis::QuadPairBasis, index::Int, op::GaussianChannel{<:QuadPairBasis,D,S}
) where {D,S}
    return embed(basis, [index], op)
end
function embed(
    basis::QuadPairBasis, indices::Vector{<:Int}, op::GaussianChannel{<:QuadPairBasis,D,S}
) where {D,S}
    return _embedchannel(basis, indices, op)
end
function embed(
    basis::QuadBlockBasis, index::Int, op::GaussianChannel{<:QuadBlockBasis,D,S}
) where {D,S}
    return embed(basis, [index], op)
end
function embed(
    basis::QuadBlockBasis, indices::Vector{<:Int}, op::GaussianChannel{<:QuadBlockBasis,D,S}
) where {D,S}
    return _embedchannel(basis, indices, op)
end

# As `_embedunitary`, with the noise matrix scattered onto a zero background:
# the untouched modes pick up no noise.
function _embedchannel(basis::SymplecticBasis, indices::Vector{<:Int}, op::GaussianChannel)
    @assert length(indices) == op.basis.nmodes "Number of indices must match number of modes in the channel"
    @assert basis.nmodes ≥ length(indices) "Target basis must be large enough"
    dim = 2 * basis.nmodes
    q = _quadindices(basis, indices)
    disp = similar(op.disp, dim)
    fill!(disp, zero(eltype(op.disp)))
    transform = similar(op.transform, dim, dim)
    fill!(transform, zero(eltype(op.transform)))
    transform[diagind(transform)] .= oneunit(eltype(op.transform))
    noise = similar(op.noise, dim, dim)
    fill!(noise, zero(eltype(op.noise)))
    disp[q] = op.disp
    transform[q, q] = op.transform
    noise[q, q] = op.noise
    return GaussianChannel(basis, disp, transform, noise; ħ = op.ħ)
end

"""
    changebasis(::SymplecticBasis, state::GaussianChannel)

Change the symplectic basis of a Gaussian channel.

# Example

```jldoctest
julia> ch = attenuator(QuadBlockBasis(2), [1.0, 2.0], [2, 4])
GaussianChannel for 2 modes.
  symplectic basis: QuadBlockBasis
displacement: 4-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
transform: 4×4 Matrix{Float64}:
 0.540302   0.0       0.0        0.0
 0.0       -0.416147  0.0        0.0
 0.0        0.0       0.540302   0.0
 0.0        0.0       0.0       -0.416147
noise: 4×4 Matrix{Float64}:
 1.41615  0.0      0.0      0.0
 0.0      3.30729  0.0      0.0
 0.0      0.0      1.41615  0.0
 0.0      0.0      0.0      3.30729

julia> changebasis(QuadPairBasis, ch)
GaussianChannel for 2 modes.
  symplectic basis: QuadPairBasis
displacement: 4-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
transform: 4×4 Matrix{Float64}:
 0.540302  0.0        0.0        0.0
 0.0       0.540302   0.0        0.0
 0.0       0.0       -0.416147   0.0
 0.0       0.0        0.0       -0.416147
noise: 4×4 Matrix{Float64}:
 1.41615  0.0      0.0      0.0
 0.0      1.41615  0.0      0.0
 0.0      0.0      3.30729  0.0
 0.0      0.0      0.0      3.30729
```
"""
function changebasis(::Type{B1}, op::GaussianChannel{B2,D,S}) where {B1<:QuadBlockBasis,B2<:QuadPairBasis,D,S}
    nmodes = op.basis.nmodes
    p = _basisperm(B1, nmodes)
    return GaussianChannel(B1(nmodes), _permute(op.disp, p), _permutesquare(op.transform, p), _permutesquare(op.noise, p); ħ = op.ħ)
end
function changebasis(::Type{B1}, op::GaussianChannel{B2,D,S}) where {B1<:QuadPairBasis,B2<:QuadBlockBasis,D,S}
    nmodes = op.basis.nmodes
    p = _basisperm(B1, nmodes)
    return GaussianChannel(B1(nmodes), _permute(op.disp, p), _permutesquare(op.transform, p), _permutesquare(op.noise, p); ħ = op.ħ)
end
changebasis(::Type{<:QuadBlockBasis}, op::GaussianChannel{<:QuadBlockBasis,D,S}) where {D,S} = op
changebasis(::Type{<:QuadPairBasis}, op::GaussianChannel{<:QuadPairBasis,D,S}) where {D,S} = op