##
# Predefined Gaussian unitaries
##

"""
    displace([Tm=Vector{Float64}, Ts=Matrix{Float64}], basis::SymplecticBasis, alpha<:Number)
    displace([Tm=Vector{Float64}, Ts=Matrix{Float64}], basis::SymplecticBasis, alpha<:Number, noise::Ts)

Gaussian operator that displaces the vacuum state into a coherent state, known
as the displacement operator. The symplectic representation is given by `basis`. The complex amplitude is given by `alpha`. Noise can
be added to the operation with `noise`.

## Mathematical description of a displacement operator

A displacement operator `D(α)` is defined by the operation
`D(α)|0⟩ = |α⟩`, where `α` is a complex amplitude. The operator `D(α)` 
is characterized by the displacement vector `√2ħ [real(α), imag(α)]` 
and symplectic matrix `I`.

## Example

```jldoctest
julia> displace(QuadPairBasis(1), 1.0+im)
GaussianUnitary for 1 mode.
  symplectic basis: QuadPairBasis
displacement: 2-element Vector{Float64}:
 2.0
 2.0
symplectic: 2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0
```
"""
function displace(::Type{Td}, ::Type{Ts}, basis::SymplecticBasis{N}, alpha::A; ħ = 2) where {Td,Ts,N<:Int,A}
    disp, symplectic = _displace(basis, alpha; ħ = ħ)
    return GaussianUnitary(basis, Td(disp), Ts(symplectic); ħ = ħ)
end
displace(::Type{T}, basis::SymplecticBasis{N}, alpha::A; ħ = 2) where {T,N<:Int,A} = displace(T, T, basis, alpha; ħ = ħ)
function displace(basis::SymplecticBasis{N}, alpha::A; ħ = 2) where {N<:Int,A}
    disp, symplectic = _displace(basis, alpha; ħ = ħ)
    return GaussianUnitary(basis, disp, symplectic; ħ = ħ)
end
function _displace(basis::QuadPairBasis{N}, alpha::A; ħ = 2) where {N<:Int,A<:Number}
    nmodes = basis.nmodes
    Rt = real(eltype(A))
    disp = repeat([sqrt(2*ħ) * real(alpha), sqrt(2*ħ) * imag(alpha)], nmodes)
    symplectic = Matrix{Rt}(I, 2*nmodes, 2*nmodes)
    return disp, symplectic
end
function _displace(basis::QuadPairBasis{N}, alpha::A; ħ = 2) where {N<:Int,A<:Vector}
    nmodes = basis.nmodes
    Rt = real(eltype(A))
    disp = collect(Iterators.flatten((real(i), imag(i)) for i in alpha))
    disp .*= sqrt(2*ħ)
    symplectic = Matrix{Rt}(I, 2*nmodes, 2*nmodes)
    return disp, symplectic
end
function _displace(basis::QuadBlockBasis{N}, alpha::A; ħ = 2) where {N<:Int,A<:Number}
    nmodes = basis.nmodes
    Rt = real(eltype(A))
    disp = repeat([sqrt(2*ħ) * real(alpha), sqrt(2*ħ) * imag(alpha)], inner = nmodes)
    symplectic = Matrix{Rt}(I, 2*nmodes, 2*nmodes)
    return disp, symplectic
end
function _displace(basis::QuadBlockBasis{N}, alpha::A; ħ = 2) where {N<:Int,A<:Vector}
    nmodes = basis.nmodes
    Rt = real(eltype(A))
    re = collect(Iterators.flatten((real(i), imag(i)) for i in alpha))
    disp = vcat(@view(re[1:2:end]), @view(re[2:2:end]))
    disp .*= sqrt(2*ħ)
    symplectic = Matrix{Rt}(I, 2*nmodes, 2*nmodes)
    return disp, symplectic
end

"""
    squeeze([Tm=Vector{Float64}, Ts=Matrix{Float64}], basis::SymplecticBasis, r<:Real, theta<:Real)
    squeeze([Tm=Vector{Float64}, Ts=Matrix{Float64}], basis::SymplecticBasis, r<:Real, theta<:Real, noise::Ts)

Gaussian operator that squeezes the vacuum state into a squeezed state, known
as the squeezing operator. The symplectic representation is given by `basis`. The amplitude and phase squeezing parameters 
are given by `r` and `theta`, respectively. Noise can be added to the operation
with `noise`.

## Mathematical description of a squeezing operator

A squeeze operator `S(r, θ)` is defined by the operation
`S(r, θ)|0⟩ = |r, θ⟩`, where `r` and `θ` are the real amplitude and phase parameters, 
respectively. The operator `S(r, θ)` is characterized by 
the zero displacement vector and symplectic
matrix `cosh(r)I - sinh(r)R(θ)`, where `R(θ)` is the rotation matrix.

## Example

```jldoctest
julia> squeeze(QuadPairBasis(1), 0.25, pi/4)
GaussianUnitary for 1 mode.
  symplectic basis: QuadPairBasis
displacement: 2-element Vector{Float64}:
 0.0
 0.0
symplectic: 2×2 Matrix{Float64}:
  0.852789  -0.178624
 -0.178624   1.21004
```
"""
function squeeze(::Type{Td}, ::Type{Ts}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {Td,Ts,N<:Int,R}
    disp, symplectic = _squeeze(basis, r, theta)
    return GaussianUnitary(basis, Td(disp), Ts(symplectic); ħ = ħ)
end
squeeze(::Type{T}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {T,N<:Int,R} = squeeze(T, T, basis, r, theta; ħ = ħ)
function squeeze(basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {N<:Int, R}
    disp, symplectic = _squeeze(basis, r, theta)
    return GaussianUnitary(basis, disp, symplectic; ħ = ħ)
end
function _squeeze(basis::QuadPairBasis{N}, r::R, theta::R) where {N<:Int,R<:Real}
    nmodes = basis.nmodes
    Rt = real(eltype(R))
    disp = zeros(R, 2*nmodes)
    cr, sr = cosh(r), sinh(r)
    ct, st = cos(theta), sin(theta)
    symplectic = zeros(R, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes)
        symplectic[2*i-1, 2*i-1] = cr - sr*ct
        symplectic[2*i-1, 2*i] = -sr * st
        symplectic[2*i, 2*i-1] = -sr * st
        symplectic[2*i, 2*i] = cr + sr*ct
    end
    return disp, symplectic
end
function _squeeze(basis::QuadPairBasis{N}, r::R, theta::R) where {N<:Int,R<:Vector}
    nmodes = basis.nmodes
    Rt = eltype(R)
    disp = zeros(Rt, 2*nmodes)
    symplectic = zeros(Rt, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes)
        cr, sr = cosh(r[i]), sinh(r[i])
        ct, st = cos(theta[i]), sin(theta[i])
        symplectic[2*i-1, 2*i-1] = cr - sr*ct
        symplectic[2*i-1, 2*i] = -sr * st
        symplectic[2*i, 2*i-1] = -sr * st
        symplectic[2*i, 2*i] = cr + sr*ct
    end
    return disp, symplectic
end
function _squeeze(basis::QuadBlockBasis{N}, r::R, theta::R) where {N<:Int,R<:Real}
    nmodes = basis.nmodes
    Rt = real(eltype(R))
    disp = zeros(R, 2*nmodes)
    cr, sr = cosh(r), sinh(r)
    ct, st = cos(theta), sin(theta)
    symplectic = zeros(R, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes)
        symplectic[i, i] = cr - sr*ct
        symplectic[i, i+nmodes] = -sr * st
        symplectic[i+nmodes, i] = -sr * st
        symplectic[i+nmodes, i+nmodes] = cr + sr*ct
    end
    return disp, symplectic
end
function _squeeze(basis::QuadBlockBasis{N}, r::R, theta::R) where {N<:Int,R<:Vector}
    nmodes = basis.nmodes
    Rt = eltype(R)
    disp = zeros(Rt, 2*nmodes)
    symplectic = zeros(Rt, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes)
        cr, sr = cosh(r[i]), sinh(r[i])
        ct, st = cos(theta[i]), sin(theta[i])
        symplectic[i, i] = cr - sr*ct
        symplectic[i, i+nmodes] = -sr * st
        symplectic[i+nmodes, i] = -sr * st
        symplectic[i+nmodes, i+nmodes] = cr + sr*ct
    end
    return disp, symplectic
end

"""
    twosqueeze([Tm=Vector{Float64}, Ts=Matrix{Float64}], basis::SymplecticBasis, r<:Real, theta<:Real)
    twosqueeze([Tm=Vector{Float64}, Ts=Matrix{Float64}], basis::SymplecticBasis, r<:Real, theta<:Real, noise::Ts)

Gaussian operator that squeezes a two-mode vacuum state into a two-mode squeezed state,
known as the two-mode squeezing operator. The symplectic representation is given by `basis`. The amplitude and phase squeezing parameters 
are given by `r` and `theta`, respectively. Noise can be added to the operation
with `noise`.

## Mathematical description of a two-mode squeezing operator

A two-mode squeeze operator `S₂(r, θ)` is defined by the operation
`S₂(r, θ)|0⟩ = |r, θ⟩`, where `r` and `θ`
are the real amplitude and phase parameters, respectively. The operator 
`S₂(r, θ)` is characterized by the zero displacement vector and symplectic
matrix `[cosh(r)I -sinh(r)R(θ); -sinh(r)R(θ) cosh(r)I]`, where `R(θ)` is the rotation matrix.

## Example

```jldoctest
julia> twosqueeze(QuadPairBasis(2), 0.25, pi/4)
GaussianUnitary for 2 modes.
  symplectic basis: QuadPairBasis
displacement: 4-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
symplectic: 4×4 Matrix{Float64}:
  1.03141    0.0       -0.178624  -0.178624
  0.0        1.03141   -0.178624   0.178624
 -0.178624  -0.178624   1.03141    0.0
 -0.178624   0.178624   0.0        1.03141
```
"""
function twosqueeze(::Type{Td}, ::Type{Ts}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {Td,Ts,N<:Int,R}
    disp, symplectic = _twosqueeze(basis, r, theta)
    return GaussianUnitary(basis, Td(disp), Ts(symplectic); ħ = ħ)
end
twosqueeze(::Type{T}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {T,N<:Int,R} = twosqueeze(T, T, basis, r, theta; ħ = ħ)
function twosqueeze(basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {N<:Int,R}
    disp, symplectic = _twosqueeze(basis, r, theta)
    return GaussianUnitary(basis, disp, symplectic; ħ = ħ)
end
function _twosqueeze(basis::QuadPairBasis{N}, r::R, theta::R) where {N<:Int,R<:Real}
    nmodes = basis.nmodes
    disp = zeros(R, 2*nmodes)
    cr, sr = cosh(r), sinh(r)
    ct, st = cos(theta), sin(theta)
    symplectic = zeros(R, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(Int(nmodes/2))
        symplectic[4*i-3, 4*i-3] = cr
        symplectic[4*i-3, 4*i-1] = -sr * ct
        symplectic[4*i-3, 4*i] = -sr * st

        symplectic[4*i-2, 4*i-2] = cr
        symplectic[4*i-2, 4*i-1] = -sr * st
        symplectic[4*i-2, 4*i] = sr * ct

        symplectic[4*i-1, 4*i-3] = -sr * ct
        symplectic[4*i-1, 4*i-2] = -sr * st
        symplectic[4*i-1, 4*i-1] = cr

        symplectic[4*i, 4*i-3] = -sr * st
        symplectic[4*i, 4*i-2] = sr * ct
        symplectic[4*i, 4*i] = cr
    end
    return disp, symplectic
end
function _twosqueeze(basis::QuadPairBasis{N}, r::R, theta::R) where {N<:Int,R<:Vector}
    nmodes = basis.nmodes
    Rt = eltype(R)
    disp = zeros(Rt, 2*nmodes)
    symplectic = zeros(Rt, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(Int(nmodes/2))
        cr, sr = cosh(r[i]), sinh(r[i])
        ct, st = cos(theta[i]), sin(theta[i])

        symplectic[4*i-3, 4*i-3] = cr
        symplectic[4*i-3, 4*i-1] = -sr * ct
        symplectic[4*i-3, 4*i] = -sr * st

        symplectic[4*i-2, 4*i-2] = cr
        symplectic[4*i-2, 4*i-1] = -sr * st
        symplectic[4*i-2, 4*i] = sr * ct

        symplectic[4*i-1, 4*i-3] = -sr * ct
        symplectic[4*i-1, 4*i-2] = -sr * st
        symplectic[4*i-1, 4*i-1] = cr

        symplectic[4*i, 4*i-3] = -sr * st
        symplectic[4*i, 4*i-2] = sr * ct
        symplectic[4*i, 4*i] = cr
    end
    return disp, symplectic
end
function _twosqueeze(basis::QuadBlockBasis{N}, r::R, theta::R) where {N<:Int,R<:Real}
    nmodes = basis.nmodes
    disp = zeros(R, 2*nmodes)
    cr, sr = cosh(r), sinh(r)
    ct, st = cos(theta), sin(theta)
    symplectic = zeros(R, 2*nmodes, 2*nmodes)
    for i in Base.OneTo(Int(nmodes/2))
        symplectic[2*i-1, 2*i-1] = cr
        symplectic[2*i-1, 2*i] = -sr * ct
        symplectic[2*i, 2*i-1] = -sr * ct
        symplectic[2*i, 2*i] = cr

        symplectic[2*i-1, 2*i+nmodes] = -sr * st
        symplectic[2*i, 2*i+nmodes-1] = -sr * st

        symplectic[2*i+nmodes-1, 2*i+nmodes-1] = cr
        symplectic[2*i+nmodes-1, 2*i+nmodes] = sr * ct
        symplectic[2*i+nmodes, 2*i+nmodes-1] = sr * ct
        symplectic[2*i+nmodes, 2*i+nmodes] = cr

        symplectic[2*i+nmodes-1, 2*i] = -sr * st
        symplectic[2*i+nmodes, 2*i-1] = -sr * st
    end
    return disp, symplectic
end
function _twosqueeze(basis::QuadBlockBasis{N}, r::R, theta::R) where {N<:Int,R<:Vector}
    nmodes = basis.nmodes
    Rt = eltype(R)
    disp = zeros(Rt, 2*nmodes)
    symplectic = zeros(Rt, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(Int(nmodes/2))
        cr, sr = cosh(r[i]), sinh(r[i])
        ct, st = cos(theta[i]), sin(theta[i])

        symplectic[2*i-1, 2*i-1] = cr
        symplectic[2*i-1, 2*i] = -sr * ct
        symplectic[2*i, 2*i-1] = -sr * ct
        symplectic[2*i, 2*i] = cr

        symplectic[2*i-1, 2*i+nmodes] = -sr * st
        symplectic[2*i, 2*i+nmodes-1] = -sr * st

        symplectic[2*i+nmodes-1, 2*i+nmodes-1] = cr
        symplectic[2*i+nmodes-1, 2*i+nmodes] = sr * ct
        symplectic[2*i+nmodes, 2*i+nmodes-1] = sr * ct
        symplectic[2*i+nmodes, 2*i+nmodes] = cr

        symplectic[2*i+nmodes-1, 2*i] = -sr * st
        symplectic[2*i+nmodes, 2*i-1] = -sr * st
    end
    return disp, symplectic
end

"""
    phaseshift([Tm=Vector{Float64}, Ts=Matrix{Float64}], basis::SymplecticBasis, theta<:Real)
    phaseshift([Tm=Vector{Float64}, Ts=Matrix{Float64}], basis::SymplecticBasis, theta<:Real, noise::Ts)

Gaussian operator that rotates the phase of a given Gaussian mode by `theta`,
as the phase shift operator. The symplectic representation is given by `basis`. Noise can be added to the operation
with `noise`.

## Mathematical description of a phase shift operator

A phase shift operator is defined by the operation
`U(θ) = exp(-iθâᵗâ)`, where `θ` is
the phase parameter, and `âᵗ` and `â` are the raising
and lowering operators, respectively. The operator `U(θ)` is characterized by 
the zero displacement vector and symplectic
matrix `[cos(θ) sin(θ); -sin(θ) cos(θ)]`.

## Example

```jldoctest
julia> phaseshift(QuadPairBasis(1), 3pi/4)
GaussianUnitary for 1 mode.
  symplectic basis: QuadPairBasis
displacement: 2-element Vector{Float64}:
 0.0
 0.0
symplectic: 2×2 Matrix{Float64}:
 -0.707107   0.707107
 -0.707107  -0.707107
```
"""
function phaseshift(::Type{Td}, ::Type{Ts}, basis::SymplecticBasis{N}, theta::R; ħ = 2) where {Td,Ts,N<:Int,R}
    disp, symplectic = _phaseshift(basis, theta)
    return GaussianUnitary(basis, Td(disp), Ts(symplectic); ħ = ħ)
end
phaseshift(::Type{T}, basis::SymplecticBasis{N}, theta::R; ħ = 2) where {T,N<:Int,R} = phaseshift(T, T, basis, theta; ħ = ħ)
function phaseshift(basis::SymplecticBasis{N}, theta::R; ħ = 2) where {N<:Int,R}
    disp, symplectic = _phaseshift(basis, theta)
    return GaussianUnitary(basis, disp, symplectic; ħ = ħ)
end
function _phaseshift(basis::QuadPairBasis{N}, theta::R) where {N<:Int,R<:Real}
    nmodes = basis.nmodes
    disp = zeros(R, 2*nmodes)
    ct, st = cos(theta), sin(theta)
    symplectic = zeros(R, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes)
        symplectic[2*i-1, 2*i-1] = ct
        symplectic[2*i-1, 2*i] = st
        symplectic[2*i, 2*i-1] = -st
        symplectic[2*i, 2*i] = ct
    end
    return disp, symplectic
end
function _phaseshift(basis::QuadPairBasis{N}, theta::R) where {N<:Int,R<:Vector}
    nmodes = basis.nmodes
    Rt = eltype(R)
    disp = zeros(Rt, 2*nmodes)
    symplectic = zeros(Rt, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes)
        ct, st = cos(theta[i]), sin(theta[i])
        symplectic[2*i-1, 2*i-1] = ct
        symplectic[2*i-1, 2*i] = st
        symplectic[2*i, 2*i-1] = -st
        symplectic[2*i, 2*i] = ct
    end
    return disp, symplectic
end
function _phaseshift(basis::QuadBlockBasis{N}, theta::R) where {N<:Int,R<:Real}
    nmodes = basis.nmodes
    disp = zeros(R, 2*nmodes)
    ct, st = cos(theta), sin(theta)
    symplectic = zeros(R, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes)
        symplectic[i, i] = ct
        symplectic[i, i+nmodes] = st
        symplectic[i+nmodes, i] = -st
        symplectic[i+nmodes, i+nmodes] = ct
    end
    return disp, symplectic
end
function _phaseshift(basis::QuadBlockBasis{N}, theta::R) where {N<:Int,R<:Vector}
    nmodes = basis.nmodes
    Rt = eltype(R)
    disp = zeros(Rt, 2*nmodes)
    symplectic = zeros(Rt, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes)
        ct, st = cos(theta[i]), sin(theta[i])
        symplectic[i, i] = ct
        symplectic[i, i+nmodes] = st
        symplectic[i+nmodes, i] = -st
        symplectic[i+nmodes, i+nmodes] = ct
    end
    return disp, symplectic
end

"""
    beamsplitter([Tm=Vector{Float64}, Ts=Matrix{Float64}], basis::SymplecticBasis, transmit<:Real)
    beamsplitter([Tm=Vector{Float64}, Ts=Matrix{Float64}], basis::SymplecticBasis, transmit<:Real, noise::Ts)

Gaussian operator that serves as the beam splitter transformation of a
two-mode Gaussian state, known as the beam splitter operator. The symplectic representation is given
by `basis`. The transmittivity of the operator is given by `transmit`. Noise can be added to the operation
with `noise`.

## Mathematical description of a beam splitter operator

A beam splitter operator `B(τ)` is defined by the operation
`B(τ) = exp(θ(âᵗb̂ - âb̂ᵗ))`, where `θ` is defined by `τ = cos²θ`, 
and `â` and `b̂` are the annihilation operators of the two modes, 
respectively. The operator `B(τ)` is characterized by 
the zero displacement vector and symplectic
matrix `[√τI √(1-τ)I; -√(1-τ)I √τI]`.

## Example

```jldoctest
julia> beamsplitter(QuadPairBasis(2), 0.75)
GaussianUnitary for 2 modes.
  symplectic basis: QuadPairBasis
displacement: 4-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
symplectic: 4×4 Matrix{Float64}:
  0.5        0.0       0.866025  0.0
  0.0        0.5       0.0       0.866025
 -0.866025   0.0       0.5       0.0
  0.0       -0.866025  0.0       0.5
```
"""
function beamsplitter(::Type{Td}, ::Type{Ts}, basis::SymplecticBasis{N}, transmit::R; ħ = 2) where {Td,Ts,N<:Int,R}
    disp, symplectic = _beamsplitter(basis, transmit)
    return GaussianUnitary(basis, Td(disp), Ts(symplectic); ħ = ħ)
end
beamsplitter(::Type{T}, basis::SymplecticBasis{N}, transmit::R; ħ = 2) where {T,N<:Int,R} = beamsplitter(T, T, basis, transmit; ħ = ħ)
function beamsplitter(basis::SymplecticBasis{N}, transmit::R; ħ = 2) where {N<:Int,R}
    disp, symplectic = _beamsplitter(basis, transmit)
    return GaussianUnitary(basis, disp, symplectic; ħ = ħ)
end
function _beamsplitter(basis::QuadPairBasis{N}, transmit::R) where {N<:Int,R<:Real}
    nmodes = basis.nmodes
    disp = zeros(R, 2*nmodes)
    a1, a2 = sqrt(transmit), sqrt(1 - transmit)
    symplectic = zeros(R, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(Int(nmodes/2))
        symplectic[4*i-3, 4*i-3] = a2
        symplectic[4*i-3, 4*i-1] = a1

        symplectic[4*i-2, 4*i-2] = a2
        symplectic[4*i-2, 4*i] = a1
    
        symplectic[4*i-1, 4*i-3] = -a1
        symplectic[4*i-1, 4*i-1] = a2

        symplectic[4*i, 4*i-2] = -a1
        symplectic[4*i, 4*i] = a2
    end
    return disp, symplectic
end
function _beamsplitter(basis::QuadPairBasis{N}, transmit::R) where {N<:Int,R<:Vector}
    nmodes = basis.nmodes
    Rt = eltype(R)
    disp = zeros(Rt, 2*nmodes)
    symplectic = zeros(Rt, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(Int(nmodes/2))
        a1, a2 = sqrt(transmit[i]), sqrt(1 - transmit[i])

        symplectic[4*i-3, 4*i-3] = a2
        symplectic[4*i-3, 4*i-1] = a1

        symplectic[4*i-2, 4*i-2] = a2
        symplectic[4*i-2, 4*i] = a1
    
        symplectic[4*i-1, 4*i-3] = -a1
        symplectic[4*i-1, 4*i-1] = a2

        symplectic[4*i, 4*i-2] = -a1
        symplectic[4*i, 4*i] = a2
    end
    return disp, symplectic
end
function _beamsplitter(basis::QuadBlockBasis{N}, transmit::R) where {N<:Int,R<:Real}
    nmodes = basis.nmodes
    disp = zeros(R, 2*nmodes)
    a1, a2 = sqrt(transmit), sqrt(1 - transmit)
    symplectic = zeros(R, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes)
        symplectic[2*i-1, 2*i-1] = a2
        symplectic[2*i-1, 2*i] = a1
        symplectic[2*i, 2*i-1] = -a1
        symplectic[2*i, 2*i] = a2
    end
    return disp, symplectic
end
function _beamsplitter(basis::QuadBlockBasis{N}, transmit::R) where {N<:Int,R<:Vector}
    nmodes = basis.nmodes
    Rt = eltype(R)
    disp = zeros(Rt, 2*nmodes)
    symplectic = zeros(Rt, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(Int(nmodes/2))
        a1, a2 = sqrt(transmit[i]), sqrt(1 - transmit[i])

        symplectic[2*i-1, 2*i-1] = a2
        symplectic[2*i-1, 2*i] = a1
        symplectic[2*i, 2*i-1] = -a1
        symplectic[2*i, 2*i] = a2

        symplectic[2*i+nmodes-1, 2*i+nmodes-1] = a2
        symplectic[2*i+nmodes-1, 2*i+nmodes] = a1
        symplectic[2*i+nmodes, 2*i+nmodes-1] = -a1
        symplectic[2*i+nmodes, 2*i+nmodes] = a2
    end
    return disp, symplectic
end

"""
Constructs the two-mode SUM gate as a Gaussian unitary operator.

```jldoctest
julia> sum_gate(QuadPairBasis(2))
GaussianUnitary for 2 modes.
  symplectic basis: QuadPairBasis
displacement: 4-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
symplectic: 4×4 Matrix{Float64}:
 1.0  0.0  0.0   0.0
 0.0  1.0  0.0  -1.0
 1.0  0.0  1.0   0.0
 0.0  0.0  0.0   1.0
```
"""
function sum_gate(::Type{Td}, ::Type{Ts}, basis::SymplecticBasis{N}; ħ = 2) where {Td, Ts, N}
    disp, symplectic = _sum_gate(basis)
    return GaussianUnitary(basis, Td(disp), Ts(symplectic); ħ = ħ)
end
function sum_gate(::Type{T}, basis::SymplecticBasis{N}; ħ = 2) where {T, N}
    return sum_gate(T, T, basis; ħ = ħ)
end
function sum_gate(basis::SymplecticBasis{N}; ħ = 2) where {N}
    disp, symplectic = _sum_gate(basis)
    return GaussianUnitary(basis, disp, symplectic; ħ = ħ)
end
function _sum_gate(basis::QuadBlockBasis{N}) where {N<:Int}
    nmodes = basis.nmodes
    if nmodes != 2
        error("SUM gate is defined for 2-mode systems only.")
    end
    disp = zeros(Float64, 2 * nmodes)
    symplectic = zeros(Float64, 2 * nmodes, 2 * nmodes)
    symplectic[1, 1] = 1.0
    symplectic[2, 1] = 1.0
    symplectic[2, 2] = 1.0
    symplectic[3, 3] = 1.0
    symplectic[3, 4] = -1.0
    symplectic[4, 4] = 1.0
    return disp, symplectic
end
function _sum_gate(basis::QuadPairBasis{N}) where {N<:Int}
    nmodes = basis.nmodes
    if nmodes != 2
        error("SUM gate is defined for 2-mode systems only.")
    end
    disp = zeros(Float64, 2 * nmodes)
    S = [1.0 0.0 0.0 0.0;
         1.0 1.0 0.0 0.0;
         0.0 0.0 1.0 -1.0;
         0.0 0.0 0.0 1.0]
    P = [1.0 0.0 0.0 0.0;
         0.0 0.0 1.0 0.0;
         0.0 1.0 0.0 0.0;
         0.0 0.0 0.0 1.0]
    symplectic = P * S * P'
    return disp, symplectic
end

##
# Operations on Gaussian unitaries
##

function tensor(::Type{Td}, ::Type{Ts}, op1::GaussianUnitary, op2::GaussianUnitary) where {Td,Ts}
    typeof(op1.basis) == typeof(op2.basis) || throw(ArgumentError(SYMPLECTIC_ERROR))
    op1.ħ == op2.ħ || throw(ArgumentError(HBAR_ERROR))
    disp, symplectic = _tensor(op1, op2)
    return GaussianUnitary(op1.basis ⊕ op2.basis, Td(disp), Ts(symplectic); ħ = op1.ħ)
end
tensor(::Type{T}, op1::GaussianUnitary, op2::GaussianUnitary) where {T} = tensor(T, T, op1, op2)
function tensor(op1::GaussianUnitary, op2::GaussianUnitary)
    typeof(op1.basis) == typeof(op2.basis) || throw(ArgumentError(SYMPLECTIC_ERROR))
    op1.ħ == op2.ħ || throw(ArgumentError(HBAR_ERROR))
    disp, symplectic = _tensor(op1, op2)
    return GaussianUnitary(op1.basis ⊕ op2.basis, disp, symplectic; ħ = op1.ħ)
end
function _tensor(op1::GaussianUnitary{B1,D1,S1}, op2::GaussianUnitary{B2,D2,S2}) where {B1<:QuadPairBasis,B2<:QuadPairBasis,D1,D2,S1,S2}
    basis1, basis2 = op1.basis, op2.basis
    nmodes1, nmodes2 = basis1.nmodes, basis2.nmodes
    nmodes = nmodes1 + nmodes2
    block1, block2 = Base.OneTo(2*nmodes1), Base.OneTo(2*nmodes2)
    # initialize direct sum of displacement vectors
    disp1, disp2 = op1.disp, op2.disp
    Dt = promote_type(eltype(disp1), eltype(disp2))
    disp′ = zeros(Dt, 2*nmodes)
    @inbounds for i in block1
        disp′[i] = disp1[i]
    end
    @inbounds for i in block2
        disp′[i+2*nmodes1] = disp2[i]
    end
    # initialize direct sum of symplectic matrices
    symp1, symp2 = op1.symplectic, op2.symplectic
    St = promote_type(eltype(symp1), eltype(symp2))
    symp′ = zeros(St, 2*nmodes, 2*nmodes)
    @inbounds for i in block1, j in block1
        symp′[i,j] = symp1[i,j]
    end
    @inbounds for i in block2, j in block2
        symp′[i+2*nmodes1,j+2*nmodes1] = symp2[i,j]
        symp′[i+2*nmodes1,j+2*nmodes1] = symp2[i,j]
    end
    # extract output array types
    disp′′ = _promote_output_vector(typeof(disp1), typeof(disp2), disp′)
    symp′′ = _promote_output_matrix(typeof(symp1), typeof(symp2), symp′)
    return disp′′, symp′′
end
function _tensor(op1::GaussianUnitary{B1,D1,S1}, op2::GaussianUnitary{B2,D2,S2}) where {B1<:QuadBlockBasis,B2<:QuadBlockBasis,D1,D2,S1,S2}
    basis1, basis2 = op1.basis, op2.basis
    nmodes1, nmodes2 = basis1.nmodes, basis2.nmodes
    nmodes = nmodes1 + nmodes2
    block1, block2 = Base.OneTo(nmodes1), Base.OneTo(nmodes2)
    # initialize direct sum of displacement vectors
    disp1, disp2 = op1.disp, op2.disp
    Dt = promote_type(eltype(disp1), eltype(disp2))
    disp′ = zeros(Dt, 2*nmodes)
    @inbounds for i in block1
        disp′[i] = disp1[i]
        disp′[i+nmodes] = disp1[i+nmodes1]
    end
    @inbounds for i in block2
        disp′[i+nmodes1] = disp2[i]
        disp′[i+nmodes+nmodes1] = disp2[i+nmodes2]
    end
    # initialize direct sum of symplectic matrices
    symp1, symp2 = op1.symplectic, op2.symplectic
    St = promote_type(eltype(symp1), eltype(symp2))
    symp′ = zeros(St, 2*nmodes, 2*nmodes)
    @inbounds for i in block1, j in block1
        symp′[i,j] = symp1[i,j]
        symp′[i,j+nmodes] = symp1[i,j+nmodes1]
        symp′[i+nmodes,j] = symp1[i+nmodes1,j]
        symp′[i+nmodes,j+nmodes] = symp1[i+nmodes1,j+nmodes1]
    end
    @inbounds for i in block2, j in block2
        symp′[i+nmodes1,j+nmodes1] = symp2[i,j]
        symp′[i+nmodes1,j+nmodes+nmodes1] = symp2[i,j+nmodes2]
        symp′[i+nmodes+nmodes1,j+nmodes1] = symp2[i+nmodes2,j]
        symp′[i+nmodes+nmodes1,j+nmodes+nmodes1] = symp2[i+nmodes2,j+nmodes2]
   end
    # extract output array types
    disp′′ = _promote_output_vector(typeof(disp1), typeof(disp2), disp′)
    symp′′ = _promote_output_matrix(typeof(symp1), typeof(symp2), symp′)
    return disp′′, symp′′
end

"""
    issymplectic(basis::SymplecticBasis, x::T)

Check if input matrix satisfies symplectic definition.

## Example

```jldoctest
julia> basis = QuadPairBasis(1);

julia> issymplectic(basis, [1.0 0.0; 0.0 1.0])
true
```
"""
function issymplectic(basis::SymplecticBasis, x::T; atol::R1 = 0, rtol::R2 = atol) where {T,R1<:Real,R2<:Real}
    form = symplecticform(basis)
    return isapprox(x * form * x', form; atol = atol, rtol = rtol)
end

"""
    changebasis(::SymplecticBasis, state::GaussianUnitary)

Change the symplectic basis of a Gaussian unitary.

## Example

```jldoctest
julia> op = displace(QuadBlockBasis(2), 1.0-im)
GaussianUnitary for 2 modes.
  symplectic basis: QuadBlockBasis
displacement: 4-element Vector{Float64}:
  2.0
  2.0
 -2.0
 -2.0
symplectic: 4×4 Matrix{Float64}:
 1.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0
 0.0  0.0  1.0  0.0
 0.0  0.0  0.0  1.0

julia> changebasis(QuadPairBasis, op)
GaussianUnitary for 2 modes.
  symplectic basis: QuadPairBasis
displacement: 4-element Vector{Float64}:
  2.0
 -2.0
  2.0
 -2.0
symplectic: 4×4 Matrix{Float64}:
 1.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0
 0.0  0.0  1.0  0.0
 0.0  0.0  0.0  1.0
```
"""
function changebasis(::Type{B1}, op::GaussianUnitary{B2,D,S}) where {B1<:QuadBlockBasis,B2<:QuadPairBasis,D,S}
    basis = op.basis
    nmodes = basis.nmodes
    T = zeros(eltype(S), 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(2*nmodes), j in Base.OneTo(2*nmodes)
        if (j == 2*i-1) || (j + 2*nmodes == 2*i)
            T[i,j] = 1.0
        end
    end
    T = typeof(T) == S ? T : S(T)
    disp = T * op.disp
    symp = T * op.symplectic * transpose(T)
    return GaussianUnitary(B1(nmodes), disp, symp; ħ = op.ħ)
end
function changebasis(::Type{B1}, op::GaussianUnitary{B2,D,S}) where {B1<:QuadPairBasis,B2<:QuadBlockBasis,D,S}
    basis = op.basis
    nmodes = basis.nmodes
    T = zeros(eltype(S), 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(2*nmodes), j in Base.OneTo(2*nmodes)
        if (i == 2*j-1) || (i + 2*nmodes == 2*j)
            T[i,j] = 1.0
        end
    end
    T = typeof(T) == S ? T : S(T)
    disp = T * op.disp
    symp = T * op.symplectic * transpose(T)
    return GaussianUnitary(B1(nmodes), disp, symp; ħ = op.ħ)
end
changebasis(::Type{<:QuadBlockBasis}, op::GaussianUnitary{<:QuadBlockBasis,D,S}) where {D,S} = op
changebasis(::Type{<:QuadPairBasis}, op::GaussianUnitary{<:QuadPairBasis,D,S}) where {D,S} = op