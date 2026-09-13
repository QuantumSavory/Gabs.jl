##
# Predefined Gaussian and non-Gaussian states
##

"""
    vacuumstate([Tm=Vector{Float64}, Tc=Matrix{Float64}], basis::SymplecticBasis)

Gaussian state with zero photons, known as the vacuum state. The symplectic representation is defined by `basis`.

## Mathematical description of a vacuum state

A vacuum state `|0⟩` is characterized by the zero mean vector and covariance
matrix `(ħ/2)I`.

## Example

```jldoctest
julia> vacuumstate(QuadPairBasis(1))
GaussianState for 1 mode.
  symplectic basis: QuadPairBasis
mean: 2-element Vector{Float64}:
 0.0
 0.0
covariance: 2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0
```
"""
function vacuumstate(::Type{Tm}, ::Type{Tc}, basis::SymplecticBasis{N}; ħ = 2) where {Tm,Tc,N<:Int}
    mean, covar = _vacuumstate(basis; ħ = ħ)
    return GaussianState(basis, Tm(mean), Tc(covar); ħ = ħ)
end
vacuumstate(::Type{T}, basis::SymplecticBasis{N}; ħ = 2) where {T,N<:Int} = vacuumstate(T, T, basis; ħ = ħ)
function vacuumstate(basis::SymplecticBasis{N}; ħ = 2) where {N<:Int}
    mean, covar = _vacuumstate(basis; ħ = ħ)
    return GaussianState(basis, mean, covar; ħ = ħ)
end
function _vacuumstate(basis::SymplecticBasis{N}; ħ = 2) where {N<:Int}
    nmodes = basis.nmodes
    mean = zeros(2*nmodes)
    covar = Matrix{Float64}((ħ/2) * I, 2*nmodes, 2*nmodes)
    return mean, covar
end

"""
    thermalstate([Tm=Vector{Float64}, Tc=Matrix{Float64},] basis::SymplecticBasis, photons<:Int)

Gaussian state at thermal equilibrium, known as the thermal state. The symplectic representation
is defined by `basis`. The mean photon number of the state is given by `photons`.

## Mathematical description of a thermal state

A thermal state `|n̄⟩`, where `n̄` is the mean number of photons,
is characterized by the zero mean vector and covariance
matrix `ħ(n̄+1/2)I`.

## Example

```jldoctest
julia> thermalstate(QuadPairBasis(1), 4)
GaussianState for 1 mode.
  symplectic basis: QuadPairBasis
mean: 2-element Vector{Float64}:
 0.0
 0.0
covariance: 2×2 Matrix{Float64}:
 9.0  0.0
 0.0  9.0
```
"""
function thermalstate(::Type{Tm}, ::Type{Tc}, basis::SymplecticBasis{N}, photons::P; ħ = 2) where {Tm,Tc,N<:Int,P}
    mean, covar = _thermalstate(basis, photons; ħ = ħ)
    return GaussianState(basis, Tm(mean), Tc(covar); ħ = ħ)
end
thermalstate(::Type{T}, basis::SymplecticBasis{N}, photons::P; ħ = 2) where {T,N<:Int,P} = thermalstate(T, T, basis, photons; ħ = ħ)
function thermalstate(basis::SymplecticBasis{N}, photons::P; ħ = 2) where {N<:Int,P}
    mean, covar = _thermalstate(basis, photons; ħ = ħ)
    return GaussianState(basis, mean, covar; ħ = ħ)
end
function _thermalstate(basis::Union{QuadPairBasis{N},QuadBlockBasis{N}}, photons::P; ħ = 2) where {N<:Int,P<:Number}
    nmodes = basis.nmodes
    Rt = float(eltype(P))
    mean = zeros(Rt, 2*nmodes)
    covar = Matrix{Rt}((2 * photons + 1) * (ħ/2) * I, 2*nmodes, 2*nmodes)
    return mean, covar
end
function _thermalstate(basis::QuadPairBasis{N}, photons::P; ħ = 2) where {N<:Int,P<:Vector}
    nmodes = basis.nmodes
    Rt = float(eltype(P))
    mean = zeros(Rt, 2*nmodes)
    covar = zeros(Rt, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes)
        val = (2 * photons[i] + 1) * (ħ/2)
        covar[2*i-1, 2*i-1] = val
        covar[2*i, 2*i] = val
    end
    return mean, covar
end
function _thermalstate(basis::QuadBlockBasis{N}, photons::P; ħ = 2) where {N<:Int,P<:Vector}
    nmodes = basis.nmodes
    Rt = float(eltype(P))
    mean = zeros(Rt, 2*nmodes)
    covar = zeros(Rt, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes)
        val = (2 * photons[i] + 1) * (ħ/2)
        covar[i, i] = val
        covar[i+nmodes, i+nmodes] = val
    end
    return mean, covar
end

"""
    coherentstate([Tm=Vector{Float64}, Tc=Matrix{Float64},] basis::SymplecticBasis, alpha<:Number)

Gaussian state that is the quantum analogue of a monochromatic electromagnetic field, known
as the coherent state. The symplectic representation is defined by `basis`.
The complex amplitude of the state is given by `alpha`.

## Mathematical description of a coherent state

A coherent state `|α⟩`, where `α` is the complex amplitude,
is characterized by the mean vector `√2ħ [real(α), imag(α)]` and covariance
matrix `(ħ/2)I`.

## Example

```jldoctest
julia> coherentstate(QuadPairBasis(1), 1.0+im)
GaussianState for 1 mode.
  symplectic basis: QuadPairBasis
mean: 2-element Vector{Float64}:
 2.0
 2.0
covariance: 2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0
```
"""
function coherentstate(::Type{Tm}, ::Type{Tc}, basis::SymplecticBasis{N}, alpha::A; ħ = 2) where {Tm,Tc,N<:Int,A}
    mean, covar = _coherentstate(basis, alpha; ħ = ħ)
    return GaussianState(basis, Tm(mean), Tc(covar); ħ = ħ)
end
coherentstate(::Type{T}, basis::SymplecticBasis{N}, alpha::A; ħ = 2) where {T,N<:Int,A} = coherentstate(T, T, basis, alpha; ħ = ħ)
function coherentstate(basis::SymplecticBasis{N}, alpha::A; ħ = 2) where {N<:Int,A}
    mean, covar = _coherentstate(basis, alpha; ħ = ħ)
    return GaussianState(basis, mean, covar; ħ = ħ)
end
function _coherentstate(basis::QuadPairBasis{N}, alpha::A; ħ = 2) where {N<:Int,A<:Number}
    nmodes = basis.nmodes
    mean = repeat([sqrt(2*ħ) * real(alpha), sqrt(2*ħ) * imag(alpha)], nmodes)
    covar = Matrix{real(A)}((ħ/2) * I, 2*nmodes, 2*nmodes)
    return mean, covar
end
function _coherentstate(basis::QuadPairBasis{N}, alpha::A; ħ = 2) where {N<:Int,A<:Vector}
    nmodes = basis.nmodes
    Rt = real(eltype(A))
    mean = sqrt(2*ħ) * reinterpret(Rt, alpha)
    covar = Matrix{Rt}((ħ/2) * I, 2*nmodes, 2*nmodes)
    return mean, covar
end
function _coherentstate(basis::QuadBlockBasis{N}, alpha::A; ħ = 2) where {N<:Int,A<:Number}
    nmodes = basis.nmodes
    mean = repeat([sqrt(2*ħ) * real(alpha), sqrt(2*ħ) * imag(alpha)], inner = nmodes)
    covar = Matrix{real(A)}((ħ/2) * I, 2*nmodes, 2*nmodes)
    return mean, covar
end
function _coherentstate(basis::QuadBlockBasis{N}, alpha::A; ħ = 2) where {N<:Int,A<:Vector}
    nmodes = basis.nmodes
    Rt = real(eltype(A))
    re = reinterpret(Rt, alpha)
    mean = vcat(@view(re[1:2:end]), @view(re[2:2:end]))
    mean .*= sqrt(2*ħ)
    covar = Matrix{Rt}((ħ/2) * I, 2*nmodes, 2*nmodes)
    return mean, covar
end

"""
    squeezedstate([Tm=Vector{Float64}, Tc=Matrix{Float64},] basis::SymplecticBasis, r<:Real, theta<:Real)

Gaussian state with quantum uncertainty in its phase and amplitude, known as
the squeezed state. The symplectic representation is defined by `basis`. The amplitude and phase squeezing parameters are given by `r`
and `theta`, respectively.

## Mathematical description of a squeezed state

A squeezed state `|r, θ⟩`, where `r` is the amplitude squeezing
parameter and `θ` is the phase squeezing parameter,
is characterized by the zero mean vector and covariance
matrix `(ħ/2) (cosh(2r)I - sinh(2r)R(θ))`, where `R(θ)` is the rotation matrix.

## Example

```jldoctest
julia> squeezedstate(QuadPairBasis(1), 0.5, pi/4)
GaussianState for 1 mode.
  symplectic basis: QuadPairBasis
mean: 2-element Vector{Float64}:
 0.0
 0.0
covariance: 2×2 Matrix{Float64}:
  0.712088  -0.830993
 -0.830993   2.37407
```
"""
function squeezedstate(::Type{Tm}, ::Type{Tc}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {Tm,Tc,N<:Int,R}
    mean, covar = _squeezedstate(basis, r, theta; ħ = ħ)
    return GaussianState(basis, Tm(mean), Tc(covar); ħ = ħ)
end
squeezedstate(::Type{T}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {T,N<:Int,R} = squeezedstate(T, T, basis, r, theta; ħ = ħ)
function squeezedstate(basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {N<:Int,R}
    mean, covar = _squeezedstate(basis, r, theta; ħ = ħ)
    return GaussianState(basis, mean, covar; ħ = ħ)
end
function _squeezedstate(basis::QuadPairBasis{N}, r::R, theta::R; ħ = 2) where {N<:Int,R<:Real}
    nmodes = basis.nmodes
    mean = zeros(R, 2*nmodes)
    covar = zeros(R, 2*nmodes, 2*nmodes)
    cr, sr = cosh(2*r), sinh(2*r)
    ct, st = cos(theta), sin(theta)
    @inbounds for i in Base.OneTo(nmodes)
        covar[2*i-1, 2*i-1] = (ħ/2) * (cr - sr*ct)
        covar[2*i-1, 2*i] = -(ħ/2) * sr * st
        covar[2*i, 2*i-1] = -(ħ/2) * sr * st
        covar[2*i, 2*i] = (ħ/2) * (cr + sr*ct)
    end
    return mean, covar
end
function _squeezedstate(basis::QuadPairBasis{N}, r::R, theta::R; ħ = 2) where {N<:Int,R<:Vector}
    nmodes = basis.nmodes
    Rt = eltype(R)
    mean = zeros(Rt, 2*nmodes)
    covar = zeros(Rt, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes)
        cr, sr = cosh(2*r[i]), sinh(2*r[i])
        ct, st = cos(theta[i]), sin(theta[i])
        covar[2*i-1, 2*i-1] = (ħ/2) * (cr - sr*ct)
        covar[2*i-1, 2*i] = -(ħ/2) * sr * st
        covar[2*i, 2*i-1] = -(ħ/2) * sr * st
        covar[2*i, 2*i] = (ħ/2) * (cr + sr*ct)
    end
    return mean, covar
end
function _squeezedstate(basis::QuadBlockBasis{N}, r::R, theta::R; ħ = 2) where {N<:Int,R<:Real}
    nmodes = basis.nmodes
    mean = zeros(R, 2*nmodes)
    covar = zeros(R, 2*nmodes, 2*nmodes)
    cr, sr = cosh(2*r), sinh(2*r)
    ct, st = cos(theta), sin(theta)
    @inbounds for i in Base.OneTo(nmodes)
        covar[i, i] = (ħ/2) * (cr - sr*ct)
        covar[i, i+nmodes] = -(ħ/2) * sr * st
        covar[i+nmodes, i] = -(ħ/2) * sr * st
        covar[i+nmodes, i+nmodes] = (ħ/2) * (cr + sr*ct)
    end
    return mean, covar
end
function _squeezedstate(basis::QuadBlockBasis{N}, r::R, theta::R; ħ = 2) where {N<:Int,R<:Vector}
    nmodes = basis.nmodes
    Rt = eltype(R)
    mean = zeros(Rt, 2*nmodes)
    covar = zeros(Rt, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(nmodes)
        cr, sr = cosh(2*r[i]), sinh(2*r[i])
        ct, st = cos(theta[i]), sin(theta[i])
        covar[i, i] = (ħ/2) * (cr - sr*ct)
        covar[i, i+nmodes] = -(ħ/2) * sr * st
        covar[i+nmodes, i] = -(ħ/2) * sr * st
        covar[i+nmodes, i+nmodes] = (ħ/2) * (cr + sr*ct)
    end
    return mean, covar
end

"""
    eprstate([Tm=Vector{Float64}, Tc=Matrix{Float64},] basis::SymplecticBasis, r<:Real, theta<:Real)

Gaussian state that is a two-mode squeezed state, known as the Einstein-Podolski-Rosen (EPR) state. The symplectic
representation is defined by `basis`. The amplitude and phase squeezing parameters are given by `r` and `theta`, respectively.

## Mathematical description of an EPR state

An EPR state `|r, θ⟩ₑₚᵣ`, where `r` is the amplitude squeezing
parameter and `θ` is the phase squeezing parameter,
is characterized by the zero mean vector and covariance
matrix `(ħ/2)[cosh(2r)I -sinh(2r)R(θ); -sinh(2r)R(θ) cosh(2r)I]`, 
where `R(θ)` is the rotation matrix.

## Example

```jldoctest
julia> eprstate(QuadPairBasis(2), 0.5, pi/4)
GaussianState for 2 modes.
  symplectic basis: QuadPairBasis
mean: 4-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
covariance: 4×4 Matrix{Float64}:
  1.54308    0.0       -0.830993  -0.830993
  0.0        1.54308   -0.830993   0.830993
 -0.830993  -0.830993   1.54308    0.0
 -0.830993   0.830993   0.0        1.54308
```
"""
function eprstate(::Type{Tm}, ::Type{Tc}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {Tm,Tc,N<:Int,R}
    mean, covar = _eprstate(basis, r, theta; ħ = ħ)
    return GaussianState(basis, Tm(mean), Tc(covar); ħ = ħ)
end
eprstate(::Type{T}, basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {T,N<:Int,R} = eprstate(T, T, basis, r, theta; ħ = ħ)
function eprstate(basis::SymplecticBasis{N}, r::R, theta::R; ħ = 2) where {N<:Int,R}
    mean, covar = _eprstate(basis, r, theta; ħ = ħ)
    return GaussianState(basis, mean, covar; ħ = ħ)
end
function _eprstate(basis::QuadPairBasis{N}, r::R, theta::R; ħ = 2) where {N<:Int,R<:Real}
    nmodes = basis.nmodes
    mean = zeros(R, 2*nmodes)
    cr, sr = (ħ/2)*cosh(2*r), (ħ/2)*sinh(2*r)
    ct, st = cos(theta), sin(theta)
    covar = zeros(R, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(Int(nmodes/2))
        covar[4*i-3, 4*i-3] = cr
        covar[4*i-3, 4*i-1] = -sr * ct
        covar[4*i-3, 4*i] = -sr * st

        covar[4*i-2, 4*i-2] = cr
        covar[4*i-2, 4*i-1] = -sr * st
        covar[4*i-2, 4*i] = sr * ct

        covar[4*i-1, 4*i-3] = -sr * ct
        covar[4*i-1, 4*i-2] = -sr * st
        covar[4*i-1, 4*i-1] = cr

        covar[4*i, 4*i-3] = -sr * st
        covar[4*i, 4*i-2] = sr * ct
        covar[4*i, 4*i] = cr
    end
    return mean, covar
end
function _eprstate(basis::QuadPairBasis{N}, r::R, theta::R; ħ = 2) where {N<:Int,R<:Vector}
    nmodes = basis.nmodes
    Rt = eltype(R)
    mean = zeros(Rt, 2*nmodes)
    covar = zeros(Rt, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(Int(nmodes/2))
        cr, sr = (ħ/2)*cosh(2*r[i]), (ħ/2)*sinh(2*r[i])
        ct, st = cos(theta[i]), sin(theta[i])

        covar[4*i-3, 4*i-3] = cr
        covar[4*i-3, 4*i-1] = -sr * ct
        covar[4*i-3, 4*i] = -sr * st

        covar[4*i-2, 4*i-2] = cr
        covar[4*i-2, 4*i-1] = -sr * st
        covar[4*i-2, 4*i] = sr * ct

        covar[4*i-1, 4*i-3] = -sr * ct
        covar[4*i-1, 4*i-2] = -sr * st
        covar[4*i-1, 4*i-1] = cr

        covar[4*i, 4*i-3] = -sr * st
        covar[4*i, 4*i-2] = sr * ct
        covar[4*i, 4*i] = cr
    end
    return mean, covar
end
function _eprstate(basis::QuadBlockBasis{N}, r::R, theta::R; ħ = 2) where {N<:Int,R<:Real}
    nmodes = basis.nmodes
    mean = zeros(R, 2*nmodes)
    cr, sr = (ħ/2)*cosh(2*r), (ħ/2)*sinh(2*r)
    ct, st = cos(theta), sin(theta)
    covar = zeros(R, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(Int(nmodes/2))
        covar[2*i-1, 2*i-1] = cr
        covar[2*i-1, 2*i] = -sr * ct
        covar[2*i, 2*i-1] = -sr * ct
        covar[2*i, 2*i] = cr

        covar[2*i-1, 2*i+nmodes] = -sr * st
        covar[2*i, 2*i+nmodes-1] = -sr * st

        covar[2*i+nmodes-1, 2*i+nmodes-1] = cr
        covar[2*i+nmodes-1, 2*i+nmodes] = sr * ct
        covar[2*i+nmodes, 2*i+nmodes-1] = sr * ct
        covar[2*i+nmodes, 2*i+nmodes] = cr

        covar[2*i+nmodes-1, 2*i] = -sr * st
        covar[2*i+nmodes, 2*i-1] = -sr * st
    end
    return mean, covar
end
function _eprstate(basis::QuadBlockBasis{N}, r::R, theta::R; ħ = 2) where {N<:Int,R<:Vector}
    nmodes = basis.nmodes
    Rt = eltype(R)
    mean = zeros(Rt, 2*nmodes)
    covar = zeros(Rt, 2*nmodes, 2*nmodes)
    @inbounds for i in Base.OneTo(Int(nmodes/2))
        cr, sr = (ħ/2)*cosh(2*r[i]), (ħ/2)*sinh(2*r[i])
        ct, st = cos(theta[i]), sin(theta[i])

        covar[2*i-1, 2*i-1] = cr
        covar[2*i-1, 2*i] = -sr * ct
        covar[2*i, 2*i-1] = -sr * ct
        covar[2*i, 2*i] = cr

        covar[2*i-1, 2*i+nmodes] = -sr * st
        covar[2*i, 2*i+nmodes-1] = -sr * st

        covar[2*i+nmodes-1, 2*i+nmodes-1] = cr
        covar[2*i+nmodes-1, 2*i+nmodes] = sr * ct
        covar[2*i+nmodes, 2*i+nmodes-1] = sr * ct
        covar[2*i+nmodes, 2*i+nmodes] = cr

        covar[2*i+nmodes-1, 2*i] = -sr * st
        covar[2*i+nmodes, 2*i-1] = -sr * st
    end
    return mean, covar
end

"""
    fockstate([Tc=Array{ComplexF64}], basis::SymplecticBasis, photons<:Int)
    fockstate([Tc=Array{ComplexF64}], basis::SymplecticBasis, photons<:AbstractVector)

`StellarState` whose core is a single Fock state and whose Gaussian factor is the identity.

A vector gives the occupation of each mode, so the stellar rank is `sum(photons)`. A scalar
is broadcast to every mode, as elsewhere in Gabs, so `fockstate(basis, n)` is `|n,…,n⟩` with
rank `n * basis.nmodes`.
"""
function fockstate(::Type{Tc}, basis::SymplecticBasis{N}, photons::P; ħ = 2) where {Tc,N<:Int,P}
    core, op = _fockstate(basis, photons; ħ = ħ)
    return StellarState(Tc(core), op)
end
function fockstate(basis::SymplecticBasis{N}, photons::P; ħ = 2) where {N<:Int,P}
    core, op = _fockstate(basis, photons; ħ = ħ)
    return StellarState(core, op)
end
function _fockstate(basis::SymplecticBasis{N}, photons::P; ħ = 2) where {N<:Int,P<:Int}
    return _fockstate(basis, fill(photons, basis.nmodes); ħ = ħ)
end
function _fockstate(basis::SymplecticBasis{N}, photons::P; ħ = 2) where {N<:Int,P<:AbstractVector}
    length(photons) == basis.nmodes || throw(DimensionMismatch(
        lazy"The occupation vector must carry one entry per mode."))
    all(≥(0), photons) || throw(ArgumentError(
        lazy"The occupation of each mode must be a nonnegative integer."))
    dims = Tuple(photons .+ 1)
    core = zeros(ComplexF64, dims)
    core[CartesianIndex(dims)] = one(ComplexF64)
    return core, displace(basis, zero(ComplexF64); ħ = ħ)
end

##
# Operations on Gaussian states
##

"""
    tensor(state1::GaussianState, state2::GaussianState)

tensor product of Gaussian states, which can also be called with `⊗`.

## Example
```jldoctest
julia> basis = QuadPairBasis(1);

julia> coherentstate(basis, 1.0+im) ⊗ thermalstate(basis, 2)
GaussianState for 2 modes.
  symplectic basis: QuadPairBasis
mean: 4-element Vector{Float64}:
 2.0
 2.0
 0.0
 0.0
covariance: 4×4 Matrix{Float64}:
 1.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0
 0.0  0.0  5.0  0.0
 0.0  0.0  0.0  5.0
```
"""
function tensor(::Type{Tm}, ::Type{Tc}, state1::GaussianState, state2::GaussianState) where {Tm,Tc}
    typeof(state1.basis) == typeof(state2.basis) || throw(ArgumentError(SYMPLECTIC_ERROR))
    state1.ħ == state2.ħ || throw(ArgumentError(HBAR_ERROR))
    mean, covar = _tensor(state1, state2)
    return GaussianState(state1.basis ⊕ state2.basis, Tm(mean), Tc(covar); ħ = state1.ħ)
end
tensor(::Type{T}, state1::GaussianState, state2::GaussianState) where {T} = tensor(T, T, state1, state2)
function tensor(state1::GaussianState, state2::GaussianState)
    typeof(state1.basis) == typeof(state2.basis) || throw(ArgumentError(SYMPLECTIC_ERROR))
    state1.ħ == state2.ħ || throw(ArgumentError(HBAR_ERROR))
    mean, covar = _tensor(state1, state2)
    return GaussianState(state1.basis ⊕ state2.basis, mean, covar; ħ = state1.ħ)
end
function _tensor(state1::GaussianState{B1,M1,V1}, state2::GaussianState{B2,M2,V2}) where {B1<:QuadPairBasis,B2<:QuadPairBasis,M1,M2,V1,V2}
    return _directsummoments(state1.mean, state1.covar, state2.mean, state2.covar,
                             _tensorperm(state1.basis, state2.basis))
end
function _tensor(state1::GaussianState{B1,M1,V1}, state2::GaussianState{B2,M2,V2}) where {B1<:QuadBlockBasis,B2<:QuadBlockBasis,M1,M2,V1,V2}
    return _directsummoments(state1.mean, state1.covar, state2.mean, state2.covar,
                             _tensorperm(state1.basis, state2.basis))
end

# Stack the two operands, then relabel. Concatenation already gives the direct
# sum in the pairwise layout; the blockwise layout needs the quadratures of the
# joint system regrouped, which `_tensorperm` expresses as a single gather.
function _directsummoments(mean1, covar1, mean2, covar2, perm)
    m1, m2 = _codevice(mean1, mean2)
    mean′ = vcat(m1, m2)
    n1, n2 = length(mean1), length(mean2)
    covar′ = _blockdiag(covar1, covar2, n1, n2)
    if perm !== nothing
        mean′ = mean′[perm]
        covar′ = covar′[perm, perm]
    end
    mean′′ = _promote_output_vector(typeof(mean1), typeof(mean2), mean′)
    covar′′ = _promote_output_matrix(typeof(covar1), typeof(covar2), covar′)
    return mean′′, covar′′
end

function _blockdiag(A, B, n1::Int, n2::Int)
    A′, B′ = _codevice(A, B)
    T = promote_type(eltype(A′), eltype(B′))
    out = similar(A′, T, n1 + n2, n1 + n2)
    fill!(out, zero(T))
    @views out[1:n1, 1:n1] .= A′
    @views out[n1+1:n1+n2, n1+1:n1+n2] .= B′
    return out
end

"""
    _codevice(A, B)

`A` and `B` as a pair that can be combined directly.

Two operands of a tensor product need not start on the same device; a backend
that cannot read the other's memory defines this to bring both onto its own.
"""
_codevice(A, B) = (A, B)

# Pairwise concatenation is already the direct sum, so no relabelling is needed.
_tensorperm(::QuadPairBasis, ::QuadPairBasis) = nothing
# Blockwise stacking gives [q⁽¹⁾,p⁽¹⁾,q⁽²⁾,p⁽²⁾]; the joint state wants
# [q⁽¹⁾,q⁽²⁾,p⁽¹⁾,p⁽²⁾].
function _tensorperm(basis1::QuadBlockBasis, basis2::QuadBlockBasis)
    n1, n2 = basis1.nmodes, basis2.nmodes
    return vcat(1:n1, 2*n1+1:2*n1+n2, n1+1:2*n1, 2*n1+n2+1:2*n1+2*n2)
end

"""
    tensor(state1::StellarState, state2::StellarState)

Tensor product of stellar states, which can also be called with `⊗`.
"""
function tensor(state1::StellarState, state2::StellarState)
    gaussian = state1.gaussian ⊗ state2.gaussian
    core1, core2 = state1.core, state2.core
    core = reshape(vec(core1) * transpose(vec(core2)), (size(core1)..., size(core2)...))
    return StellarState(core, gaussian)
end
function tensor(::Type{Tc}, ::Type{Td}, ::Type{Ts}, x::StellarState, y::StellarState) where {Tc,Td,Ts}
    core1, core2 = x.core, y.core
    core = reshape(vec(core1) * transpose(vec(core2)), (size(core1)..., size(core2)...))
    return StellarState(Tc(core), tensor(Td, Ts, x.gaussian, y.gaussian))
end

"""
    ptrace([Tm=Vector{Float64}, Tc=Matrix{Float64},] state::GaussianState, idx<:Int)
    ptrace([Tm=Vector{Float64}, Tc=Matrix{Float64},] state::GaussianState, indices<:AbstractVector)

Partial trace of a Gaussian state over a subsystem indicated by `idx`, or multiple subsystems
indicated by `indices`.

## Example
```jldoctest
julia> basis = QuadPairBasis(1);

julia> state = coherentstate(basis, 1.0+im) ⊗ thermalstate(basis, 2) ⊗ squeezedstate(basis, 3.0, pi/4);

julia> ptrace(state, 2)
GaussianState for 2 modes.
  symplectic basis: QuadPairBasis
mean: 4-element Vector{Float64}:
 2.0
 2.0
 0.0
 0.0
covariance: 4×4 Matrix{Float64}:
 1.0  0.0     0.0        0.0
 0.0  1.0     0.0        0.0
 0.0  0.0    59.0829  -142.633
 0.0  0.0  -142.633    344.348

julia> ptrace(state, [1, 3])
GaussianState for 1 mode.
  symplectic basis: QuadPairBasis
mean: 2-element Vector{Float64}:
 0.0
 0.0
covariance: 2×2 Matrix{Float64}:
 5.0  0.0
 0.0  5.0
```
"""
function ptrace(::Type{Tm}, ::Type{Tc}, state::GaussianState, indices::N) where {Tm,Tc,N}
    basis = state.basis
    mean, covar = _ptrace(state, indices)
    return GaussianState(typeof(basis)(basis.nmodes - length(indices)), Tm(mean), Tc(covar); ħ = state.ħ)
end
ptrace(::Type{T}, state::GaussianState, indices::N) where {T,N} = ptrace(T, T, state, indices)
function ptrace(state::GaussianState, indices::T) where {T}
    basis = state.basis
    mean, covar = _ptrace(state, indices)
    return GaussianState(typeof(basis)(basis.nmodes - length(indices)), mean, covar; ħ = state.ħ)
end
function _ptrace(state::GaussianState{B,M,V}, indices::T) where {B<:QuadPairBasis,M,V,T}
    basis, mean, covar = state.basis, state.mean, state.covar
    length(indices) < basis.nmodes || throw(ArgumentError(INDEX_ERROR))
    notindices = setdiff(1:basis.nmodes, indices)
    quad = _quadindices(basis, notindices)
    return _gathermoments(mean, covar, quad)
end
function _ptrace(state::GaussianState{B,M,V}, indices::T) where {B<:QuadBlockBasis,M,V,T}
    basis, mean, covar = state.basis, state.mean, state.covar
    length(indices) < basis.nmodes || throw(ArgumentError(INDEX_ERROR))
    notindices = setdiff(1:basis.nmodes, indices)
    quad = _quadindices(basis, notindices)
    return _gathermoments(mean, covar, quad)
end

# Keeping a set of modes is a symmetric gather on the quadratures they own, so
# one indexing expression serves both bases and every array backend.
function _gathermoments(mean, covar, quad)
    mean′ = mean[quad]
    covar′ = covar[quad, quad]
    mean′′ = _promote_output_vector(typeof(mean), mean′, length(quad))
    covar′′ = _promote_output_matrix(typeof(covar), covar′, length(quad))
    return mean′′, covar′′
end

ptrace(::StellarState, ::Int) = throw(ArgumentError(STELLAR_PTRACE_ERROR))
ptrace(::StellarState, ::AbstractVector{<:Int}) = throw(ArgumentError(STELLAR_PTRACE_ERROR))
ptrace(::Type{Tm}, ::Type{Tc}, ::StellarState, ::Any) where {Tm,Tc} =
    throw(ArgumentError(STELLAR_PTRACE_ERROR))

"""
    embed(basis::SymplecticBasis, idx::Int, state::GaussianState)
    embed(basis::SymplecticBasis, indices::AbstractVector{<:Int}, state::GaussianState)

Embed a smaller Gaussian state into a larger Hilbert space specified by a target
symplectic basis, inserting it at the mode index (or indices) specified by `idx` or `indices`.

The embedded state's mean vector and covariance matrix are inserted into the larger space,
and all other modes are initialized in the vacuum state (zero mean, covariance equal to ħ/2 × I).
The function returns a new `GaussianState` with the full `basis`.

## Example
```jldoctest
julia> state = squeezedstate(QuadBlockBasis(1), 1.0, pi/4);

julia> embed(QuadBlockBasis(3), 2, state)
GaussianState for 3 modes.
  symplectic basis: QuadBlockBasis
mean: 6-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
covariance: 6×6 Matrix{Float64}:
 1.0   0.0      0.0  0.0   0.0      0.0
 0.0   1.19762  0.0  0.0  -2.56458  0.0
 0.0   0.0      1.0  0.0   0.0      0.0
 0.0   0.0      0.0  1.0   0.0      0.0
 0.0  -2.56458  0.0  0.0   6.32677  0.0
 0.0   0.0      0.0  0.0   0.0      1.0
```
"""
function embed(
    basis::QuadPairBasis, index::Int, state::GaussianState{<:QuadPairBasis,M,V}
    ) where {M,V}
    return embed(basis, [index], state)
end
function embed(
    basis::QuadPairBasis, indices::Vector{<:Int}, state::GaussianState{<:QuadPairBasis,M,V}
) where {M,V}
    return _embedstate(basis, indices, state)
end
function embed(
    basis::QuadBlockBasis, index::Int, state::GaussianState{<:QuadBlockBasis,M,V}
) where {M,V}
    return embed(basis, [index], state)
end
function embed(
    basis::QuadBlockBasis, indices::Vector{<:Int}, state::GaussianState{<:QuadBlockBasis,M,V}
) where {M,V}
    return _embedstate(basis, indices, state)
end

# Scatter the substate onto a vacuum background at the quadratures owned by
# `indices`; the layout only enters through `_quadindices`.
function _embedstate(basis::SymplecticBasis, indices::Vector{<:Int}, state::GaussianState)
    @assert length(indices) == state.basis.nmodes "Number of indices must match number of modes in the state"
    @assert basis.nmodes ≥ length(indices) "Target basis must be large enough"
    ħ = state.ħ
    dim = 2 * basis.nmodes
    q = _quadindices(basis, indices)
    mean = similar(state.mean, dim)
    fill!(mean, zero(eltype(state.mean)))
    covar = similar(state.covar, dim, dim)
    fill!(covar, zero(eltype(state.covar)))
    covar[diagind(covar)] .= eltype(state.covar)(ħ / 2)
    mean[q] = state.mean
    covar[q, q] = state.covar
    return GaussianState(basis, mean, covar; ħ)
end

"""
    embed(basis::SymplecticBasis, idx::Int, state::StellarState)
    embed(basis::SymplecticBasis, indices::AbstractVector{<:Int}, state::StellarState)

Embed a smaller stellar state into a larger Hilbert space specified by a target
symplectic basis, inserting it at the mode index (or indices) specified by `idx` or `indices`.
"""
function embed(basis::SymplecticBasis, index::Int, x::StellarState)
    return embed(basis, [index], x)
end
function embed(basis::SymplecticBasis, indices::Vector{<:Int}, x::StellarState)
    @assert length(indices) == nmodes(x) "Number of indices must match number of modes in the state"
    @assert basis.nmodes ≥ length(indices) "Target basis must be large enough"
    total = basis.nmodes
    dims = ones(Int, total)
    @inbounds for (i, idx) in enumerate(indices)
        dims[idx] = size(x.core, i)
    end
    core = zeros(eltype(x.core), dims...)
    target = ones(Int, total)
    @inbounds for I in CartesianIndices(x.core)
        for (i, idx) in enumerate(indices)
            target[idx] = I[i]
        end
        core[CartesianIndex(target...)] = x.core[I]
    end
    return StellarState(core, embed(basis, indices, x.gaussian))
end

"""
    changebasis(::SymplecticBasis, state::GaussianState)

Change the symplectic basis of a Gaussian state.

# Example

```jldoctest
julia> st = squeezedstate(QuadBlockBasis(2), 1.0, 2.0)
GaussianState for 2 modes.
  symplectic basis: QuadBlockBasis
mean: 4-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
covariance: 4×4 Matrix{Float64}:
  5.2715    0.0      -3.29789   0.0
  0.0       5.2715    0.0      -3.29789
 -3.29789   0.0       2.25289   0.0
  0.0      -3.29789   0.0       2.25289

julia> changebasis(QuadPairBasis, st)
GaussianState for 2 modes.
  symplectic basis: QuadPairBasis
mean: 4-element Vector{Float64}:
 0.0
 0.0
 0.0
 0.0
covariance: 4×4 Matrix{Float64}:
  5.2715   -3.29789   0.0       0.0
 -3.29789   2.25289   0.0       0.0
  0.0       0.0       5.2715   -3.29789
  0.0       0.0      -3.29789   2.25289
```
"""
function changebasis(::Type{B1}, state::GaussianState{B2,M,V}) where {B1<:QuadBlockBasis,B2<:QuadPairBasis,M,V}
    nmodes = state.basis.nmodes
    p = _basisperm(B1, nmodes)
    return GaussianState(B1(nmodes), state.mean[p], state.covar[p, p]; ħ = state.ħ)
end
function changebasis(::Type{B1}, state::GaussianState{B2,M,V}) where {B1<:QuadPairBasis,B2<:QuadBlockBasis,M,V}
    nmodes = state.basis.nmodes
    p = _basisperm(B1, nmodes)
    return GaussianState(B1(nmodes), state.mean[p], state.covar[p, p]; ħ = state.ħ)
end
changebasis(::Type{<:QuadBlockBasis}, state::GaussianState{<:QuadBlockBasis,M,V}) where {M,V} = state
changebasis(::Type{<:QuadPairBasis}, state::GaussianState{<:QuadPairBasis,M,V}) where {M,V} = state

"""
    changebasis(::SymplecticBasis, state::StellarState)

Change the symplectic basis of a stellar state.
"""
changebasis(::Type{B1}, x::StellarState) where {B1<:SymplecticBasis} = StellarState(x.core, changebasis(B1, x.gaussian))
changebasis(::Type{<:QuadPairBasis}, x::StellarState{C,<:GaussianUnitary{<:QuadPairBasis}}) where {C} = x
changebasis(::Type{<:QuadBlockBasis}, x::StellarState{C,<:GaussianUnitary{<:QuadBlockBasis}}) where {C} = x

"""
    sympspectrum(state::GaussianState)

Compute the symplectic spectrum of a Gaussian state.
"""
sympspectrum(state::GaussianState) = _sympspectrum(state.covar, x -> x > 0; pre = _symplecticform(state.basis, state.covar))
function _sympspectrum(M::AbstractMatrix{<:Number}, select::Function; pre::Union{Nothing, AbstractMatrix{<:Number}} = nothing, post::Union{Nothing, AbstractMatrix{<:Number}} = nothing, invscale::Union{Nothing, Real} = nothing)
    M = isnothing(pre) ? M : pre * M
    M = isnothing(post) ? M : M * post
    M = isnothing(invscale) ? imag.(eigvals(M)) : imag.(eigvals(M)) ./ invscale
    return filter(x -> select(x), M)
end
