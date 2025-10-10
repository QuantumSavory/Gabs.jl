##
# Predefined Gaussian states
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
    mean1, mean2 = state1.mean, state2.mean
    Mt = promote_type(eltype(mean1), eltype(mean2))
    basis1, basis2 = state1.basis, state2.basis
    nmodes1, nmodes2 = basis1.nmodes, basis2.nmodes
    nmodes = nmodes1 + nmodes2
    block1, block2 = Base.OneTo(2*nmodes1), Base.OneTo(2*nmodes2)
    # initialize direct sum of mean vectors
    mean′ = zeros(Mt, 2*nmodes)
    @inbounds for i in block1
        mean′[i] = mean1[i]
    end
    @inbounds for i in block2
        mean′[i+2*nmodes1] = mean2[i]
    end
    # initialize direct sum of covariance matrices
    covar1, covar2 = state1.covar, state2.covar
    Vt = promote_type(eltype(covar1), eltype(covar2))
    covar′ = zeros(Vt, 2*nmodes, 2*nmodes)
    @inbounds for i in block1, j in block1
        covar′[i,j] = covar1[i,j]
    end
    @inbounds for i in block2, j in block2
        covar′[i+2*nmodes1,j+2*nmodes1] = covar2[i,j]
    end
    # extract output array types
    mean′′ = _promote_output_vector(typeof(mean1), typeof(mean2), mean′)
    covar′′ = _promote_output_matrix(typeof(covar1), typeof(covar2), covar′)
    return mean′′, covar′′
end
function _tensor(state1::GaussianState{B1,M1,V1}, state2::GaussianState{B2,M2,V2}) where {B1<:QuadBlockBasis,B2<:QuadBlockBasis,M1,M2,V1,V2}
    mean1, mean2 = state1.mean, state2.mean
    Mt = promote_type(eltype(mean1), eltype(mean2))
    basis1, basis2 = state1.basis, state2.basis
    nmodes1, nmodes2 = basis1.nmodes, basis2.nmodes
    nmodes = nmodes1 + nmodes2
    block1, block2 = Base.OneTo(nmodes1), Base.OneTo(nmodes2)
    # initialize direct sum of mean vectors
    mean1, mean2 = state1.mean, state2.mean
    mean′ = zeros(Mt, 2*nmodes)
    @inbounds for i in block1
        mean′[i] = mean1[i]
        mean′[i+nmodes] = mean1[i+nmodes1]
    end
    @inbounds for i in block2
        mean′[i+nmodes1] = mean2[i]
        mean′[i+nmodes+nmodes1] = mean2[i+nmodes2]
    end
    # initialize direct sum of covariance matrices
    covar1, covar2 = state1.covar, state2.covar
    Vt = promote_type(eltype(covar1), eltype(covar2))
    covar′ = zeros(Vt, 2*nmodes, 2*nmodes)
    @inbounds for i in block1, j in block1
        covar′[i,j] = covar1[i,j]
        covar′[i,j+nmodes] = covar1[i,j+nmodes1]
        covar′[i+nmodes,j] = covar1[i+nmodes1,j]
        covar′[i+nmodes,j+nmodes] = covar1[i+nmodes1,j+nmodes1]
    end
    @inbounds for i in block2, j in block2
        covar′[i+nmodes1,j+nmodes1] = covar2[i,j]
        covar′[i+nmodes1,j+nmodes+nmodes1] = covar2[i,j+nmodes2]
        covar′[i+nmodes+nmodes1,j+nmodes1] = covar2[i+nmodes2,j]
        covar′[i+nmodes+nmodes1,j+nmodes+nmodes1] = covar2[i+nmodes2,j+nmodes2]
    end
    # extract output array types
    mean′′ = _promote_output_vector(typeof(mean1), typeof(mean2), mean′)
    covar′′ = _promote_output_matrix(typeof(covar1), typeof(covar2), covar′)
    return mean′′, covar′′
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
    notidxlength = length(notindices)
    # initialize partial trace of mean vector
    mean′ = zeros(eltype(mean), 2*notidxlength)
    @inbounds for i in eachindex(notindices)
        idx = notindices[i]
        mean′[2*i-1] = mean[2*idx-1]
        mean′[2*i] = mean[2*idx]
    end
    # initialize partial trace of covariance matrix
    covar′ = zeros(eltype(covar), 2*notidxlength, 2*notidxlength)
    @inbounds for i in eachindex(notindices)
        idx = notindices[i]
        @inbounds for j in i:notidxlength
            otheridx = notindices[j]
            covar′[2*i-1, 2*j-1] = covar[2*idx-1, 2*otheridx-1]
            covar′[2*i-1, 2*j] = covar[2*idx-1, 2*otheridx]
            covar′[2*i, 2*j-1] = covar[2*idx, 2*otheridx-1]
            covar′[2*i, 2*j] = covar[2*idx, 2*otheridx]
            covar′[2*j-1, 2*i-1] = covar[2*otheridx-1, 2*idx-1]
            covar′[2*j-1, 2*i] = covar[2*otheridx-1, 2*idx]
            covar′[2*j, 2*i-1] = covar[2*otheridx, 2*idx-1]
            covar′[2*j, 2*i] = covar[2*otheridx, 2*idx]
        end
    end 
    mean′′ = _promote_output_vector(typeof(mean), mean′, 2*notidxlength)
    covar′′ = _promote_output_matrix(typeof(covar), covar′, 2*notidxlength)
    return mean′′, covar′′
end
function _ptrace(state::GaussianState{B,M,V}, indices::T) where {B<:QuadBlockBasis,M,V,T}
    basis, mean, covar = state.basis, state.mean, state.covar
    nmodes = basis.nmodes
    length(indices) < nmodes || throw(ArgumentError(INDEX_ERROR))
    notindices = setdiff(1:basis.nmodes, indices)
    notidxlength = length(notindices)
    # initialize partial trace of mean vector
    mean′ = zeros(eltype(mean), 2*notidxlength)
    @inbounds for i in eachindex(notindices)
        idx = notindices[i]
        mean′[i] = mean[idx]
        mean′[i+notidxlength] = mean[idx+nmodes]
    end
    # initialize partial trace of covariance matrix
    covar′ = zeros(eltype(covar), 2*notidxlength, 2*notidxlength)
    @inbounds for i in Base.OneTo(notidxlength)
        idx = notindices[i]
        @inbounds for j in i:notidxlength
            otheridx = notindices[j]
            covar′[i,j] = covar[idx,otheridx]
            covar′[j,i] = covar[otheridx,idx]
            covar′[i+notidxlength,j] = covar[idx+nmodes,otheridx]
            covar′[i,j+notidxlength] = covar[idx,otheridx+nmodes]
            covar′[j,i+notidxlength] = covar[otheridx,idx+nmodes]
            covar′[j+notidxlength,i] = covar[otheridx+nmodes,idx]
            covar′[i+notidxlength,j+notidxlength] = covar[idx+nmodes, otheridx+nmodes]
            covar′[j+notidxlength,i+notidxlength] = covar[otheridx+nmodes, idx+nmodes]
        end
    end 
    mean′′ = _promote_output_vector(typeof(mean), mean′, 2*notidxlength)
    covar′′ = _promote_output_matrix(typeof(covar), covar′, 2*notidxlength)
    return mean′′, covar′′
end

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
    @assert length(indices) == state.basis.nmodes "Number of indices must match number of modes in the state"
    @assert basis.nmodes ≥ length(indices) "Target basis must be large enough"

    total_modes = basis.nmodes
    sub_modes = (state.basis).nmodes
    dim = 2 * total_modes
    ħ = state.ħ
    # initialize mean and covariance with vacuum
    mean = zeros(eltype(state.mean), dim)
    covar = Matrix{eltype(state.covar)}(I, dim, dim) .* (ħ / 2)

    # map substate mean and covar into full system
    @inbounds for i in Base.OneTo(sub_modes)
        idx = indices[i]
        mean[2idx-1] = state.mean[2i-1]
        mean[2idx]   = state.mean[2i]
    end

    @inbounds for i in Base.OneTo(sub_modes)
        idx_i = indices[i]
        @inbounds for j in Base.OneTo(sub_modes)
            idx_j = indices[j]
            covar[2idx_i-1, 2idx_j-1] = state.covar[2i-1, 2j-1]
            covar[2idx_i-1, 2idx_j]   = state.covar[2i-1, 2j]
            covar[2idx_i,   2idx_j-1] = state.covar[2i,   2j-1]
            covar[2idx_i,   2idx_j]   = state.covar[2i,   2j]
        end
    end
    return Gabs.GaussianState(basis, mean, covar; ħ)
end
function embed(
    basis::QuadBlockBasis, index::Int, state::GaussianState{<:QuadBlockBasis,M,V}
) where {M,V}
    return embed(basis, [index], state)
end
function embed(
    basis::QuadBlockBasis, indices::Vector{<:Int}, state::GaussianState{<:QuadBlockBasis,M,V}
) where {M,V}
    @assert length(indices) == state.basis.nmodes "Number of indices must match number of modes in the state"
    @assert basis.nmodes ≥ length(indices) "Target basis must be large enough"

    total_modes = basis.nmodes
    sub_modes = state.basis.nmodes
    ħ = state.ħ

    mean = zeros(eltype(state.mean), 2 * total_modes)
    covar = Matrix{eltype(state.covar)}(I*(ħ / 2), 2 * total_modes, 2 * total_modes)

    @inbounds for i in Base.OneTo(sub_modes)
        idx = indices[i]
        mean[idx] = state.mean[i]
        mean[idx + total_modes] = state.mean[i + sub_modes]
    end
    @inbounds for i in Base.OneTo(sub_modes)
        idx_i = indices[i]
        @inbounds for j in Base.OneTo(sub_modes)
            idx_j = indices[j]
            covar[idx_i, idx_j] = state.covar[i, j]
            covar[idx_i, idx_j + total_modes] = state.covar[i, j + sub_modes]
            covar[idx_i + total_modes, idx_j] = state.covar[i + sub_modes, j]
            covar[idx_i + total_modes, idx_j + total_modes] = state.covar[i + sub_modes, j + sub_modes]
        end
    end
    return GaussianState(basis, mean, covar; ħ)
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
    mean = similar(state.mean)
    covar = similar(state.covar)
    @inbounds for i in Base.OneTo(nmodes)
        mean[i] = state.mean[2*i - 1]
        mean[nmodes + i] = state.mean[2*i]
        # split into two loops for better cache efficiency
        @inbounds for j in Base.OneTo(nmodes)
            covar[j, i] = state.covar[2*j - 1, 2*i - 1]
            covar[nmodes + j, i] = state.covar[2*j, 2*i - 1]
        end
        @inbounds for j in Base.OneTo(nmodes)
            covar[j, nmodes + i] = state.covar[2*j - 1, 2*i]
            covar[nmodes + j, nmodes + i] = state.covar[2*j, 2*i]
        end
    end
    return GaussianState(B1(nmodes), mean, covar)
end
function changebasis(::Type{B1}, state::GaussianState{B2,M,V}) where {B1<:QuadPairBasis,B2<:QuadBlockBasis,M,V}
    nmodes = state.basis.nmodes
    mean = similar(state.mean)
    covar = similar(state.covar)
    @inbounds for i in Base.OneTo(nmodes)
        mean[2*i - 1] = state.mean[i]
        mean[2*i] = state.mean[nmodes + i]
        # split into two loops for better cache efficiency
        @inbounds for j in Base.OneTo(nmodes)
            covar[2*j - 1, 2*i - 1] = state.covar[j, i]
            covar[2*j, 2*i - 1] = state.covar[nmodes + j, i]
        end
        @inbounds for j in Base.OneTo(nmodes)
            covar[2*j - 1, 2*i] = state.covar[j, nmodes + i]
            covar[2*j, 2*i] = state.covar[nmodes + j, nmodes + i]
        end
    end
    return GaussianState(B1(nmodes), mean, covar)
end
changebasis(::Type{<:QuadBlockBasis}, state::GaussianState{<:QuadBlockBasis,M,V}) where {M,V} = state
changebasis(::Type{<:QuadPairBasis}, state::GaussianState{<:QuadPairBasis,M,V}) where {M,V} = state


"""
    sympspectrum(state::GaussianState)

Compute the symplectic spectrum of a Gaussian state.
"""
sympspectrum(state::GaussianState) = _sympspectrum(state.covar, x -> x > 0; pre = symplecticform(state.basis))
function _sympspectrum(M::AbstractMatrix{<:Number}, select::Function; pre::Union{Nothing, AbstractMatrix{<:Number}} = nothing, post::Union{Nothing, AbstractMatrix{<:Number}} = nothing, invscale::Union{Nothing, Real} = nothing)
    M = isnothing(pre) ? M : pre * M
    M = isnothing(post) ? M : M * post
    M = isnothing(invscale) ? imag.(eigvals(M)) : imag.(eigvals(M)) ./ invscale
    return filter(x -> select(x), M)
end