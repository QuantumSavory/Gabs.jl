"""
    wigner(state::GaussianState, x)

Compute the Wigner function of an N-mode Gaussian state at `x`, a vector of size 2N.
"""
function wigner(state::GaussianState, x::T) where {T}
    basis = state.basis
    nmodes = basis.nmodes
    mean = state.mean
    isequal(length(mean), length(x)) || throw(ArgumentError(WIGNER_ERROR))

    V = state.covar
    diff = x .- mean
    arg = -(1/2) * transpose(diff) * inv(V) * diff

    return exp(arg)/((2pi)^nmodes * sqrt(det(V)))
end

"""
    wigner(x::StellarState, xi)

Wigner function of a stellar state. A Gaussian unitary acts on phase space by `ξ → Sξ + d`
with `det S = 1`, so the Wigner function is the pullback of the core's Wigner function
along `inv(x.gaussian)` with no Jacobian.
"""
function wigner(x::StellarState, xi)
    basis = x.basis
    n = basis.nmodes
    length(xi) == 2*n || throw(ArgumentError(WIGNER_ERROR))
    ginv = inv(x.gaussian)
    y = ginv.symplectic * xi .+ ginv.disp
    z = zeros(ComplexF64, n)
    if basis isa QuadPairBasis
        @inbounds for i in Base.OneTo(n)
            z[i] = (y[2i-1] + im*y[2i]) / sqrt(2*x.ħ)
        end
    else
        @inbounds for i in Base.OneTo(n)
            z[i] = (y[i] + im*y[i+n]) / sqrt(2*x.ħ)
        end
    end
    core = x.core
    result = zero(ComplexF64)
    @inbounds for K in CartesianIndices(core)
        iszero(core[K]) && continue
        for L in CartesianIndices(core)
            iszero(core[L]) && continue
            term = core[K] * conj(core[L])
            for i in Base.OneTo(n)
                term *= _fockwigner(K[i]-1, L[i]-1, z[i])
            end
            result += term
        end
    end
    return real(result) / (2*x.ħ)^n
end

function _laguerre(n::Int, k::Int, t::Real)
    n == 0 && return one(t)
    Lm, L = one(t), one(t) + k - t
    @inbounds for j in Base.OneTo(n-1)
        Lm, L = L, ((2j + 1 + k - t) * L - (j + k) * Lm) / (j + 1)
    end
    return L
end
# Wigner transform of |m⟩⟨n|, normalized so that ∫ w_{nn} d²z = 1
function _fockwigner(m::Int, n::Int, z::Number)
    m < n && return conj(_fockwigner(n, m, z))
    t = 4 * abs2(z)
    r = 1.0
    @inbounds for j in (n+1):m
        r /= sqrt(j)
    end
    return (2/π) * (-1)^n * r * (2*conj(z))^(m-n) * exp(-t/2) * _laguerre(n, m-n, t)
end

"""
    wignerchar(state::GaussianState, xi)

Compute the Wigner characteristic function of an N-mode Gaussian state at `xi`, a vector of size 2N.
"""
function wignerchar(state::GaussianState, xi::T) where {T}
    basis = state.basis
    nmodes = basis.nmodes
    mean = state.mean
    isequal(length(mean), length(xi)) || throw(ArgumentError(WIGNER_ERROR))

    V = state.covar
    Omega = symplecticform(basis)

    arg1 = -(1/2) * transpose(xi) * (Omega*V*transpose(Omega))*xi
    arg2 = im * transpose(Omega*mean) * xi

    return exp(arg1 .- arg2)
end