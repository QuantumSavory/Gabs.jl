function _stellargaussian(op::GaussianUnitary)
    mu, nu, gamma = _bogoliubov(op)
    Sigma = conj(nu) / mu
    tau = conj(gamma) - Sigma * gamma
    n = nmodes(op)
    Vsum = (op.ħ/2) .* (op.symplectic * transpose(op.symplectic) + I)
    scale = exp((n/2)*log(op.ħ) - logdet(Vsum)/4 - dot(op.disp, Vsum \ op.disp)/4)
    return Sigma, tau, scale
end

# applies 𝒟̃_j = μ⁻¹(z - γ) - νᵀ∇ to a coefficient array
function _stellarderiv(q::Array{ComplexF64,N}, j::Int, minv, nu, shift) where {N}
    r = zeros(ComplexF64, size(q))
    for I in CartesianIndices(q)
        c = q[I]
        iszero(c) && continue
        r[I] += shift[j] * c
        for k in Base.OneTo(N)
            r[CartesianIndex(ntuple(l -> l == k ? I[l] + 1 : I[l], Val(N)))] += minv[j,k] * c
            I[k] > 1 && (r[CartesianIndex(ntuple(l -> l == k ? I[l] - 1 : I[l], Val(N)))] -=
                nu[k,j] * (I[k] - 1) * c)
        end
    end
    return r
end

# coefficients of p(z) = F_ψ(z) / G(z), a polynomial of total degree stellarrank(x)
function _stellarpolynomial(x::StellarState)
    mu, nu, gamma = _bogoliubov(x.gaussian)
    n = nmodes(x)
    minv = inv(mu)
    shift = -(minv * gamma)
    deg = stellarrank(x)
    dims = ntuple(_ -> deg + 1, n)
    p = zeros(ComplexF64, dims)
    @inbounds for K in CartesianIndices(x.core)
        c = x.core[K]
        iszero(c) && continue
        q = zeros(ComplexF64, dims)
        q[ntuple(_ -> 1, n)...] = c
        for j in Base.OneTo(n), _ in Base.OneTo(K[j]-1)
            q = _stellarderiv(q, j, minv, nu, shift)
        end
        for j in Base.OneTo(n), l in Base.OneTo(K[j]-1)
            q ./= sqrt(l)
        end
        p .+= q
    end
    return p
end

"""
    stellarfunction(x::StellarState, z)

Bargmann function `F(z) = Σₖ ψₖ z^k / √(k!)` of a stellar state, evaluated at `z`.
It factors as `𝒩 exp(½zᵀΣz + τᵀz) p(z)` with `Σ = ν̄μ⁻¹`, `τ = γ̄ - Σγ`, and `p` a
polynomial of total degree `stellarrank(x)`. The exponential factor has no zeros, so the zeros
of `F` are the zeros of `p` and are `stellarrank(x)` in number.

`GaussianUnitary` records `(d,S)` and not a metaplectic element, so `F` is fixed only up to
a global phase. The gauge here is `𝒩 > 0`.
"""
function stellarfunction(x::StellarState, z::AbstractVector)
    n = nmodes(x)
    length(z) == n || throw(DimensionMismatch(STELLAR_ERROR))
    Sigma, tau, scale = _stellargaussian(x.gaussian)
    p = _stellarpolynomial(x)
    val = zero(ComplexF64)
    @inbounds for I in CartesianIndices(p)
        c = p[I]
        iszero(c) && continue
        for j in Base.OneTo(n)
            c *= z[j]^(I[j]-1)
        end
        val += c
    end
    return scale * exp(dot(conj(z), Sigma * z)/2 + dot(conj(tau), z)) * val
end
stellarfunction(x::StellarState, z::Number) = stellarfunction(x, [z])