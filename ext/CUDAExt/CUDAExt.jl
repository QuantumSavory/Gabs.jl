module CUDAExt

using CUDA: CuArray, CuVector, CuMatrix
using LinearAlgebra: Symmetric, cholesky, diag

using Gabs
import Gabs: _promote_output_matrix, _promote_output_vector, _det, _logdet,
             _symplecticform, _complexform, _like, _codevice
using Gabs: SymplecticBasis, symplecticform

include("utils.jl")

end
