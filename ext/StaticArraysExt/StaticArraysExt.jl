module StaticArraysExt

using StaticArrays: SVector, SMatrix, SArray

using Gabs
using LinearAlgebra
import Gabs: ptrace, tensor, ⊗, _promote_output_matrix, _promote_output_vector,
SymplecticBasis, vacuumstate, thermalstate, coherentstate, squeezedstate, eprstate

include("cleaner_dispatch.jl")
include("utils.jl")

end
