SYMPLECTIC_ERROR = lazy"The symplectic forms of the Gaussian objects do not match.
    To resolve this issue, either revise the Gaussian objects or consider using
    the function `changebasis` to arrive at a common symplectic form."

STATE_ERROR = lazy"Either the input covariance matrix is not
    square or its dimensions do not match the mean vector. An
    arbitrary N-mode Gaussian state is characterized by a mean
    vector of length 2N and a covariance matrix of size 2N x 2N."

UNITARY_ERROR = lazy"Either the input symplectic matrix is not square 
    or there is a dimension mismatch. An arbitrary N-mode Gaussian 
    unitary operator is characterized by a displacement vector of 
    length 2N and a symplectic matrix of size 2N x 2N." 

CHANNEL_ERROR = lazy"Either an input matrix is not square or 
    there is a dimension mismatch. An arbitrary N-mode Gaussian 
    channel is characterized by a displacement vector of length
    2N, a transform matrix of size 2N x 2N, and a noise matrix
    of size 2N x 2N." 

STELLAR_ERROR = lazy"The core tensor must carry one index per mode of the Gaussian factor."

CORE_ERROR = lazy"The core tensor must have at least one nonzero amplitude."

RANK_ERROR = lazy"Only a stellar state of rank zero is a Gaussian state."

STELLAR_PURITY_ERROR = lazy"Only a pure Gaussian state has a stellar decomposition."

STELLAR_PTRACE_ERROR = lazy"The reduction of a stellar state is generally mixed and
    is not itself a stellar state. Convert to a GaussianState when the rank is zero,
    or work with the density operator directly."

ACTION_ERROR = lazy"The number of modes for the Gaussian operator
    does not match the number of modes for the Gaussian state, or the symplectic representations for
    the two types are different."

GENERALDYNE_ERROR = lazy"The number of modes for the measurement state
    does not match the number of modes for the projected subsystem."

WIGNER_ERROR = lazy"The length of your input array does not match the
    number of modes for the Gaussian state."

HEATMAP_ERROR = lazy"The input Gaussian state describes more than one mode. 
    A heat map visualization for a multi-mode Gaussian state is not possible."

HBAR_ERROR = lazy"There are at least two different conventions for `ħ` used in this operation. Make sure
    that each Gaussian object has the same convention for `ħ`."

INDEX_ERROR = lazy"The provided indices must all be positive integer values that are no greater than the number of modes encompassed within the Gaussian state."
