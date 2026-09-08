module GrassmannTensorNetworks

using HDF5: create_group, delete_object, h5open
using LinearAlgebra
using Printf
using Random
using Shuffle
using TensorOperations
using TupleTools
using TupleTools: flatten, permute, insertat, insertafter, deleteat, getindices
using VectorInterface

include("grassmann.jl")
include("fermionsign.jl")

global_sign = auto_sign

include("base.jl")
include("linalg.jl")
include("contract.jl")
include("fusion.jl")
include("decomp.jl")
include("tupletools.jl")
include("auxiliary.jl")
include("algorithms.jl")

export Grassmann, AbstractGrassmann, GrassmannScalar, GrassmannVector, GrassmannMatrix
export _fixed_parity_blocks
export even, odd, data, index_type, tensor_parity, tensor_rank, scalar
export index_conjugation, nonzero_pairs, nonzero_keys, nonzero_vals

export auto_sign, trivial_sign, add_parity_sign, add_perm_sign

export trace, contract
export fuse
export gsvd, gevd, gortho
export save, load

export Nmod, compare_weights, prepare_bond_weight
export Square_GPEPS, absorb_Schmidt_weights, square_gpeps_parameter_count
export SpinlessFermionModel, InteractingSpinlessFermion, TJModel, HubbardModel, BilayerHubbardModel, n_site, nn_bond, gate
export Gross_Neveu_Wilson_model, PartitionFunctionTensor
export reduced_tensor, reduced_tensor_alpha, reduced_tensor_vbond, reduced_tensor_hbond
export CTMRGEnv, run_GCTMRG!, find_maxiter, read_CTMRG_env
export compute_exp_site, compute_exp_hbond, compute_exp_vbond
export NestedLayout, NestedNetwork
export nested_network, adapt_CTMRG, initialize_nested_environment, run_nested_GCTMRG!
export nested_x_operator, nested_y_operator
export compute_nested_exp_site, compute_nested_exp_hbond, compute_nested_exp_vbond
export Grassmann_SU

end
