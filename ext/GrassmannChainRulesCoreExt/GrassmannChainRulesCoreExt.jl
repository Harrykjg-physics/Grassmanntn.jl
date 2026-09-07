module GrassmannChainRulesCoreExt

using ChainRulesCore
using LinearAlgebra
using Zygote
using GrassmannTensorNetworks

import GrassmannTensorNetworks: Grassmann, AbstractGrassmann, GrassmannScalar, GrassmannVector, GrassmannMatrix,
    Square_GPEPS,
    nonzero_pairs, nonzero_keys, nonzero_vals, data,
    even, odd, index_type, tensor_parity, tensor_rank,
    trivial_sign, auto_sign,
    add_parity_sign, add_perm_sign,
    index_conjugation, prepare_range_dict,
    _parity_mask, _fixed_parity_blocks, _similar_arraytype,
    Nmod, CTMRGEnv,
    conjugate, fuse, calculate_sectors, calculate_fused_size, prepare_fused_info,
    trace, contract, gsvd, gevd, gortho, truncation, check_parity,
    reduced_tensor, reduced_tensor_alpha,
    compute_exp_hbond, compute_exp_vbond,
    compute_exp_hbond_alpha, compute_exp_vbond_alpha

import GrassmannTensorNetworks:
    NestedLayout, NestedNetwork,
    nested_network, nested_x_operator, nested_y_operator,
    compute_nested_exp_hbond, compute_nested_exp_vbond,
    _source_site,
    _layout_down_source, _layout_right_source,
    _layout_bra_site, _layout_ket_site, _layout_x_site,
    _nested_ket, _nested_bra,
    _nested_x,
    _bond_operator_gsvd, _nested_x_bond_operator,
    _nested_scalar_or_zero,
    _contract_nested_hpatch3, _contract_nested_vpatch3,
    _contract_nested_hpatch3_alpha, _contract_nested_vpatch3_alpha,
    global_sign

# AD rules for grassmann.jl (constructors, convert, index_conjugation)
include("grassmann.jl")

# AD rules for fermionsign.jl (auto_sign, trivial_sign, add_parity_sign, add_perm_sign)
include("fermionsign.jl")

# AD rules for base.jl (copy, +, -, *, /, real, conj, permutedims, sqrt, convert(Array, ...))
include("base.jl")

# AD rules for linalg.jl (log, norm, diag, transpose, inv, dot)
include("linalg.jl")

# AD rules for contract.jl (trace and contract)
include("contract.jl")

# AD rules for fusion.jl (fuse, split)
include("fusion.jl")

# AD rules for decomp.jl (gsvd, gevd, gortho)
include("decomp.jl")

# AD rules for CTMRG measurements
include("measurements.jl")

include(joinpath(
    @__DIR__, "..", "..", "algorithms", "Nested_CTMRG", "nested_chainrules.jl"
))

end
