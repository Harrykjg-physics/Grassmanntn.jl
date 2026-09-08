using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
Pkg.instantiate()

using GrassmannTensorNetworks

function run_nested_CTMRG_Square_Hubbard(
    t::Float64,
    U::Float64,
    μ::Float64,
    peps_filename::String,
    peps_param_str::String,
    χ::Int,
    ctmrg_iter::Int;
    load_env::String="random")

    ##################### Load the Square GPEPS from the ground state optimization #####################

    wpeps = load(peps_filename, peps_param_str, Square_GPEPS)
    peps = absorb_Schmidt_weights(wpeps)

    ##################### Introduce the bond Hamiltonian  #####################

    model = HubbardModel(t, U, μ)
    H_nn_bond = nn_bond(model)
    N_site = n_site(model)
    H_nn_bonds = fill(H_nn_bond, size(peps))
    N_sites = fill(N_site, size(peps))

    ##################### Construct reduced tensors in nested representation from Square GPEPS #####################

    nested = adapt_CTMRG(nested_network(peps))

    ##################### Running Grassmann CTMRG to compute environment tensors #####################

    ctmrg_env = (load_env == "random" ? initialize_nested_environment(nested, χ, div(χ, 2)) : 
    load("ctmrg_nested_env", load_env, CTMRGEnv))

    run_nested_GCTMRG!(peps, N_sites, nested, ctmrg_env, χ; ctmrg_iter=ctmrg_iter, ctmrg_tol=1e-12, average_trunc=true,
    verbosity=2, save_iter=20, save_filename="ctmrg_nested_env")

    """
    run_nested_GCTMRG!(peps, H_nn_bonds, nested, ctmrg_env, χ; ctmrg_iter=ctmrg_iter, ctmrg_tol=1e-12, average_trunc=true,
    verbosity=2, save_iter=20, save_filename="ctmrg_nested_env")
    """

    _, ns = compute_nested_exp_site(nested, peps, N_sites, ctmrg_env)
    ns_avg = sum(ns) / length(ns)
    _, Eh = compute_nested_exp_hbond(nested, peps, H_nn_bonds, ctmrg_env)
    _, Ev = compute_nested_exp_vbond(nested, peps, H_nn_bonds, ctmrg_env)
    Es_avg = (sum(Eh) + sum(Ev)) / length(Eh)
    println("Average ground energy per site: $Es_avg at U = $U, μ = $μ, χ = $χ")
    save("exp_nested_ctmrg", "χ$χ", "ns", ns, "ns_avg", ns_avg, "Eh", Eh, "Ev", Ev, "Es_avg", Es_avg)
end

t = 1.0
U = 2.0
μ = U/2
peps_filename = "tensor_file"
peps_param_str = "iter1000"*"_δτ0.01"
ctmrg_iter = 100
load_env = "random"

GrassmannTensorNetworks.global_sign = auto_sign

run_nested_CTMRG_Square_Hubbard(t, U, μ, peps_filename, peps_param_str, 16, ctmrg_iter; load_env=load_env)
run_nested_CTMRG_Square_Hubbard(t, U, μ, peps_filename, peps_param_str, 32, ctmrg_iter; load_env=load_env)
run_nested_CTMRG_Square_Hubbard(t, U, μ, peps_filename, peps_param_str, 48, ctmrg_iter; load_env=load_env)
