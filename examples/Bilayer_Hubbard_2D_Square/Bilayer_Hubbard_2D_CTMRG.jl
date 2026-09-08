using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
Pkg.instantiate()

using GrassmannTensorNetworks

function run_CTMRG_Square_Bilayer_Hubbard(
    t::Float64, 
    U::Float64, 
    μ::Float64,
    J::Float64,
    peps_filename::String,
    peps_param_str::String,  
    χ::Int, 
    ctmrg_iter::Int; 
    load_env::String="random")

    ##################### Load the Square GPEPS from the ground state optimization #####################

    wpeps = load(peps_filename, peps_param_str, Square_GPEPS)
    peps = absorb_Schmidt_weights(wpeps)

    ##################### Introduce the bond Hamiltonian  #####################

    model = BilayerHubbardModel(t, U, μ, J)
    H_nn_bond = nn_bond(model)
    N1_site = n1_site(model)
    N2_site = n2_site(model)
    N_site = n_site(model)

    ##################### Construct reduced tensors and the site impurity tensor from Square GPEPS #####################

    T_square_mat = reduced_tensor(peps)
    T_n1_imp_mat = reduced_tensor(peps, N1_site)
    T_n2_imp_mat = reduced_tensor(peps, N2_site)
    T_n_imp_mat = reduced_tensor(peps, N_site)

    ##################### Running Grassmann CTMRG to compute environment tensors #####################

    ctmrg_env = (load_env == "random" ? CTMRGEnv(T_square_mat, χ, Int(χ/2)) : load("ctmrg_env", load_env, CTMRGEnv))

    run_GCTMRG!(T_square_mat, T_n_imp_mat, ctmrg_env, χ; 
    ctmrg_iter=ctmrg_iter, ctmrg_tol=1e-12, average_trunc=true, 
    verbosity=2, save_iter=20, save_filename="ctmrg_env")

    """
    # Monitor the convergence of the ground state energy
    run_GCTMRG!(peps, T_square_mat, H_nn_bond, ctmrg_env, χ; 
    ctmrg_iter=ctmrg_iter, ctmrg_tol=1e-12, average_trunc=true, 
    verbosity=2, save_iter=20, save_filename="ctmrg_env")
    """

    _, ns1 = compute_exp_site(T_square_mat, T_n1_imp_mat, ctmrg_env)
    ns1_avg = sum(ns1)/(size(ns1, 1) * size(ns1, 2))
    _, ns2 = compute_exp_site(T_square_mat, T_n2_imp_mat, ctmrg_env)
    ns2_avg = sum(ns2)/(size(ns2, 1) * size(ns2, 2))
    _, ns = compute_exp_site(T_square_mat, T_n_imp_mat, ctmrg_env)
    ns_avg = sum(ns)/(size(ns, 1) * size(ns, 2))
    _, Eh = compute_exp_hbond(T_square_mat, peps, H_nn_bond, ctmrg_env)
    _, Ev = compute_exp_vbond(T_square_mat, peps, H_nn_bond, ctmrg_env)
    Es_avg = (sum(Eh) + sum(Ev))/(size(Eh, 1) * size(Eh, 2))

    println("Average ground energy per site: $Es_avg at U = $U, μ = $μ, χ = $χ")
    save("exp_ctmrg", "χ$χ", 
    "ns1", ns1, "ns1_avg", ns1_avg, 
    "ns2", ns2, "ns2_avg", ns2_avg, 
    "ns", ns, "ns_avg", ns_avg, 
    "Eh", Eh, "Ev", Ev, "Es_avg", Es_avg)
end

t = 1.0
U = 0.0
μ = 0.0
J = 6.0
peps_filename = "tensor_file"
peps_param_str = "iter3000"*"_δτ0.0001"
ctmrg_iter = 100
load_env = "random"

GrassmannTensorNetworks.global_sign = auto_sign

run_CTMRG_Square_Bilayer_Hubbard(t, U, μ, J, peps_filename, peps_param_str, 16, ctmrg_iter; load_env=load_env)
run_CTMRG_Square_Bilayer_Hubbard(t, U, μ, J, peps_filename, peps_param_str, 32, ctmrg_iter; load_env=load_env)
run_CTMRG_Square_Bilayer_Hubbard(t, U, μ, J, peps_filename, peps_param_str, 48, ctmrg_iter; load_env=load_env)
