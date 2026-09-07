using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
Pkg.instantiate()

using GrassmannTensorNetworks
using LinearAlgebra
using Optim
using Printf
using Random
using Zygote

function normalize_params!(params::AbstractVector{<:Real})

    scale = norm(params) / sqrt(length(params))
    scale > eps(eltype(params)) && (params ./= scale)
    
    return params
end

function normalized_params(params::AbstractVector{<:Real})

    scale = norm(params) / sqrt(length(params))

    return collect(params) ./ (scale + eps(Float64))
end

function update_ctmrg_env!(
    Dphys::Int,
    Dphys_even::Int,
    Dbond::Int,
    Lx::Int,
    Ly::Int,
    params::AbstractVector{Q1},
    h_bond::Grassmann{Q2, 4},
    χ::Int,
    ctmrg_iter::Int;
    env::Union{Nothing, CTMRGEnv}=nothing,
    ctmrg_tol::Float64=1e-12,
    average_trunc::Bool=true,
    verbosity::Int=0) where {Q1, Q2}

    peps = Square_GPEPS(Dphys, Dphys_even, Dbond, Lx, Ly, params, false)
    T_bulk = reduced_tensor(peps)
    ctmrg_env = (env === nothing ? CTMRGEnv(T_bulk, χ, div(χ, 2)) : env)
    run_GCTMRG!(peps, T_bulk, h_bond, ctmrg_env, χ; 
    ctmrg_iter=ctmrg_iter, ctmrg_tol=ctmrg_tol, average_trunc=average_trunc, verbosity=verbosity, save_iter=0)

    return ctmrg_env
end

function compute_energy(
    Dphys::Int,
    Dphys_even::Int,
    Dbond::Int,
    Lx::Int,
    Ly::Int,
    params::AbstractVector{Q1},
    h_bond::Grassmann{Q2, 4},
    ctmrg_env::CTMRGEnv{Q1}) where {Q1, Q2}

    peps = Square_GPEPS(Dphys, Dphys_even, Dbond, Lx, Ly, params, false)
    T_square_mat = reduced_tensor(peps)
    _, Eh = compute_exp_hbond(T_square_mat, peps, h_bond, ctmrg_env)
    _, Ev = compute_exp_vbond(T_square_mat, peps, h_bond, ctmrg_env)
    Es_avg = (sum(Eh) + sum(Ev))/(size(Eh, 1) * size(Eh, 2))

    return Es_avg
end

function optimize_fixed_env!(
    Dphys::Int,
    Dphys_even::Int,
    Dbond::Int,
    Lx::Int,
    Ly::Int,
    params::Vector{Q1},
    h_bond::Grassmann{Q2, 4},
    env::CTMRGEnv{Q1};
    optim_iter::Int=5,
    optim_method=Optim.LBFGS()) where {Q1, Q2}

    objective = x -> real(compute_energy(Dphys, Dphys_even, Dbond, Lx, Ly, normalized_params(x), h_bond, env))
    gradient! = (G, x) -> copyto!(G, Zygote.gradient(objective, x)[1])
    energy_before = objective(params)

    result = Optim.optimize(
        objective, 
        gradient!, 
        copy(params), 
        optim_method, 
        Optim.Options(iterations=optim_iter, show_trace=false))

    copyto!(params, normalized_params(Optim.minimizer(result)))
    energy_after = objective(params)

    return energy_before, energy_after, result
end

function CTMRG_Square_Hubbard_AD(
    t::Float64,
    U::Float64,
    μ::Float64,
    Dbond::Int,
    Lx::Int,
    Ly::Int,
    χ::Int,
    ctmrg_iter::Int;
    outer_ad_iter::Int=8,
    inner_optim_iter::Int=2,
    ad_tol::Float64=1e-6,
    seed::Int=1234,
    average_trunc::Bool=true,
    ctmrg_tol::Float64=1e-10,
    verbosity::Int=0)

    Dphys = 4
    Dphys_even = 2

    Random.seed!(seed)
    nparams = square_gpeps_parameter_count(Dphys, Dphys_even, Dbond, Lx, Ly)
    params = normalize_params!(randn(nparams))
    h_bond = nn_bond(HubbardModel(t, U, μ))
    history = NamedTuple[]
    env = nothing

    for ad_iter in 1:outer_ad_iter

        @printf("Updating CTMRG environment %d/%d...\n", ad_iter, outer_ad_iter)
        flush(stdout)

        env = update_ctmrg_env!(
            Dphys, Dphys_even, Dbond, Lx, Ly, params, h_bond, χ, ctmrg_iter;
            env=env,
            ctmrg_tol=ctmrg_tol,
            average_trunc=average_trunc,
            verbosity=verbosity)

        energy_before, energy_after, opt_result = optimize_fixed_env!(
            Dphys, Dphys_even, Dbond, Lx, Ly, params, h_bond, env;
            optim_iter=inner_optim_iter)

        grad_norm = Optim.g_residual(opt_result)

        push!(history, (
            ad_iter=ad_iter,
            iterations=Optim.iterations(opt_result),
            energy=energy_before,
            new_energy=energy_after,
            grad_norm=grad_norm,
            converged=Optim.converged(opt_result)))

        @printf(
            "AD outer %2d | iters %3d | E %.12f -> %.12f | |g| = %.6e%s\n", 
            ad_iter, Optim.iterations(opt_result), energy_before, energy_after, 
            grad_norm, Optim.converged(opt_result) ? " (converged)" : "")
        
        flush(stdout)

        grad_norm < ad_tol && break
    end

    env = update_ctmrg_env!(
        Dphys, Dphys_even, Dbond, Lx, Ly, params, h_bond, χ, ctmrg_iter;
        env=env,
        ctmrg_tol=ctmrg_tol,
        average_trunc=average_trunc,
        verbosity=verbosity)
        
    final_energy = compute_energy(Dphys, Dphys_even, Dbond, Lx, Ly, params, h_bond, env)

    @printf("Final refreshed CTMRG energy: %.12f\n", final_energy)

    return (
        params=params,
        peps=Square_GPEPS(Dphys, Dphys_even, Dbond, Lx, Ly, params, false),
        env=env,
        history=history,
        final_energy=final_energy)
end

GrassmannTensorNetworks.global_sign = auto_sign

t = 1.0
U = 8.0
μ = U/2
Dbond = 2
Lx = 2
Ly = 2
χ = 20
ctmrg_iter = 100

result = CTMRG_Square_Hubbard_AD(t, U, μ, Dbond, Lx, Ly, χ, ctmrg_iter; outer_ad_iter=2, inner_optim_iter=2, verbosity=0)

@printf("Stored %d optimization records.\n", length(result.history))
