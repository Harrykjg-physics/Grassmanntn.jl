using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
Pkg.instantiate()

using GrassmannTensorNetworks
using Random

function run_SU_Square_Bilayer_Hubbard(
    t::Float64, 
    U::Float64, 
    μ::Float64,
    J::Float64,
    Dbond::Int64, 
    Lx::Int, 
    Ly::Int, 
    iter_vec::Vector{Int}, 
    tol_vec::Vector{Float64}, 
    dτ_vec::Vector{Float64})

    # Initialize a random PEPS with identity Schmidt weights
    peps = Square_GPEPS(16, 8, Dbond, Lx, Ly, Float64, true; weight_init=:identity)
    # Initialize a PEPS from an optimized state
    # peps = load("tensor_file", "iter3000" * "_δτ0.01", Square_GPEPS)
    model = BilayerHubbardModel(t, U, μ, J)

    for (dτ, iter, tol) in zip(dτ_vec, iter_vec, tol_vec)
        G = gate(model, dτ)
        peps = Grassmann_SU(G, peps, dτ, Dbond; su_iter=iter, su_tol=tol, save_iter=100, average_trunc=true, start=0)
    end

    return peps
end

t = 1.0
U = 0.0
μ = 0.0
J = 6.0
Dbond = 4
Lx = 2
Ly = 2

Random.seed!(2928528937)

GrassmannTensorNetworks.global_sign = auto_sign

run_SU_Square_Bilayer_Hubbard(t, U, μ, J, Dbond, Lx, Ly, [100, 100, 100], [1e-6, 1e-10, 1e-12], [1e-2, 1e-3, 1e-4])
