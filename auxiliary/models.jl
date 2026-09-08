
abstract type AbstractModel end

############################# Non-relativistic fermions #############################

"""
2D Spinless Fermion model on the square lattice

H = ∑_(⟨i,j⟩) [ -t (c†_{i} c_{j} + h.c.)  - γ (c†_{i} c_{j}† + h.c.) ]
    - 2λ ∑_i c†_{i} c_{i}

H_nn_bond = - t ( c†i ⊗ cj + c†j ⊗ ci) 
            - γ ( c†i ⊗ c†j + cj ⊗ ci) 
            - λ/2 ( c†i ci ⊗ Ij + Ii ⊗ c†j cj) 
"""

struct SpinlessFermionModel{T<:Real} <: AbstractModel
    t::T
    γ::T
    λ::T
end

function SpinlessFermionModel(t::Real, γ::Real, λ::Real)
    t, γ, λ = promote(t, γ, λ)
    return SpinlessFermionModel{typeof(t)}(t, γ, λ)
end

function n_site_Fock_basis(model::SpinlessFermionModel{T}) where {T}
    n_coef = zeros(T, (2, 2))
    n_coef[2, 2] = 1
    return n_coef
end

function n_site(model::SpinlessFermionModel)
    n_coef = n_site_Fock_basis(model)
    n_site_out = Grassmann(n_coef, (2, 2), (1, 1), (:out, :in))
end

function nn_bond_Fock_basis(model::SpinlessFermionModel{T}) where {T}

    t = model.t
    γ = model.γ
    λ = model.λ
    
    H_coef = zeros(T, (2, 2, 2, 2))
    # < 1ᵢ0ⱼ | c†i ⊗ cj | 0ᵢ1ⱼ > = 1; < 0ᵢ1ⱼ | c†j ⊗ ci | 1ᵢ0ⱼ > = 1
    H_coef[2, 1, 1, 2] = -t; H_coef[1, 2, 2, 1] = -t
    # < 1ᵢ1ⱼ | c†i ⊗ c†j | 0ᵢ0ⱼ > = 1; < 0ᵢ0ⱼ | cj ⊗ ci | 1ᵢ1ⱼ > = 1
    H_coef[2, 2, 1, 1] = -γ; H_coef[1, 1, 2, 2] = -γ
    # < 1ᵢ0ⱼ | c†i ci ⊗ Ij | 1ᵢ0ⱼ > = 1; < 1ᵢ1ⱼ | c†i ci ⊗ Ij | 1ᵢ1ⱼ > = 1
    H_coef[2, 1, 2, 1] = -λ/2; H_coef[2, 2, 2, 2] = -λ/2
    # < 0ᵢ1ⱼ | Ii ⊗ c†j cj | 0ᵢ1ⱼ > = 1; < 1ᵢ1ⱼ | Ii ⊗ c†j cj | 1ᵢ1ⱼ > = 1
    H_coef[1, 2, 1, 2] = -λ/2; H_coef[2, 2, 2, 2] += -λ/2

    return H_coef
end

function nn_bond(model::SpinlessFermionModel)

    H_coef = nn_bond_Fock_basis(model)
    nn_bond_out1 = Grassmann(H_coef, (2, 2, 2, 2), (1, 1, 1, 1), (:out, :out, :in, :in))
    nn_bond_out2 = add_perm_sign(nn_bond_out1, (1, 2, 4, 3))
end

function gate(model::SpinlessFermionModel, dτ::Real)

    H_coef = nn_bond_Fock_basis(model)
    H_coef_mat = reshape(H_coef, (4, 4))
    G_coef_mat = exp(-dτ * H_coef_mat)
    G_coef = reshape(G_coef_mat, (2, 2, 2, 2))
    G = Grassmann(G_coef, (2, 2, 2, 2), (1, 1, 1, 1), (:out, :out, :in, :in))
    G_out = add_perm_sign(G, (1, 2, 4, 3))
end

"""
2D interacting spinless fermion model on the square lattice

H = ∑_(⟨i,j⟩) [ -t (c†_{i} c_{j} + h.c.) ]
    - μ ∑_i c†_{i} c_{i}
    + V ∑_(⟨i,j⟩) c†_{i} c_{i} c†_{j} c_{j}

H_nn_bond = - t (c†i ⊗ cj + c†j ⊗ ci)
            - μ/4 (c†i ci ⊗ Ij + Ii ⊗ c†j cj)
            + V (c†i ci ⊗ c†j cj)
"""

struct InteractingSpinlessFermion{T<:Real} <: AbstractModel
    t::T
    μ::T
    V::T
end

function InteractingSpinlessFermion(t::Real, μ::Real, V::Real)
    t, μ, V = promote(t, μ, V)
    return InteractingSpinlessFermion{typeof(t)}(t, μ, V)
end

function n_site_Fock_basis(model::InteractingSpinlessFermion{T}) where {T}
    n_coef = zeros(T, (2, 2))
    n_coef[2, 2] = 1
    return n_coef
end

function n_site(model::InteractingSpinlessFermion)
    n_coef = n_site_Fock_basis(model)
    n_site_out = Grassmann(n_coef, (2, 2), (1, 1), (:out, :in))
end

function nn_bond_Fock_basis(model::InteractingSpinlessFermion{T}) where {T}

    t = model.t
    μ = model.μ
    V = model.V

    H_coef = zeros(T, (2, 2, 2, 2))
    # < 1ᵢ0ⱼ | c†i ⊗ cj | 0ᵢ1ⱼ > = 1; < 0ᵢ1ⱼ | c†j ⊗ ci | 1ᵢ0ⱼ > = 1
    H_coef[2, 1, 1, 2] = -t; H_coef[1, 2, 2, 1] = -t
    # < 1ᵢ0ⱼ | c†i ci ⊗ Ij | 1ᵢ0ⱼ > = 1; < 1ᵢ1ⱼ | c†i ci ⊗ Ij | 1ᵢ1ⱼ > = 1
    H_coef[2, 1, 2, 1] = -μ/4; H_coef[2, 2, 2, 2] = -μ/4
    # < 0ᵢ1ⱼ | Ii ⊗ c†j cj | 0ᵢ1ⱼ > = 1; < 1ᵢ1ⱼ | Ii ⊗ c†j cj | 1ᵢ1ⱼ > = 1
    H_coef[1, 2, 1, 2] = -μ/4; H_coef[2, 2, 2, 2] += -μ/4
    # < 1ᵢ1ⱼ | c†i ci ⊗ c†j cj | 1ᵢ1ⱼ > = 1
    H_coef[2, 2, 2, 2] += V

    return H_coef
end

function nn_bond(model::InteractingSpinlessFermion)

    H_coef = nn_bond_Fock_basis(model)
    nn_bond_out1 = Grassmann(H_coef, (2, 2, 2, 2), (1, 1, 1, 1), (:out, :out, :in, :in))
    nn_bond_out2 = add_perm_sign(nn_bond_out1, (1, 2, 4, 3))
end

function gate(model::InteractingSpinlessFermion, dτ::Real)

    H_coef = nn_bond_Fock_basis(model)
    H_coef_mat = reshape(H_coef, (4, 4))
    G_coef_mat = exp(-dτ * H_coef_mat)
    G_coef = reshape(G_coef_mat, (2, 2, 2, 2))
    G = Grassmann(G_coef, (2, 2, 2, 2), (1, 1, 1, 1), (:out, :out, :in, :in))
    G_out = add_perm_sign(G, (1, 2, 4, 3))
end

"""
2D t-J model on the square lattice

H = -t ∑_(⟨i,j⟩,σ) (c̃†_{iσ} c̃_{jσ} + h.c.)
    + J ∑_(⟨i,j⟩) (S_i ⋅ S_j - 1/4 n_i n_j)
    - μ ∑_i n_i

The local basis is (0, ↑, ↓), with double occupancy projected out.

H_nn_bond = -t projected hopping
            + J (S_i ⋅ S_j - 1/4 n_i n_j)
            - μ/4 (n_i ⊗ Ij + Ii ⊗ n_j)
"""

struct TJModel{T<:Real} <: AbstractModel
    t::T
    J::T
    μ::T
end

function TJModel(t::Real, J::Real, μ::Real)
    t, J, μ = promote(t, J, μ)
    return TJModel{typeof(t)}(t, J, μ)
end

function n_site_Fock_basis(model::TJModel{T}) where {T}
    n_coef = zeros(T, (3, 3))
    n_coef[2, 2] = 1
    n_coef[3, 3] = 1
    return n_coef
end

function n_site(model::TJModel)
    n_coef = n_site_Fock_basis(model)
    n_site_out = Grassmann(n_coef, (3, 3), (1, 1), (:out, :in))
end

function nn_bond_Fock_basis(model::TJModel{T}) where {T}

    t = model.t
    J = model.J
    μ = model.μ

    H_coef = zeros(T, (3, 3, 3, 3))
    # < ↑ᵢ0ⱼ | c̃†i↑ ⊗ c̃j↑ | 0ᵢ↑ⱼ > = 1; < 0ᵢ↑ⱼ | c̃†j↑ ⊗ c̃i↑ | ↑ᵢ0ⱼ > = 1
    H_coef[2, 1, 1, 2] = -t; H_coef[1, 2, 2, 1] = -t
    # < ↓ᵢ0ⱼ | c̃†i↓ ⊗ c̃j↓ | 0ᵢ↓ⱼ > = 1; < 0ᵢ↓ⱼ | c̃†j↓ ⊗ c̃i↓ | ↓ᵢ0ⱼ > = 1
    H_coef[3, 1, 1, 3] = -t; H_coef[1, 3, 3, 1] = -t
    # Onsite chemical potential, split over the four square-lattice bonds touching each site.
    H_coef[2, 1, 2, 1] = -μ/4; H_coef[3, 1, 3, 1] = -μ/4
    H_coef[1, 2, 1, 2] = -μ/4; H_coef[1, 3, 1, 3] = -μ/4
    H_coef[2, 2, 2, 2] = -μ/2; H_coef[3, 3, 3, 3] = -μ/2
    H_coef[2, 3, 2, 3] = -μ/2; H_coef[3, 2, 3, 2] = -μ/2
    # S_i ⋅ S_j - 1/4 n_i n_j: only antiparallel spins have diagonal -1/2 and spin-flip +1/2.
    H_coef[2, 3, 2, 3] += -J/2; H_coef[3, 2, 3, 2] += -J/2
    H_coef[3, 2, 2, 3] += J/2; H_coef[2, 3, 3, 2] += J/2

    return H_coef
end

function nn_bond(model::TJModel)

    H_coef = nn_bond_Fock_basis(model)
    nn_bond_out1 = Grassmann(H_coef, (3, 3, 3, 3), (1, 1, 1, 1), (:out, :out, :in, :in))
    nn_bond_out2 = add_perm_sign(nn_bond_out1, (1, 2, 4, 3))
end

function gate(model::TJModel, dτ::Real)

    H_coef = nn_bond_Fock_basis(model)
    H_coef_mat = reshape(H_coef, (9, 9))
    G_coef_mat = exp(-dτ * H_coef_mat)
    G_coef = reshape(G_coef_mat, (3, 3, 3, 3))
    G = Grassmann(G_coef, (3, 3, 3, 3), (1, 1, 1, 1), (:out, :out, :in, :in))
    G_out = add_perm_sign(G, (1, 2, 4, 3))
end

"""
2D Fermi-Hubbard model on the square lattice

H = -t ∑_(⟨i,j⟩,σ) (c†_{iσ} c_{jσ} + h.c.) 
    + U ∑_i n_{i↑}n_{i↓} 
    - μ ∑_i(n_i↑ + n_i↓)

H_nn_bond = - t( c†i↑ ⊗ cj↑ + c†j↑ ⊗ ci↑  + c†i↓ ⊗ cj↓ + c†j↓ ⊗ ci↓) 
         + U/4 (ni↑ ni↓ ⊗ Ij + Ii ⊗ nj↑ nj↓) 
         - μ/4 (ni↑ ⊗ Ij + ni↓ ⊗ Ij + Ii ⊗ nj↑ + Ii ⊗ nj↓)
"""

struct HubbardModel{T<:Real} <: AbstractModel
    t::T
    U::T
    μ::T
end

function HubbardModel(t::Real, U::Real, μ::Real)
    t, U, μ = promote(t, U, μ)
    return HubbardModel{typeof(t)}(t, U, μ)
end

function nu_site_Fock_basis(model::HubbardModel{T}) where {T}
    
    Nu_coef = zeros(T, (4, 4))
    # < D | c†↑ c↑ | D > = 1
    Nu_coef[2, 2] = 1
    # < ↑ | c†↑ c↑ | ↑ > = 1 
    Nu_coef[3, 3] = 1

    return Nu_coef
end

function nd_site_Fock_basis(model::HubbardModel{T}) where {T}
    
    Nd_coef = zeros(T, (4, 4))
    # < D | c†↓ c↓ | D > = 1
    Nd_coef[2, 2] = 1
    # < ↓ | c†↓ c↓ | ↓ > = 1 
    Nd_coef[4, 4] = 1

    return Nd_coef
end

function n_site_Fock_basis(model::HubbardModel{T}) where {T}
    nu_site_Fock_basis(model) + nd_site_Fock_basis(model)
end

function n_site(model::HubbardModel)
    n_coef = n_site_Fock_basis(model)
    n_site_out = Grassmann(n_coef, (4, 4), (2, 2), (:out, :in))
end

function nn_bond_Fock_basis(model::HubbardModel{T}) where {T}

    t = model.t
    U = model.U
    μ = model.μ
    
    H_coef = zeros(T, (4, 4, 4, 4))
    # < ↑ᵢ0ⱼ | c†i↑ ⊗ cj↑ | 0ᵢ↑ⱼ > = 1; < Dᵢ0ⱼ | c†i↑ ⊗ cj↑ | ↓ᵢ↑ⱼ > = -1
    # < ↑ᵢ↓ⱼ | c†i↑ ⊗ cj↑ | 0ᵢDⱼ > = 1; < Dᵢ↓ⱼ | c†i↑ ⊗ cj↑ | ↓ᵢDⱼ > = -1
    H_coef[3, 1, 1, 3] = -t; H_coef[2, 1, 4, 3] = t
    H_coef[3, 4, 1, 2] = -t; H_coef[2, 4, 4, 2] = t
    # < 0ᵢ↑ⱼ | c†j↑ ⊗ ci↑ | ↑ᵢ0ⱼ > = 1; < ↓ᵢ↑ⱼ | c†j↑ ⊗ ci↑ | Dᵢ0ⱼ > = -1
    # < 0ᵢDⱼ | c†j↑ ⊗ ci↑ | ↑ᵢ↓ⱼ > = 1; < ↓ᵢDⱼ | c†j↑ ⊗ ci↑ | Dᵢ↓ⱼ > = -1
    H_coef[1, 3, 3, 1] = -t; H_coef[4, 3, 2, 1] = t
    H_coef[1, 2, 3, 4] = -t; H_coef[4, 2, 2, 4] = t
    # < ↓ᵢ0ⱼ | c†i↓ ⊗ cj↓ | 0ᵢ↓ⱼ > = 1; < Dᵢ0ⱼ | c†i↓ ⊗ cj↓ | ↑ᵢ↓ⱼ > = 1
    # < ↓ᵢ↑ⱼ | c†i↓ ⊗ cj↓ | 0ᵢDⱼ > = -1; < Dᵢ↑ⱼ | c†i↓ ⊗ cj↓ | ↑ᵢDⱼ > = -1
    H_coef[4, 1, 1, 4] = -t; H_coef[2, 1, 3, 4] = -t
    H_coef[4, 3, 1, 2] = t; H_coef[2, 3, 3, 2] = t
    # < 0ᵢ↓ⱼ | c†j↓ ⊗ ci↓ | ↓ᵢ0ⱼ > = 1; < ↑ᵢ↓ⱼ | c†j↓ ⊗ ci↓ | Dᵢ0ⱼ > = 1
    # < 0ᵢDⱼ | c†j↓ ⊗ ci↓ | ↓ᵢ↑ⱼ > = -1; < ↑ᵢDⱼ | c†j↓ ⊗ ci↓ | Dᵢ↑ⱼ > = -1
    H_coef[1, 4, 4, 1] = -t; H_coef[3, 4, 2, 1] = -t
    H_coef[1, 2, 4, 3] = t; H_coef[3, 2, 2, 3] = t
    # < Dᵢ~ⱼ | ni↑ ni↓ ⊗ Ij | Dᵢ~ⱼ > = 1
    H_coef[2, 1, 2, 1] += U/4; H_coef[2, 2, 2, 2] += U/4
    H_coef[2, 3, 2, 3] += U/4; H_coef[2, 4, 2, 4] += U/4
    # < ~ᵢDⱼ | Ii ⊗ nj↑ nj↓ | ~ᵢDⱼ > = 1
    H_coef[1, 2, 1, 2] += U/4; H_coef[2, 2, 2, 2] += U/4
    H_coef[3, 2, 3, 2] += U/4; H_coef[4, 2, 4, 2] += U/4
    # < Dᵢ~ⱼ | ni↑ ⊗ Ij | Dᵢ~ⱼ > = 1
    H_coef[2, 1, 2, 1] -= μ/4; H_coef[2, 2, 2, 2] -= μ/4 
    H_coef[2, 3, 2, 3] -= μ/4; H_coef[2, 4, 2, 4] -= μ/4
    # < ↑ᵢ~ⱼ | ni↑ ⊗ Ij | ↑ᵢ~ⱼ > = 1
    H_coef[3, 1, 3, 1] -= μ/4; H_coef[3, 2, 3, 2] -= μ/4 
    H_coef[3, 3, 3, 3] -= μ/4; H_coef[3, 4, 3, 4] -= μ/4 
    # < Dᵢ~ⱼ | ni↓ ⊗ Ij | Dᵢ~ⱼ > = 1
    H_coef[2, 1, 2, 1] -= μ/4; H_coef[2, 2, 2, 2] -= μ/4
    H_coef[2, 3, 2, 3] -= μ/4; H_coef[2, 4, 2, 4] -= μ/4 
    # < ↓ᵢ~ⱼ | ni↓ ⊗ Ij | ↓ᵢ~ⱼ > = 1
    H_coef[4, 1, 4, 1] -= μ/4; H_coef[4, 2, 4, 2] -= μ/4
    H_coef[4, 3, 4, 3] -= μ/4; H_coef[4, 4, 4, 4] -= μ/4 
    # < ~ᵢDⱼ | Ii ⊗ nj↑ | ~ᵢDⱼ > = 1
    H_coef[1, 2, 1, 2] -= μ/4; H_coef[2, 2, 2, 2] -= μ/4 
    H_coef[3, 2, 3, 2] -= μ/4; H_coef[4, 2, 4, 2] -= μ/4 
    # < ~ᵢ↑ⱼ | Ii ⊗ nj↑ | ~ᵢ↑ⱼ > = 1
    H_coef[1, 3, 1, 3] -= μ/4; H_coef[2, 3, 2, 3] -= μ/4 
    H_coef[3, 3, 3, 3] -= μ/4; H_coef[4, 3, 4, 3] -= μ/4 
    # < ~ᵢDⱼ | Ii ⊗ nj↓ | ~ᵢDⱼ > = 1
    H_coef[1, 2, 1, 2] -= μ/4; H_coef[2, 2, 2, 2] -= μ/4
    H_coef[3, 2, 3, 2] -= μ/4; H_coef[4, 2, 4, 2] -= μ/4
    # < ~ᵢ↓ⱼ | Ii ⊗ nj↓ | ~ᵢ↓ⱼ > = 1
    H_coef[1, 4, 1, 4] -= μ/4; H_coef[2, 4, 2, 4] -= μ/4
    H_coef[3, 4, 3, 4] -= μ/4; H_coef[4, 4, 4, 4] -= μ/4

    return H_coef
end

function nn_bond(model::HubbardModel)

    H_coef = nn_bond_Fock_basis(model)
    nn_bond_out1 = Grassmann(H_coef, (4, 4, 4, 4), (2, 2, 2, 2), (:out, :out, :in, :in))
    nn_bond_out2 = add_perm_sign(nn_bond_out1, (1, 2, 4, 3))
end

function gate(model::HubbardModel, dτ::Real)

    H_coef = nn_bond_Fock_basis(model)
    H_coef_mat = reshape(H_coef, (16, 16))
    G_coef_mat = exp(-dτ * H_coef_mat)
    G_coef = reshape(G_coef_mat, (4, 4, 4, 4))
    G = Grassmann(G_coef, (4, 4, 4, 4), (2, 2, 2, 2), (:out, :out, :in, :in))
    G_out = add_perm_sign(G, (1, 2, 4, 3))
end

"""
2D bilayer Fermi-Hubbard model on the square lattice

H = -t ∑_(⟨i,j⟩,σα) (c†_{iσα} c_{jσα} + h.c.)
    + U ∑_i (n_{i↑α} - 1/2)(n_{i↓α} - 1/2)
    - μ ∑_(iα)(n_i↑α + n_i↓α)
    + J ∑_i S_{i1} ⋅ S_{i2}

The local basis is (|0>, |↑>, |↓>, |↑↓>) for each layer,
therefore, the local basis for the bilayer model is the tensor product of the two layers, which has 16 states :
(|01>, |↑1>, |↓1>, |↑↓1>) ⊗ (|02>, |↑2>, |↓2>, |↑↓2>) :

The first 8 states are even-parity states, which are numbered as follows :

    | 01 ; 02 > : 1
    | 01 ; D2 > : 2
    | D1 ; 02 > : 3
    | D1 ; D2 > : 4
    | ↑1 ; ↑2 > : 5
    | ↑1 ; ↓2 > : 6
    | ↓1 ; ↑2 > : 7
    | ↓1 ; ↓2 > : 8

The last 8 states are odd-parity states, which are numbered as follows :
    | 01 ; ↑2 > : 9
    | 01 ; ↓2 > : 10
    | D1 ; ↑2 > : 11
    | D1 ; ↓2 > : 12
    | ↑1 ; 02 > : 13
    | ↑1 ; D2 > : 14
    | ↓1 ; 02 > : 15
    | ↓1 ; D2 > : 16
"""

struct BilayerHubbardModel{T<:Real} <: AbstractModel
    t::T
    U::T
    μ::T
    J::T
end

function _bilayer_hubbard_model(t::Real, U::Real, μ::Real, J::Real)
    t, U, μ, J = promote(t, U, μ, J)
    T = typeof(one(typeof(t)) / 2)
    return BilayerHubbardModel{T}(T(t), T(U), T(μ), T(J))
end

function BilayerHubbardModel(t::T, U::T, μ::T, J::T) where {T<:Real}
    return _bilayer_hubbard_model(t, U, μ, J)
end

function BilayerHubbardModel(t::Real, U::Real, μ::Real, J::Real)
    return _bilayer_hubbard_model(t, U, μ, J)
end

const _BILAYER_HUBBARD_LOCAL_OCCUPATIONS = NTuple{4, Int}[
    (0, 0, 0, 0), 
    (0, 0, 1, 1), 
    (1, 1, 0, 0), 
    (1, 1, 1, 1), 
    (1, 0, 1, 0), 
    (1, 0, 0, 1), 
    (0, 1, 1, 0), 
    (0, 1, 0, 1), 
    (0, 0, 1, 0), 
    (0, 0, 0, 1), 
    (1, 1, 1, 0), 
    (1, 1, 0, 1), 
    (1, 0, 0, 0), 
    (1, 0, 1, 1), 
    (0, 1, 0, 0), 
    (0, 1, 1, 1)]

function _fermion_creation_matrix(
    ::Type{T}, 
    occupations::AbstractVector{<:Tuple}, 
    mode::Int) where {T}

    index_of = Dict(occ => i for (i, occ) in enumerate(occupations))
    C = zeros(T, length(occupations), length(occupations))
    @inbounds for (col, occ) in enumerate(occupations)
        occ[mode] == 1 && continue
        sign_exponent = mode == 1 ? 0 : sum(occ[k] for k in 1:(mode - 1))
        new_occ = collect(occ)
        new_occ[mode] = 1
        C[index_of[Tuple(new_occ)], col] = isodd(sign_exponent) ? -one(T) : one(T)
    end

    return C
end

function _bilayer_hubbard_local_number(::Type{T}, mode::Int) where {T}
    Cdag = _fermion_creation_matrix(T, _BILAYER_HUBBARD_LOCAL_OCCUPATIONS, mode)
    return Cdag * transpose(Cdag)
end

function _bilayer_hubbard_global_occupations()

    return NTuple{8, Int}[(left..., right...) 
    for right in _BILAYER_HUBBARD_LOCAL_OCCUPATIONS for left in _BILAYER_HUBBARD_LOCAL_OCCUPATIONS]
end

function nu1_site_Fock_basis(model::BilayerHubbardModel{T}) where {T}
    return _bilayer_hubbard_local_number(T, 1)
end

function nd1_site_Fock_basis(model::BilayerHubbardModel{T}) where {T}
    return _bilayer_hubbard_local_number(T, 2)
end

function nu2_site_Fock_basis(model::BilayerHubbardModel{T}) where {T}
    return _bilayer_hubbard_local_number(T, 3)
end

function nd2_site_Fock_basis(model::BilayerHubbardModel{T}) where {T}
    return _bilayer_hubbard_local_number(T, 4)
end

function n1_site_Fock_basis(model::BilayerHubbardModel{T}) where {T}
    return nu1_site_Fock_basis(model) + nd1_site_Fock_basis(model)
end

function n1_site(model::BilayerHubbardModel)
    n1_coef = n1_site_Fock_basis(model)
    n_site_out = Grassmann(n1_coef, (16, 16), (8, 8), (:out, :in))
end

function n2_site_Fock_basis(model::BilayerHubbardModel{T}) where {T}
    return nu2_site_Fock_basis(model) + nd2_site_Fock_basis(model)
end

function n2_site(model::BilayerHubbardModel)
    n2_coef = n2_site_Fock_basis(model)
    n_site_out = Grassmann(n2_coef, (16, 16), (8, 8), (:out, :in))
end

function n_site_Fock_basis(model::BilayerHubbardModel{T}) where {T}
    return nu1_site_Fock_basis(model) + nd1_site_Fock_basis(model) + 
    nu2_site_Fock_basis(model) + nd2_site_Fock_basis(model)
end

function n_site(model::BilayerHubbardModel)
    n_coef = n_site_Fock_basis(model)
    n_site_out = Grassmann(n_coef, (16, 16), (8, 8), (:out, :in))
end

function nn_bond_Fock_basis(model::BilayerHubbardModel{T}) where {T}

    occupations = _bilayer_hubbard_global_occupations()
    Cdag = [_fermion_creation_matrix(T, occupations, mode) for mode in 1:8]
    C = transpose.(Cdag)
    n = [Cdag[mode] * C[mode] for mode in 1:8]
    I256 = Matrix{T}(I, 256, 256)

    H_coef_mat = zeros(T, 256, 256)

    for mode in 1:4
        H_coef_mat .+= -model.t .* (Cdag[mode] * C[mode + 4] + Cdag[mode + 4] * C[mode])
    end

    half = one(T) / 2

    for offset in (0, 4)
        for (up, down) in ((offset + 1, offset + 2), (offset + 3, offset + 4))
            H_coef_mat .+= (model.U / 4) .* ((n[up] .- half .* I256) * (n[down] .- half .* I256))
            H_coef_mat .+= (-model.μ / 4) .* (n[up] + n[down])
        end

        sz1 = half .* (n[offset + 1] - n[offset + 2])
        sz2 = half .* (n[offset + 3] - n[offset + 4])
        sp1 = Cdag[offset + 1] * C[offset + 2]
        sm1 = Cdag[offset + 2] * C[offset + 1]
        sp2 = Cdag[offset + 3] * C[offset + 4]
        sm2 = Cdag[offset + 4] * C[offset + 3]
        H_coef_mat .+= (model.J / 4) .* (sz1 * sz2 + half .* (sp1 * sm2 + sm1 * sp2))
    end

    return reshape(H_coef_mat, (16, 16, 16, 16))
end

function nn_bond(model::BilayerHubbardModel)

    H_coef = nn_bond_Fock_basis(model)
    nn_bond_out1 = Grassmann(H_coef, (16, 16, 16, 16), (8, 8, 8, 8), (:out, :out, :in, :in))
    nn_bond_out2 = add_perm_sign(nn_bond_out1, (1, 2, 4, 3))
end

function gate(model::BilayerHubbardModel, dτ::Real)

    H_coef = nn_bond_Fock_basis(model)
    H_coef_mat = reshape(H_coef, (256, 256))
    G_coef_mat = exp(-dτ * H_coef_mat)
    G_coef = reshape(G_coef_mat, (16, 16, 16, 16))
    G = Grassmann(G_coef, (16, 16, 16, 16), (8, 8, 8, 8), (:out, :out, :in, :in))
    G_out = add_perm_sign(G, (1, 2, 4, 3))
end

############################# Relativistic fermions #############################

struct Nf1_Gross_Neveu_Wilson_model{T<:Real} <: AbstractModel
    m::T
    g2::T
    μ::T
    r::T
end

function Nf1_Gross_Neveu_Wilson_model(m::Real, g2::Real, μ::Real; r::Real=1.0)
    m, g2, μ, r = promote(m, g2, μ, r)
    return Nf1_Gross_Neveu_Wilson_model{typeof(m)}(m, g2, μ, r)
end

function generate_Ā_tensor()

    # Ā[i1, i2, j1p, j2p]
    Ā = zeros(ComplexF64, 2, 2, 2, 2)

    Ā[2, 2, 1, 1] = - 1 + im    
    Ā[2, 1, 2, 1] = - 1 - 1
    Ā[2, 1, 1, 2] = - 1 - im
    Ā[1, 2, 2, 1] = - im - 1
    Ā[1, 2, 1, 2] = - im - im
    Ā[1, 1, 2, 2] = 1 - im

    return Ā
end

function generate_A_tensor()

    # A[j1, j2, i1p, i2p]
    A = zeros(ComplexF64, 2, 2, 2, 2)

    A[2, 2, 1, 1] = 1 + im    
    A[2, 1, 2, 1] = 1 + 1
    A[2, 1, 1, 2] = 1 - im
    A[1, 2, 2, 1] = - im + 1
    A[1, 2, 1, 2] = - im - im
    A[1, 1, 2, 2] = -1 - im

    return A
end

function PartitionFunctionTensor(model::Nf1_Gross_Neveu_Wilson_model{T}) where {T}

    T_coef = zeros(ComplexF64, (4, 4, 4, 4))
    Ā = generate_Ā_tensor()
    A = generate_A_tensor()

    for i1 in 0:1, j1 in 0:1, i2 in 0:1, j2 in 0:1
        for i1p in 0:1, j1p in 0:1, i2p in 0:1, j2p in 0:1

            I = index_fuse(i1, j1)
            J = index_fuse(i2, j2)
            Ip = index_fuse(i1p, j1p)
            Jp = index_fuse(i2p, j2p)

            sign1 = i1 * (j1 + j2 + i1p + i2p) + i2 * (j2 + i1p + i2p) + j1p * (i1p + i2p) + j2p * i2p + i1p + i2p
            sign2 = i2 - j2 + i2p -j2p
            sign3 = i1 + j1 + i2 + j2 + i1p + j1p + i2p + j2p
            sign4 = i1 + i2 + j2 + i1p
            sign5 = i2 + j2 + i2p + j2p

            T_coef[I, J, Ip, Jp] = (-1)^sign1 * exp(0.5*model.μ*sign2) * (1/sqrt(2))^sign3 * (
                ((model.m+2*model.r)^2 + 2*model.g2) * isequal(i1 + i2 + j1p + j2p, 0) * isequal(j1 + j2 + i1p + i2p, 0) -
                (model.m+2*model.r) * isequal(i1 + i2 + j1p + j2p, 1) * isequal(j1 + j2 + i1p + i2p, 1) - 
                (-1)^sign4 * (+im)^sign5 * (model.m+2*model.r) * isequal(i1 + i2 + j1p + j2p, 1) * isequal(j1 + j2 + i1p + i2p, 1) - 
                Ā[i1+1, i2+1, j1p+1, j2p+1] * A[j1+1, j2+1, i1p+1, i2p+1])

        end
    end

    return T_coef
end

function index_fuse(ix::Int64, jx::Int64)

    Ix = 0

    if (ix, jx) == (0, 0)
        Ix = 1
    elseif (ix, jx) == (1, 1)
        Ix = 2
    elseif (ix, jx) == (0, 1)
        Ix = 3
    else
        Ix = 4
    end

    return Ix
end
