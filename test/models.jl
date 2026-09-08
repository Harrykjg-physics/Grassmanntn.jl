@testset "Model helpers" begin
    @test isdefined(GrassmannTensorNetworks, :InteractingSpinlessFermion)

    model = InteractingSpinlessFermion(2, 3.0, 5)
    @test model.t === 2.0
    @test model.μ === 3.0
    @test model.V === 5.0

    @test GrassmannTensorNetworks.n_site_Fock_basis(model) == [0.0 0.0; 0.0 1.0]

    H = GrassmannTensorNetworks.nn_bond_Fock_basis(model)
    expected = zeros(Float64, 2, 2, 2, 2)
    expected[2, 1, 1, 2] = -2.0
    expected[1, 2, 2, 1] = -2.0
    expected[2, 1, 2, 1] = -3.0 / 4
    expected[1, 2, 1, 2] = -3.0 / 4
    expected[2, 2, 2, 2] = 5.0 - 3.0 / 2
    @test H == expected

    @test size(n_site(model)) == (2, 2)
    @test size(nn_bond(model)) == (2, 2, 2, 2)
    @test size(gate(model, 0.1)) == (2, 2, 2, 2)
end

@testset "t-J model helpers" begin
    @test isdefined(GrassmannTensorNetworks, :TJModel)

    model = TJModel(2, 6.0, 4)
    @test model.t === 2.0
    @test model.J === 6.0
    @test model.μ === 4.0

    @test GrassmannTensorNetworks.n_site_Fock_basis(model) == [
        0.0 0.0 0.0
        0.0 1.0 0.0
        0.0 0.0 1.0
    ]

    H = GrassmannTensorNetworks.nn_bond_Fock_basis(model)
    expected = zeros(Float64, 3, 3, 3, 3)

    expected[2, 1, 1, 2] = -2.0
    expected[1, 2, 2, 1] = -2.0
    expected[3, 1, 1, 3] = -2.0
    expected[1, 3, 3, 1] = -2.0

    expected[2, 1, 2, 1] = -1.0
    expected[3, 1, 3, 1] = -1.0
    expected[1, 2, 1, 2] = -1.0
    expected[1, 3, 1, 3] = -1.0

    expected[2, 2, 2, 2] = -2.0
    expected[3, 3, 3, 3] = -2.0
    expected[2, 3, 2, 3] = -5.0
    expected[3, 2, 3, 2] = -5.0
    expected[3, 2, 2, 3] = 3.0
    expected[2, 3, 3, 2] = 3.0
    @test H == expected

    @test size(n_site(model)) == (3, 3)
    @test size(nn_bond(model)) == (3, 3, 3, 3)
    @test size(gate(model, 0.1)) == (3, 3, 3, 3)
end

const BILAYER_LOCAL_OCCUPATIONS = NTuple{4, Int}[
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
    (0, 1, 1, 1),
]

function _fermion_creation(occupations, mode)
    index_of = Dict(occ => i for (i, occ) in enumerate(occupations))
    C = zeros(Float64, length(occupations), length(occupations))
    for (col, occ) in enumerate(occupations)
        occ[mode] == 1 && continue
        sign_exponent = mode == 1 ? 0 : sum(occ[k] for k in 1:(mode - 1))
        new_occ = collect(occ)
        new_occ[mode] = 1
        C[index_of[Tuple(new_occ)], col] = isodd(sign_exponent) ? -1.0 : 1.0
    end
    return C
end

function _bilayer_local_number(mode)
    Cdag = _fermion_creation(BILAYER_LOCAL_OCCUPATIONS, mode)
    return Cdag * transpose(Cdag)
end

function _bilayer_global_occupations()
    return NTuple{8, Int}[Tuple(vcat(collect(left), collect(right))) for
                          right in BILAYER_LOCAL_OCCUPATIONS for left in BILAYER_LOCAL_OCCUPATIONS]
end

function _bilayer_reference_bond(model)
    occupations = _bilayer_global_occupations()
    Cdag = [_fermion_creation(occupations, mode) for mode in 1:8]
    C = transpose.(Cdag)
    n = [Cdag[mode] * C[mode] for mode in 1:8]
    I256 = Matrix{Float64}(I, 256, 256)

    H = zeros(Float64, 256, 256)
    for mode in 1:4
        H .+= -model.t .* (Cdag[mode] * C[mode + 4] + Cdag[mode + 4] * C[mode])
    end

    for offset in (0, 4)
        for (up, down) in ((offset + 1, offset + 2), (offset + 3, offset + 4))
            H .+= (model.U / 4) .* ((n[up] .- 0.5 .* I256) * (n[down] .- 0.5 .* I256))
            H .+= (-model.μ / 4) .* (n[up] + n[down])
        end
        sz1 = 0.5 .* (n[offset + 1] - n[offset + 2])
        sz2 = 0.5 .* (n[offset + 3] - n[offset + 4])
        sp1 = Cdag[offset + 1] * C[offset + 2]
        sm1 = Cdag[offset + 2] * C[offset + 1]
        sp2 = Cdag[offset + 3] * C[offset + 4]
        sm2 = Cdag[offset + 4] * C[offset + 3]
        H .+= (model.J / 4) .* (sz1 * sz2 + 0.5 .* (sp1 * sm2 + sm1 * sp2))
    end

    return reshape(H, 16, 16, 16, 16)
end

@testset "bilayer Hubbard model helpers" begin
    @test isdefined(GrassmannTensorNetworks, :BilayerHubbardModel)

    model = BilayerHubbardModel(2, 4.0, 1, 8)
    @test model.t === 2.0
    @test model.U === 4.0
    @test model.μ === 1.0
    @test model.J === 8.0

    integer_model = BilayerHubbardModel(2, 4, 1, 8)
    @test eltype(GrassmannTensorNetworks.nn_bond_Fock_basis(integer_model)) === Float64

    @test GrassmannTensorNetworks.nu1_site_Fock_basis(model) == _bilayer_local_number(1)
    @test GrassmannTensorNetworks.nd1_site_Fock_basis(model) == _bilayer_local_number(2)
    @test GrassmannTensorNetworks.nu2_site_Fock_basis(model) == _bilayer_local_number(3)
    @test GrassmannTensorNetworks.nd2_site_Fock_basis(model) == _bilayer_local_number(4)
    @test GrassmannTensorNetworks.n_site_Fock_basis(model) ==
          sum(_bilayer_local_number(mode) for mode in 1:4)

    H = GrassmannTensorNetworks.nn_bond_Fock_basis(model)
    expected = _bilayer_reference_bond(model)
    @test H == expected
    @test H[13, 1, 1, 13] == -2.0
    @test H[3, 1, 15, 13] == 2.0
    @test H[7, 1, 6, 1] == 1.0

    @test size(n_site(model)) == (16, 16)
    @test size(nn_bond(model)) == (16, 16, 16, 16)
    @test size(gate(model, 0.1)) == (16, 16, 16, 16)

    expected_gate_coef = reshape(exp(-0.1 * reshape(expected, 256, 256)), 16, 16, 16, 16)
    expected_gate = GrassmannTensorNetworks.add_perm_sign(
        Grassmann(expected_gate_coef, (16, 16, 16, 16), (8, 8, 8, 8), (:out, :out, :in, :in)),
        (1, 2, 4, 3),
    )
    @test convert(Array, gate(model, 0.1)) ≈ convert(Array, expected_gate)
end
