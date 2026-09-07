################################ AD rules for CTMRG measurements ################################

function _reduced_tensor_nomutation(peps::Square_GPEPS{Q}) where {Q}
    Lx, Ly = size(peps)
    return [reduced_tensor(peps.A[r, c]) for r in 1:Lx, c in 1:Ly]
end

function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(reduced_tensor),
    peps::Square_GPEPS)

    y = reduced_tensor(peps)
    _, raw_pullback = rrule_via_ad(config, _reduced_tensor_nomutation, peps)

    function reduced_tensor_pullback(Δy)
        Δy = unthunk(Δy)
        Δy isa AbstractZero && return NoTangent(), ZeroTangent()
        _, Δpeps = raw_pullback(Δy)
        return NoTangent(), Δpeps
    end

    return y, reduced_tensor_pullback
end

function _compute_exp_hbond_nomutation(
    Tbulk::Matrix{Grassmann{Q1, 4}},
    peps::Square_GPEPS{Q1},
    H_bond::Grassmann{Q2, 4},
    env::CTMRGEnv) where {Q1, Q2}

    size(Tbulk) == size(peps) || throw(DimensionMismatch("Tbulk and peps should have the same unit cell size"))
    size(Tbulk) == size(env) || throw(DimensionMismatch("Tbulk and CTMRG environment should have the same unit cell size"))

    Lx, Ly = size(Tbulk)
    left_op, right_op = _bond_operator_gsvd(H_bond)

    pairs = [
        begin
            c_p1 = Nmod(c + 1, Ly)
            Timp1 = reduced_tensor_alpha(peps.A[r, c], left_op)
            Timp2 = reduced_tensor_alpha(peps.A[r, c_p1], right_op)
            compute_exp_hbond_alpha(
                Tbulk[r, c], Tbulk[r, c_p1], Timp1, Timp2,
                env.El[r, c], env.Er[r, c_p1], env.Eu[r, c], env.Eu[r, c_p1],
                env.Ed[r, c], env.Ed[r, c_p1],
                env.Clu[r, c], env.Cru[r, c_p1], env.Cld[r, c], env.Crd[r, c_p1])
        end
        for r in 1:Lx, c in 1:Ly
    ]

    return first.(pairs), last.(pairs)
end

function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(compute_exp_hbond),
    Tbulk::Matrix{Grassmann{Q1, 4}},
    peps::Square_GPEPS{Q1},
    H_bond::Grassmann{Q2, 4},
    env::CTMRGEnv) where {Q1, Q2}

    y = compute_exp_hbond(Tbulk, peps, H_bond, env)
    _, raw_pullback = rrule_via_ad(
        config,
        (T, P) -> _compute_exp_hbond_nomutation(T, P, H_bond, env),
        Tbulk,
        peps)

    function compute_exp_hbond_pullback(Δy)
        Δy = unthunk(Δy)
        Δy isa AbstractZero && return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent(), NoTangent()
        _, ΔTbulk, Δpeps = raw_pullback(Δy)
        return NoTangent(), ΔTbulk, Δpeps, NoTangent(), NoTangent()
    end

    return y, compute_exp_hbond_pullback
end

function _compute_exp_vbond_nomutation(
    Tbulk::Matrix{Grassmann{Q1, 4}},
    peps::Square_GPEPS{Q1},
    H_bond::Grassmann{Q2, 4},
    env::CTMRGEnv) where {Q1, Q2}

    size(Tbulk) == size(peps) || throw(DimensionMismatch("Tbulk and peps should have the same unit cell size"))
    size(Tbulk) == size(env) || throw(DimensionMismatch("Tbulk and CTMRG environment should have the same unit cell size"))

    Lx, Ly = size(Tbulk)
    bottom_op, top_op = _bond_operator_gsvd(H_bond)

    pairs = [
        begin
            r_p1 = Nmod(r + 1, Lx)
            Talpha_top = reduced_tensor_alpha(peps.A[r, c], top_op)
            Talpha_bottom = reduced_tensor_alpha(peps.A[r_p1, c], bottom_op)
            compute_exp_vbond_alpha(
                Tbulk[r, c], Tbulk[r_p1, c], Talpha_top, Talpha_bottom,
                env.El[r, c], env.Er[r, c], env.El[r_p1, c], env.Er[r_p1, c],
                env.Eu[r, c], env.Ed[r_p1, c],
                env.Clu[r, c], env.Cru[r, c], env.Cld[r_p1, c], env.Crd[r_p1, c])
        end
        for r in 1:Lx, c in 1:Ly
    ]

    return first.(pairs), last.(pairs)
end

function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(compute_exp_vbond),
    Tbulk::Matrix{Grassmann{Q1, 4}},
    peps::Square_GPEPS{Q1},
    H_bond::Grassmann{Q2, 4},
    env::CTMRGEnv) where {Q1, Q2}

    y = compute_exp_vbond(Tbulk, peps, H_bond, env)
    _, raw_pullback = rrule_via_ad(
        config,
        (T, P) -> _compute_exp_vbond_nomutation(T, P, H_bond, env),
        Tbulk,
        peps)

    function compute_exp_vbond_pullback(Δy)
        Δy = unthunk(Δy)
        Δy isa AbstractZero && return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent(), NoTangent()
        _, ΔTbulk, Δpeps = raw_pullback(Δy)
        return NoTangent(), ΔTbulk, Δpeps, NoTangent(), NoTangent()
    end

    return y, compute_exp_vbond_pullback
end
