
# For partition function 
function compute_exp_site(
    Tbulk::Matrix{Grassmann{Q, 4}}, 
    env::CTMRGEnv) where {Q}

    Lx, Ly = size(Tbulk)
    Z = Matrix{Float64}(undef, Lx, Ly)

    for y = 1:Ly
        for x = 1:Lx
            Z[x, y] = compute_exp_site(Tbulk[x, y], 
            env.El[x, y], env.Er[x, y], env.Eu[x, y], env.Ed[x, y], 
            env.Clu[x, y], env.Cru[x, y], env.Cld[x, y], env.Crd[x, y])  
        end
    end

    return Z
end

function compute_exp_site(
    Tbulk::Grassmann{Q, 4}, 
    El::Grassmann{Q, 3}, 
    Er::Grassmann{Q, 3}, 
    Eu::Grassmann{Q, 3}, 
    Ed::Grassmann{Q, 3}, 
    Clu::Grassmann{Q, 2}, 
    Cru::Grassmann{Q, 2}, 
    Cld::Grassmann{Q, 2}, 
    Crd::Grassmann{Q, 2}) where {Q}

    # La[h1, v2, h2] = Clu[h1, v1] * El[v1, v2, h2]
    # Time ——— χ³D, Mem ———— χ²D
    La = contract(Clu, El, (2, 1); sign_function=global_sign)
    # L[h1, h2, h3] = La[h1, v2, h2] * Cld[h3, v2]
    # Time ———— χ³D, Mem ———— χ²D
    L = contract(La, Cld, (2, 2); sign_function=global_sign)

    # Ra[h1, v2, h2] = Cru[h1, v1] * Er[v1, v2, h2]
    # Time ———— χ³D, Mem ———— χ²D
    Ra = contract(Cru, Er, (2, 1); sign_function=global_sign)
    # R[h1, h2, h3] = Ra[h1, v2, h2] * Crd[h3, v2]
    # Time ———— χ³D, Mem ———— χ²D
    R = contract(Ra, Crd, (2, 2); sign_function=global_sign)

    # Ua[v1, h2, v2] = Clu[h1, v1] * Eu[h1, h2, v2]
    # Time ———— χ³D, Mem ———— χ²D
    Ua = contract(Clu, Eu, (1, 1); sign_function=global_sign)
    # U[v1, v2, v3] = Ua[v1, h2, v2] * Cru[h2, v3]
    # Time ———— χ³D, Mem ———— χ²D
    U = contract(Ua, Cru, (2, 1); sign_function=global_sign)

    # Da[v1, h2, v2] = Cld[h1, v1] * Ed[h1, h2, v2]
    # Time ———— χ³D, Mem ———— χ²D
    Da = contract(Cld, Ed, (1, 1); sign_function=global_sign)
    # D[v1, v2, v3] = Da[v1, h2, v2] * Crd[h2, v3]
    D = contract(Da, Crd, (2, 1); sign_function=global_sign)
    # Time ———— χ³D, Mem ———— χ²D

    # C4a[v1, v2] = Clu[h1, v1] * Cru[h1, v2] 
    # Time ———— χ³, Mem ———— χ²
    C4a = contract(Clu, Cru, (1, 1); sign_function=global_sign)
    # C4b[h2, v1] <-- C4b[v1, h2] = C4a[v1, v2] * Crd[h2, v2]
    # Time ———— χ³, Mem ———— χ²
    C4b = contract(C4a, Crd, (2, 2); perm=(2, 1), sign_function=global_sign)
    # C4 = C4b[h2, v1] * Cld[h2, v1]
    # Time ———— χ², Mem ———— 1
    C4 = contract(C4b, Cld, ((1, 2), (1, 2)); sign_function=global_sign)

    # normLR = L[h1, h2, h3] * R[h1, h2, h3]
    # Time ———— χ²D, Mem ———— 1
    normLR = contract(L, R, ((1, 2, 3), (1, 2, 3)); sign_function=global_sign)

    # normUD = U[v1, v2, v3] * D[v1, v2, v3]
    # Time ———— χ²D, Mem ———— 1
    normUD = contract(U, D, ((1, 2, 3), (1, 2, 3)); sign_function=global_sign)

    # normT1[h2, h3, h4, v1] = L[h1, h2, h3] * Eu[h1, h4, v1]
    # Time ———— χ³D², Mem ———— χ²D²
    normT1 = contract(L, Eu, (1, 1); sign_function=global_sign)
    # normT2[h3, h4, h5, v2] = normT1[h2, h3, h4, v1] * Tbulk[h2, h5, v1, v2]
    # Time ———— χ²D⁴ , Mem ———— χ²D²
    normT2 = contract(normT1, Tbulk, ((1, 4), (1, 3)); sign_function=global_sign)
    # normT3[h4, h5, h6] = normT2[h3, h4, h5, v2] * Ed[h3, h6, v2]
    # Time ———— χ³D²  , Mem ———— χ²D
    normT3 = contract(normT2, Ed, ((1, 4), (1, 3)); sign_function=global_sign)
    # normT = normT3[h4, h5, h6] * R[h4, h5, h6]
    # Time ———— χ²D  , Mem ———— 1
    normT = contract(normT3, R, ((1, 2, 3), (1, 2, 3)); sign_function=global_sign)

    Z = abs(scalar(normT) * scalar(C4) / (scalar(normLR) * scalar(normUD)))
end

function compute_exp_site(
    Tbulk::Matrix{Grassmann{Q1, 4}}, 
    Timp::Grassmann{Q2, 4}, 
    env::CTMRGEnv) where {Q1, Q2}
    
    Lx, Ly = size(Tbulk)

    Q = promote_type(Q1, Q2)

    Z = Matrix{Q}(undef, Lx, Ly)
    O = Matrix{Q}(undef, Lx, Ly)

    for y = 1:Ly
        for x = 1:Lx
            Z[x, y], O[x, y] = compute_exp_site(
                Tbulk[x, y], Timp, 
                env.El[x, y], env.Er[x, y], env.Eu[x, y], env.Ed[x, y], 
                env.Clu[x, y], env.Cru[x, y], env.Cld[x, y], env.Crd[x, y]
                )  
        end
    end

    return Z, O
end

function compute_exp_site(
    Tbulk::Matrix{Grassmann{Q1, 4}}, 
    Timp::Matrix{Grassmann{Q2, 4}}, 
    env::CTMRGEnv) where {Q1, Q2}
    
    size(Tbulk) == size(Timp) || throw(DimensionMismatch("Tbulk and Timp should have the same unit cell size"))

    Q = promote_type(Q1, Q2)

    Lx, Ly = size(Tbulk)
    Z = Matrix{Q}(undef, Lx, Ly)
    O = Matrix{Q}(undef, Lx, Ly)

    for y = 1:Ly
        for x = 1:Lx
            Z[x, y], O[x, y] = compute_exp_site(
                Tbulk[x, y], Timp[x, y], 
                env.El[x, y], env.Er[x, y], env.Eu[x, y], env.Ed[x, y], 
                env.Clu[x, y], env.Cru[x, y], env.Cld[x, y], env.Crd[x, y]
                )  
        end
    end

    return Z, O
end

function compute_exp_site(
    Tbulk::Grassmann{Q1, 4}, 
    Timp::Grassmann{Q2, 4}, 
    El::Grassmann{Q1, 3}, 
    Er::Grassmann{Q1, 3}, 
    Eu::Grassmann{Q1, 3}, 
    Ed::Grassmann{Q1, 3}, 
    Clu::Grassmann{Q1, 2}, 
    Cru::Grassmann{Q1, 2}, 
    Cld::Grassmann{Q1, 2}, 
    Crd::Grassmann{Q1, 2}) where {Q1, Q2}

    # La[h1, v2, h2] = Clu[h1, v1] * El[v1, v2, h2]
    # Time ——— χ³D, Mem ———— χ²D
    La = contract(Clu, El, (2, 1); sign_function=global_sign)
    # L[h1, h2, h3] = La[h1, v2, h2] * Cld[h3, v2]
    # Time ———— χ³D, Mem ———— χ²D
    L = contract(La, Cld, (2, 2); sign_function=global_sign)

    # Ra[h1, v2, h2] = Cru[h1, v1] * Er[v1, v2, h2]
    # Time ———— χ³D, Mem ———— χ²D
    Ra = contract(Cru, Er, (2, 1); sign_function=global_sign)
    # R[h1, h2, h3] = Ra[h1, v2, h2] * Crd[h3, v2]
    # Time ———— χ³D, Mem ———— χ²D
    R = contract(Ra, Crd, (2, 2); sign_function=global_sign)

    # normT1[h2, h3, h4, v1] = L[h1, h2, h3] * Eu[h1, h4, v1]
    # Time ———— χ³D², Mem ———— χ²D²
    normT1 = contract(L, Eu, (1, 1); sign_function=global_sign)
    # normT2[h3, h4, h5, v2] = normT1[h2, h3, h4, v1] * Tbulk[h2, h5, v1, v2]
    # Time ———— χ²D⁴ , Mem ———— χ²D²
    normT2 = contract(normT1, Tbulk, ((1, 4), (1, 3)); sign_function=global_sign)
    # normT3[h4, h5, h6] = normT2[h3, h4, h5, v2] * Ed[h3, h6, v2]
    # Time ———— χ³D²  , Mem ———— χ²D
    normT3 = contract(normT2, Ed, ((1, 4), (1, 3)); sign_function=global_sign)
    # normT = normT3[h4, h5, h6] * R[h4, h5, h6]
    # Time ———— χ²D  , Mem ———— 1
    normT = contract(normT3, R, ((1, 2, 3), (1, 2, 3)); sign_function=global_sign)

    # normTimp1[h2, h3, h4, v1] = L[h1, h2, h3] * Eu[h1, h4, v1]
    # Time ———— χ³D², Mem ———— χ²D²
    normTimp1 = contract(L, Eu, (1, 1); sign_function=global_sign)
    # normTimp2[h3, h4, h5, v2] = normTimp1[h2, h3, h4, v1] * Tbulk[h2, h5, v1, v2]
    # Time ———— χ²D⁴ , Mem ———— χ²D²
    normTimp2 = contract(normTimp1, Timp, ((1, 4), (1, 3)); sign_function=global_sign)
    # normTimp3[h4, h5, h6] = normTimp2[h3, h4, h5, v2] * Ed[h3, h6, v2]
    # Time ———— χ³D²  , Mem ———— χ²D
    normTimp3 = contract(normTimp2, Ed, ((1, 4), (1, 3)); sign_function=global_sign)
    # normTimp = normTimp3[h4, h5, h6] * R[h4, h5, h6]
    # Time ———— χ²D  , Mem ———— 1
    normTimp = contract(normTimp3, R, ((1, 2, 3), (1, 2, 3)); sign_function=global_sign)

    Z = scalar(normT)
    O = scalar(normTimp)

    return Z, O/Z
end

"""
   Clu1 ←← (h1) ←← Eu1 ←← (h2) ←← Eu2 ←← (h3) ←← Cru2                  
    ↓               ↓              ↓              ↑
    ↓               ↓              ↓              ↑
   (v1)            (v2)           (v3)           (v4)
    ↓               ↓              ↓              ↑
    ↓               ↓              ↓              ↑
   El1 ←← (h4) ←← #################### ←← (h6) ←← Er2
    ↓               ↓              ↓              ↑
    ↓               ↓              ↓              ↑
   (v5)            (v6)           (v7)           (v8)
    ↓               ↓              ↓              ↑
    ↓               ↓              ↓              ↑
   Cld1 →→ (h7) →→ Ed1 →→ (h8) →→  Ed2 →→ (h9) → Crd2

——————————————————————————————————————————————————————————————

  Clu1 ←← (h1) ←← Eu1 ←← (h2) ←← Eu2 ←← (h3) ←← Cru2                  
    ↓               ↓              ↓              ↑
    ↓               ↓              ↓              ↑
   (v1)            (v2)           (v3)           (v4)
    ↓               ↓              ↓              ↑
    ↓               ↓              ↓              ↑
   El1 ←← (h4) ← Tbulk1 ← (h5) ← Tbulk2 ← (h6) ← Er2
    ↓               ↓              ↓              ↑
    ↓               ↓              ↓              ↑
   (v5)            (v6)           (v7)           (v8)
    ↓               ↓              ↓              ↑
    ↓               ↓              ↓              ↑
   Cld1 →→ (h7) →→ Ed1 →→ (h8) →→  Ed2 →→ (h9) → Crd2

function contra1(Tbulk1, Tbulk2, El1, Er2, Eu1, Eu2, Ed1, Ed2,  Clu1, Cru2, Cld1, Crd2) 
    @tensoropt_verbose out[] := Clu1[h1, v1] * El1[v1, v5, h4] * Cld1[h7, v5] * 
    Eu1[h1, h2, v2] * Tbulk1[h4, h5, v2, v6] * Ed1[h7, h8, v6] * Eu2[h2, h3, v3] * 
    Tbulk2[h5, h6, v3, v7] * Ed2[h8, h9, v7] * Cru2[h3, v4] * Er2[v4, v8, h6] * Crd2[h9, v8]
end

Solution found with cost 2*χ^6 + 4*χ^5 + 4*χ^4 + 0*χ^3 + 1*χ^2 + 0*χ + 0
                         tree Any[3, 
                         Any[
                         Any[1, 2], 
                         Any[6, 
                         Any[5, 
                         Any[4, 
                         Any[
                         Any[10, 7], 
                         Any[8, Any[11, Any[9, 12]]]
                         ]]]]]]

function contra2(Tbulk1, Tbulk2, El1, Er2, Eu1, Eu2, Ed1, Ed2,  Clu1, Cru2, Cld1, Crd2) 
    @tensoropt_verbose out[] := Clu1[h1, v1] * El1[v1, v5, h4] * Cld1[h7, v5] * 
    Eu1[h1, h2, v2] * Timp[h4, h6, v2, v3, v6, v7] * Ed1[h7, h8, v6] * Eu2[h2, h3, v3] * 
    Ed2[h8, h9, v7] * Cru2[h3, v4] * Er2[v4, v8, h6] * Crd2[h9, v8]
end

Solution found with cost 1*χ^8 + 0*χ^7 + 2*χ^6 + 2*χ^5 + 4*χ^4 + 0*χ^3 + 1*χ^2 + 0*χ + 0
                         tree Any[3, 
                         Any[
                         Any[1, 2], 
                         Any[
                         Any[6, Any[11, 8]], 
                         Any[5, 
                         Any[Any[4, 7], Any[9, 10]]
                         ]]]]
"""

function compute_exp_hbond(
    Tbulk::Matrix{Grassmann{Q1, 4}}, 
    Timp::Matrix{Grassmann{Q2, 6}}, 
    env::CTMRGEnv) where {Q1, Q2}
    
    Lx, Ly = size(Tbulk)

    Q = promote_type(Q1, Q2)

    Z = Matrix{Q}(undef, Lx, Ly)
    O = Matrix{Q}(undef, Lx, Ly)

    for c in 1:Ly, r in 1:Lx

        c_p1 = Nmod(c + 1, Ly)

        Z[r, c], O[r, c] = compute_exp_hbond(
            Tbulk[r, c], Tbulk[r, c_p1], Timp[r, c],
            env.El[r, c], env.Er[r, c_p1], env.Eu[r, c], env.Eu[r, c_p1], 
            env.Ed[r, c], env.Ed[r, c_p1], 
            env.Clu[r, c], env.Cru[r, c_p1], env.Cld[r, c], env.Crd[r, c_p1]) 
    end

    return Z, O
end

function _compute_exp_hbond_denominator(
    Tbulk1::Grassmann{Q1, 4},
    Tbulk2::Grassmann{Q1, 4},
    El1::Grassmann{Q1, 3},
    Er2::Grassmann{Q1, 3},
    Eu1::Grassmann{Q1, 3},
    Eu2::Grassmann{Q1, 3},
    Ed1::Grassmann{Q1, 3},
    Ed2::Grassmann{Q1, 3},
    Clu1::Grassmann{Q1, 2},
    Cru2::Grassmann{Q1, 2},
    Cld1::Grassmann{Q1, 2},
    Crd2::Grassmann{Q1, 2}) where {Q1}

    # left_top[h1, v5, h4] = Clu1[h1, v1] * El1[v1, v5, h4]
    left_top = contract(Clu1, El1, (2, 1); sign_function=global_sign)
    # state[h1, h4, h7] = left_top[h1, v5, h4] * Cld1[h7, v5]
    state = contract(left_top, Cld1, (2, 2); sign_function=global_sign)
    state, _ = left_move(state, Ed1, Tbulk1, Eu1)
    state, _ = left_move(state, Ed2, Tbulk2, Eu2)
    # right_top[h3, v8, h6] = Cru2[h3, v4] * Er2[v4, v8, h6]
    right_top = contract(Cru2, Er2, (2, 1); sign_function=global_sign)
    # right[h3, h6, h9] = right_top[h3, v8, h6] * Crd2[h9, v8]
    right = contract(right_top, Crd2, (2, 2); sign_function=global_sign)
    # out = state[h3, h6, h9] * right[h3, h6, h9]
    out = contract(state, right, ((1, 2, 3), (1, 2, 3)); sign_function=global_sign)

    return out
end

function _bond_operator_gsvd(operator::Grassmann{T, 4}) where {T}

    row_size = size(operator)[1] * size(operator)[3]
    col_size = size(operator)[2] * size(operator)[4]
    Dcut = min(row_size, col_size)

    # operator_perm[pul, pdl, pur, pdr] <-- operator[pul, pur, pdl, pdr]
    operator_perm = permutedims(operator, (1, 3, 2, 4); sign_function=global_sign)
    # U[(pul, pdl), b], S[b, b], Vdag[b, (pur, pdr)] <-- operator_perm[pul, pdl, pur, pdr]
    # V[(pur, pdr), b] <-- Vdag[b, (pur, pdr)]
    U, S, V, _ = gsvd(operator_perm, (1, 2), (3, 4), Dcut; trunc=false, sign_function=global_sign)
    sqrtS = sqrt(S)

    # L[pul, pdl, a] = U[pul, pdl, b] * sqrt(S)[b, a]
    L = contract(U, sqrtS, (3, 1); sign_function=global_sign)
    # R0[a, pur, pdr] = sqrt(S)[a, b] * conj(V)[(pur, pdr), b]
    R0 = contract(sqrtS, V, (2, 3); cj=(false, true), sign_function=global_sign)
    # R[pur, pdr, a] <-- R0[a, pur, pdr]
    R = permutedims(R0, (2, 3, 1); sign_function=global_sign)

    return L, R
end

function compute_exp_hbond_alpha(
    Tbulk1::Grassmann{Q1, 4},
    Tbulk2::Grassmann{Q1, 4},
    Talpha1::Grassmann{Q2, 5},
    Talpha2::Grassmann{Q2, 5},
    El1::Grassmann{Q1, 3},
    Er2::Grassmann{Q1, 3},
    Eu1::Grassmann{Q1, 3},
    Eu2::Grassmann{Q1, 3},
    Ed1::Grassmann{Q1, 3},
    Ed2::Grassmann{Q1, 3},
    Clu1::Grassmann{Q1, 2},
    Cru2::Grassmann{Q1, 2},
    Cld1::Grassmann{Q1, 2},
    Crd2::Grassmann{Q1, 2}) where {Q1, Q2}

    Den = _compute_exp_hbond_denominator(Tbulk1, Tbulk2, 
    El1, Er2, Eu1, Eu2, Ed1, Ed2, Clu1, Cru2, Cld1, Crd2)

    left_top = contract(Clu1, El1, (2, 1); sign_function=global_sign)
    state = contract(left_top, Cld1, (2, 2); sign_function=global_sign)
    state, _ = _left_move_keep_open(state, Ed1, Talpha1, Eu1)
    state, _ = _left_move_keep_open(state, Ed2, Talpha2, Eu2)
    # trace out the alpha indices
    state = trace(state, (4, 5); sign_function=global_sign)

    right_top = contract(Cru2, Er2, (2, 1); sign_function=global_sign)
    right = contract(right_top, Crd2, (2, 2); sign_function=global_sign)
    Num = contract(state, right, ((1, 2, 3), (1, 2, 3)); sign_function=global_sign)

    return scalar(Den), scalar(Num)/scalar(Den)
end

function compute_exp_hbond(
    Tbulk::Matrix{Grassmann{Q1, 4}},
    peps::Square_GPEPS{Q1},
    H_bond::Grassmann{Q2, 4},
    env::CTMRGEnv) where {Q1, Q2}

    size(Tbulk) == size(peps) || throw(DimensionMismatch("Tbulk and peps should have the same unit cell size"))
    size(Tbulk) == size(env) || throw(DimensionMismatch("Tbulk and CTMRG environment should have the same unit cell size"))

    Lx, Ly = size(Tbulk)
    Q = promote_type(Q1, Q2)

    Z = Matrix{Q}(undef, Lx, Ly)
    O = Matrix{Q}(undef, Lx, Ly)
    left_op, right_op = _bond_operator_gsvd(H_bond)

    for c in 1:Ly, r in 1:Lx
        c_p1 = Nmod(c + 1, Ly)
        Timp1 = reduced_tensor_alpha(peps.A[r, c], left_op)
        Timp2 = reduced_tensor_alpha(peps.A[r, c_p1], right_op)

        Z[r, c], O[r, c] = compute_exp_hbond_alpha(
            Tbulk[r, c], Tbulk[r, c_p1], Timp1, Timp2, 
            env.El[r, c], env.Er[r, c_p1], env.Eu[r, c], env.Eu[r, c_p1], 
            env.Ed[r, c], env.Ed[r, c_p1], 
            env.Clu[r, c], env.Cru[r, c_p1], env.Cld[r, c], env.Crd[r, c_p1])

        # Release the memory of Timp1 and Timp2 
        Timp1 = nothing
        Timp2 = nothing
    end

    return Z, O
end

function compute_exp_hbond(
    Tbulk1::Grassmann{Q1, 4}, 
    Tbulk2::Grassmann{Q1, 4}, 
    Timp::Grassmann{Q2, 6}, 
    El1::Grassmann{Q1, 3}, 
    Er2::Grassmann{Q1, 3}, 
    Eu1::Grassmann{Q1, 3}, 
    Eu2::Grassmann{Q1, 3},
    Ed1::Grassmann{Q1, 3}, 
    Ed2::Grassmann{Q1, 3},  
    Clu1::Grassmann{Q1, 2}, 
    Cru2::Grassmann{Q1, 2}, 
    Cld1::Grassmann{Q1, 2}, 
    Crd2::Grassmann{Q1, 2}) where {Q1, Q2}

    # out1[h2, v3, v4] <-- out1[v4, h2, v3] = Cru2[h3, v4] * Eu2[h2, h3, v3]
    out1 = contract(Cru2, Eu2, (1, 2); perm=(2, 3, 1), sign_function=global_sign)
    # out2[h8, v7, v8] = Ed2[h8, h9, v7] * Crd2[h9, v8]
    out2 = contract(Ed2, Crd2, (2, 1); sign_function=global_sign)
    # out3[v4, h6, h8, v7] = Er2[v4, v8, h6] * out2[h8, v7, v8]
    out3 = contract(Er2, out2, (2, 3); sign_function=global_sign)
    # out4[h5, v3, v4, h8] = Tbulk2[h5, h6, v3, v7] * out3[v4, h6, h8, v7]
    out4 = contract(Tbulk2, out3, ((2, 4), (2, 4)); sign_function=global_sign)
    # out5[h2, h5, h8] = out1[h2, v3, v4] * out4[h5, v3, v4, h8]
    out5 = contract(out1, out4, ((2, 3), (2, 3)); sign_function=global_sign)
    # out6[h1, h5, v2, h8] <-- out6[h1, v2, h5, h8] = Eu1[h1, h2, v2] * out5[h2, h5, h8]
    out6 = contract(Eu1, out5, (2, 1); perm=(1, 3, 2, 4), sign_function=global_sign)
    # out7[h1, h8, h4, v6] = out6[h1, h5, v2, h8] * Tbulk1[h4, h5, v2, v6]
    out7 = contract(out6, Tbulk1, ((2, 3), (2, 3)); sign_function=global_sign)
    # out8[h7, h1, h4] = Ed1[h7, h8, v6] * out7[h1, h8, h4, v6]
    out8 = contract(Ed1, out7, ((2, 3), (2, 4)); sign_function=global_sign)
    # out9[h1, v5, h4] = Clu1[h1, v1] * El1[v1, v5, h4]
    out9 = contract(Clu1, El1, (2, 1); sign_function=global_sign)
    # out10[h7, v5] <-- out10[v5, h7] = out9[h1, v5, h4] * out8[h7, h1, h4]
    out10 = contract(out9, out8, ((1, 3), (2, 3)); perm=(2, 1), sign_function=global_sign)
    # Den = Cld1[h7, v5] * out10[h7, v5]
    Den = contract(Cld1, out10, ((1, 2), (1, 2)); sign_function=global_sign)

    # out11[h1, v2, h3, v3] = Eu1[h1, h2, v2] * Eu2[h2, h3, v3]
    out11 = contract(Eu1, Eu2, (2, 1); sign_function=global_sign)
    # out12[h3, v8, h6] = Cru2[h3, v4] * Er2[v4, v8, h6]
    out12 = contract(Cru2, Er2, (2, 1); sign_function=global_sign)
    # out13[h1, h6, v2, v3, v8] <-- out13[h1, v2, v3, v8, h6] = out11[h1, v2, h3, v3] * out12[h3, v8, h6]
    out13 = contract(out11, out12, (3, 1); perm=(1, 5, 2, 3, 4), sign_function=global_sign)
    # out14[h4, v6, v7, h1, v8] = Timp[h4, h6, v2, v3, v6, v7] * out13[h1, h6, v2, v3, v8]
    out14 = contract(Timp, out13, ((2, 3, 4), (2, 3, 4)); sign_function=global_sign)
    # out15[h7, v6, v7, v8] = Ed1[h7, h8, v6] * out2[h8, v7, v8]
    out15 = contract(Ed1, out2, (2, 1); sign_function=global_sign)
    # out16[h1, h4, h7] <-- out16[h7, h4, h1] = out15[h7, v6, v7, v8] * out14[h4, v6, v7, h1, v8]
    out16 = contract(out15, out14, ((2, 3, 4), (2, 3, 5)); perm=(3, 2, 1), sign_function=global_sign)
    # out17[h7, v5] = out16[h1, h4, h7] * out9[h1, v5, h4]
    out17 = contract(out16, out9, ((1, 2), (1, 3)); sign_function=global_sign)
    # Num = out17[h7, v5] * Cld1[h7, v5]
    Num = contract(out17, Cld1, ((1, 2), (1, 2)); sign_function=global_sign)

    return scalar(Den), scalar(Num)/scalar(Den)
end

"""
Clu1 ←← (h1) ←← Eu1 ←← (h2) ←← Cru1
 ↓               ↓               ↑           
 ↓               ↓               ↑
(v1)            (v4)            (v6)            
 ↓               ↓               ↑              
 ↓               ↓               ↑              Crd2 ←← (v8) ←← Er2 ←← (v7) ←← Er1 ←← (v6) ←← Cru1 
El1  ←← (h3) ←← #### ←← (h4) ←← Er1              ↓               ↓              ↓              ↑
 ↓              ####             ↑               ↓               ↓              ↓              ↑
 ↓              ####             ↑              (h8)            (h6)           (h4)           (h2)
(v2)            ####            (v7)             ↓               ↓              ↓              ↑
 ↓              ####             ↑               ↓               ↓              ↓              ↑
 ↓              ####             ↑      ===>    Ed2 ←← (v5) ←← #################### ←← (v4) ←← Eu1
El2  ←← (h5) ←← #### ←← (h6) ←← Er2              ↓               ↓              ↓              ↑
 ↓               ↓               ↑               ↓               ↓              ↓              ↑
 ↓               ↓               ↑              (h7)            (h5)           (h3)           (h1)
(v3)            (v5)            (v8)             ↓               ↓              ↓              ↑ 
 ↓               ↓               ↑               ↓               ↓              ↓              ↑
 ↓               ↓               ↑              Cld2 →→ (v3) →→ El2 →→ (v2) →→  El1 →→ (v1) → Clu1
Cld2 →→ (h7) →→ Ed2 →→ (h8) →→ Crd2

Clu1 ←← (h1) ←← Eu1 ←← (h2) ←← Cru1
 ↓               ↓               ↑           
 ↓               ↓               ↑
(v1)            (v4)            (v6)            Crd2 ←← (v8) ←← Er2 ←← (v7) ←← Er1 ←← (v6) ←← Cru1 
 ↓               ↓               ↑               ↓               ↓              ↓              ↑ 
 ↓               ↓               ↑               ↓               ↓              ↓              ↑ 
El1  ←← (h3) ←← Tb1 ←← (h4) ←← Er1              (h8)            (h6)           (h4)           (h2)
 ↓               ↓               ↑               ↓               ↓              ↓              ↑
 ↓               ↓               ↑               ↓               ↓              ↓              ↑
(v2)           (v8)             (v7)   ===>     Ed2 ←← (v5) ←←← Tb2 ←← (v8) ←← Tb1 ←← (v4) ←← Eu1
 ↓               ↓               ↑               ↓               ↓              ↓              ↑
 ↓               ↓               ↑               ↓               ↓              ↓              ↑
El2  ←← (h5) ←← Tb2 ←← (h6) ←← Er2              (h7)            (h5)           (h3)           (h1) 
 ↓               ↓               ↑               ↓               ↓              ↓              ↑ 
 ↓               ↓               ↑               ↓               ↓              ↓              ↑
(v3)            (v5)            (v8)            Cld2 →→ (v3) →→ El2 →→ (v2) →→  El1 →→ (v1) → Clu1
 ↓               ↓               ↑ 
 ↓               ↓               ↑
Cld2 →→ (h7) →→ Ed2 →→ (h8) →→ Crd2

"""

function compute_exp_vbond(
    Tbulk::Matrix{Grassmann{Q1, 4}}, 
    Timp::Matrix{Grassmann{Q2, 6}}, 
    env::CTMRGEnv) where {Q1, Q2}
    
    Lx, Ly = size(Tbulk)

    Q = promote_type(Q1, Q2)

    Z = Matrix{Q}(undef, Lx, Ly)
    O = Matrix{Q}(undef, Lx, Ly)

    for c in 1:Ly, r in 1:Lx

        r_p1 = Nmod(r + 1, Lx)

        Z[r, c], O[r, c] = compute_exp_vbond(
            Tbulk[r_p1, c], Tbulk[r, c], Timp[r, c],
            env.Ed[r_p1, c], env.Eu[r, c], env.Er[r_p1, c], env.Er[r, c], 
            env.El[r_p1, c], env.El[r, c], 
            env.Crd[r_p1, c], env.Cru[r, c], env.Cld[r_p1, c], env.Clu[r, c]) 
    end

    return Z, O
end

function _compute_exp_vbond_denominator(
    Ttop::Grassmann{Q1, 4},
    Tbottom::Grassmann{Q1, 4},
    El_top::Grassmann{Q1, 3},
    Er_top::Grassmann{Q1, 3},
    El_bottom::Grassmann{Q1, 3},
    Er_bottom::Grassmann{Q1, 3},
    Eu_top::Grassmann{Q1, 3},
    Ed_bottom::Grassmann{Q1, 3},
    Clu_top::Grassmann{Q1, 2},
    Cru_top::Grassmann{Q1, 2},
    Cld_bottom::Grassmann{Q1, 2},
    Crd_bottom::Grassmann{Q1, 2}) where {Q1}

    upper_left = contract(Clu_top, Eu_top, (1, 1); sign_function=global_sign)
    state = contract(upper_left, Cru_top, (2, 1); sign_function=global_sign)
    state, _ = up_move(state, El_top, Ttop, Er_top)
    state, _ = up_move(state, El_bottom, Tbottom, Er_bottom)

    lower_left = contract(Cld_bottom, Ed_bottom, (1, 1); sign_function=global_sign)
    lower = contract(lower_left, Crd_bottom, (2, 1); sign_function=global_sign)

    out = contract(state, lower, ((1, 2, 3), (1, 2, 3)); sign_function=global_sign)

    return out
end

function compute_exp_vbond_alpha(
    Ttop::Grassmann{Q1, 4},
    Tbottom::Grassmann{Q1, 4},
    Talpha_top::Grassmann{Q2, 5},
    Talpha_bottom::Grassmann{Q2, 5},
    El_top::Grassmann{Q1, 3},
    Er_top::Grassmann{Q1, 3},
    El_bottom::Grassmann{Q1, 3},
    Er_bottom::Grassmann{Q1, 3},
    Eu_top::Grassmann{Q1, 3},
    Ed_bottom::Grassmann{Q1, 3},
    Clu_top::Grassmann{Q1, 2},
    Cru_top::Grassmann{Q1, 2},
    Cld_bottom::Grassmann{Q1, 2},
    Crd_bottom::Grassmann{Q1, 2}) where {Q1, Q2}

    Den = _compute_exp_vbond_denominator(
        Ttop, Tbottom, 
        El_top, Er_top, El_bottom, Er_bottom, 
        Eu_top, Ed_bottom, 
        Clu_top, Cru_top, Cld_bottom, Crd_bottom)

    upper_left = contract(Clu_top, Eu_top, (1, 1); sign_function=global_sign)
    state = contract(upper_left, Cru_top, (2, 1); sign_function=global_sign)
    state, _ = _up_move_keep_open(state, El_top, Talpha_top, Er_top)
    state, _ = _up_move_keep_open(state, El_bottom, Talpha_bottom, Er_bottom)
    state = trace(state, (4, 5); sign_function=global_sign)

    lower_left = contract(Cld_bottom, Ed_bottom, (1, 1); sign_function=global_sign)
    lower = contract(lower_left, Crd_bottom, (2, 1); sign_function=global_sign)
    Num = contract(state, lower, ((1, 2, 3), (1, 2, 3)); sign_function=global_sign)

    return scalar(Den), scalar(Num)/scalar(Den)
end

function compute_exp_vbond(
    Tbulk::Matrix{Grassmann{Q1, 4}}, 
    peps::Square_GPEPS{Q1}, 
    H_bond::Grassmann{Q2, 4}, 
    env::CTMRGEnv) where {Q1, Q2}

    size(Tbulk) == size(peps) || throw(DimensionMismatch("Tbulk and peps should have the same unit cell size"))
    size(Tbulk) == size(env) || throw(DimensionMismatch("Tbulk and CTMRG environment should have the same unit cell size"))

    Lx, Ly = size(Tbulk)
    Q = promote_type(Q1, Q2)

    Z = Matrix{Q}(undef, Lx, Ly)
    O = Matrix{Q}(undef, Lx, Ly)
    bottom_op, top_op = _bond_operator_gsvd(H_bond)

    for c in 1:Ly, r in 1:Lx

        r_p1 = Nmod(r + 1, Lx)
        Talpha_top = reduced_tensor_alpha(peps.A[r, c], top_op)
        Talpha_bottom = reduced_tensor_alpha(peps.A[r_p1, c], bottom_op)

        Z[r, c], O[r, c] = compute_exp_vbond_alpha(
            Tbulk[r, c], Tbulk[r_p1, c], Talpha_top, Talpha_bottom, 
            env.El[r, c], env.Er[r, c], env.El[r_p1, c], env.Er[r_p1, c], 
            env.Eu[r, c], env.Ed[r_p1, c], 
            env.Clu[r, c], env.Cru[r, c], env.Cld[r_p1, c], env.Crd[r_p1, c])

        # Release the memory of Talpha_top and Talpha_bottom
        Talpha_top = nothing
        Talpha_bottom = nothing
    end

    return Z, O
end

function compute_exp_vbond(
    Tb2::Grassmann{Q1, 4}, Tb1::Grassmann{Q1, 4}, Timp::Grassmann{Q2, 6}, 
    Ed2::Grassmann{Q1, 3}, Eu1::Grassmann{Q1, 3}, 
    Er2::Grassmann{Q1, 3}, Er1::Grassmann{Q1, 3},
    El2::Grassmann{Q1, 3}, El1::Grassmann{Q1, 3},  
    Crd2::Grassmann{Q1, 2}, Cru1::Grassmann{Q1, 2}, 
    Cld2::Grassmann{Q1, 2}, Clu1::Grassmann{Q1, 2}) where {Q1, Q2}

    # Tb2[v5, v8, h6, h5] <-- Tb2[h5, h6, v8, v5]
    Tb2 = permutedims(Tb2, (4, 3, 2, 1); sign_function=global_sign)
    # Tb1[v8, v4, h4, h3] <-- Tb1[h3, h4, v4, v8]
    Tb1 = permutedims(Tb1, (4, 3, 2, 1); sign_function=global_sign)
    # Ed2[h8, h7, v5] <-- Ed2[h7, h8, v5]
    Ed2 = permutedims(Ed2, (2, 1, 3); sign_function=global_sign)
    # Eu1[h2, h1, v4] <-- Eu1[h1, h2, v4]
    Eu1 = permutedims(Eu1, (2, 1, 3); sign_function=global_sign)
    # Er2[v8, v7, h6] <-- Er2[v7, v8, h6]
    Er2 = permutedims(Er2, (2, 1, 3); sign_function=global_sign)
    # Er1[v7, v6, h4] <-- Er1[v6, v7, h4]
    Er1 = permutedims(Er1, (2, 1, 3); sign_function=global_sign)
    # El2[v3, v2, h5] <-- El2[v2, v3, h5]
    El2 = permutedims(El2, (2, 1, 3); sign_function=global_sign)
    # El1[v2, v1, h3] <-- El1[v1, v2, h3]
    El1 = permutedims(El1, (2, 1, 3); sign_function=global_sign)
    # Crd2[v8, h8] <-- Crd2[h8, v8]
    Crd2 = permutedims(Crd2, (2, 1); sign_function=global_sign)
    # Cru1[v6, h2] <-- Cru1[h2, v6]
    Cru1 = permutedims(Cru1, (2, 1); sign_function=global_sign)
    # Cld2[v3, h7] <-- Cld2[h7, v3]
    Cld2 = permutedims(Cld2, (2, 1); sign_function=global_sign)
    # Clu1[v1, h1] <-- Clu1[h1, v1]
    Clu1 = permutedims(Clu1, (2, 1); sign_function=global_sign)
    # Timp[v5, v4, h6, h4, h5, h3] <-- Timp[h3, h5, h4, h6, v4, v5]
    Timp = permutedims(Timp, (6, 5, 4, 3, 2, 1); sign_function=global_sign)

    # Reverse the arrows in the environment tensors to match the horizontal bond case
    Crd2 = index_conjugation(Crd2, (1, 2))
    Crd2 = add_parity_sign(Crd2, 2; sign_function=global_sign)

    Ed2 = index_conjugation(Ed2, (1, 2))
    Ed2 = add_parity_sign(Ed2, 2; sign_function=global_sign)

    Cld2 = index_conjugation(Cld2, (1, 2))
    Cld2 = add_parity_sign(Cld2, 1; sign_function=global_sign)

    El2 = index_conjugation(El2, (1, 2))
    El2 = add_parity_sign(El2, 2; sign_function=global_sign)

    El1 = index_conjugation(El1, (1, 2))
    El1 = add_parity_sign(El1, 2; sign_function=global_sign)

    Clu1 = index_conjugation(Clu1, (1, 2))
    Clu1 = add_parity_sign(Clu1, 2; sign_function=global_sign)

    Eu1 = index_conjugation(Eu1, (1, 2))
    Eu1 = add_parity_sign(Eu1, 1; sign_function=global_sign)

    Cru1 = index_conjugation(Cru1, (1, 2))
    Cru1 = add_parity_sign(Cru1, 1; sign_function=global_sign)

    Er1 = index_conjugation(Er1, (1, 2))
    Er1 = add_parity_sign(Er1, 1; sign_function=global_sign)

    Er2 = index_conjugation(Er2, (1, 2))
    Er2 = add_parity_sign(Er2, 1; sign_function=global_sign)

    norm, exp = compute_exp_hbond(Tb2, Tb1, Timp, Ed2, Eu1, Er2, Er1, El2, El1, Crd2, Cru1, Cld2, Clu1)

    return norm, exp
end



"""
Clu[r, c] ←←←←←←←← Eu[r, c] ←←←←←←←← Eu[r, c+1] ←←←←←←←← Eu[r, c+2] ←←←←←←←← ... ←←←←←←←← Eu[r, c+Δc-1] ←←←←←←←← Eu[r, c+Δc] ←←←←←←←← Cru[r, c+Δc]
    ↓                  ↓                  ↓                    ↓                                  ↓                     ↓                   ↓
    ↓                  ↓                  ↓                    ↓                                  ↓                     ↓                   ↓
    ↓                  ↓                  ↓                    ↓                                  ↓                     ↓                   ↓
    ↓                  ↓                  ↓                    ↓                                  ↓                     ↓                   ↓  
    ↓                  ↓                  ↓                    ↓                                  ↓                     ↓                   ↓
 El[r, c] ←←←←←← Timp1[r, c] ←←←←←← Tbulk[r, c+1] ←←←←←← Tbulk[r, c+2] ←←←←←← ... ←←←←←← Tbulk[r, c+Δc-1] ←←←←←← Timp2[r, c+Δc] ←←←←←← Er[r, c+Δc]
    ↓                  ↓                  ↓                    ↓                                  ↓                     ↓                   ↓
    ↓                  ↓                  ↓                    ↓                                  ↓                     ↓                   ↓
    ↓                  ↓                  ↓                    ↓                                  ↓                     ↓                   ↓
    ↓                  ↓                  ↓                    ↓                                  ↓                     ↓                   ↓
    ↓                  ↓                  ↓                    ↓                                  ↓                     ↓                   ↓
 Cld[r, c] →→→→→→→ Ed[r, c] →→→→→→→ Ed[r, c+1] →→→→→→→→→ Ed[r, c+2] →→→→→→→→ ... →→→→→→→→ Ed[r, c+Δc-1] →→→→→→→→ Ed[r, c+Δc] →→→→→→→→ Crd[r, c+Δc]

 ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

 Clu[r, c] ←←←←←←←← Eu[r, c] ←←←←←←←← Eu[r, c+1] ←←←←←←←← Eu[r, c+2] ←←←←←←←← ... ←←←←←←←← Eu[r, c+Δc-1] ←←←←←←←← Eu[r, c+Δc] ←←←←←←←← Cru[r, c+Δc]
    ↓                  ↓                  ↓                    ↓                                  ↓                     ↓                   ↓
    ↓                  ↓                  ↓                    ↓                                  ↓                     ↓                   ↓
    ↓                  ↓                  ↓                    ↓                                  ↓                     ↓                   ↓
    ↓                  ↓                  ↓                    ↓                                  ↓                     ↓                   ↓  
    ↓                  ↓                  ↓                    ↓                                  ↓                     ↓                   ↓
 El[r, c] ←←←←←← Tbulk[r, c] ←←←←←← Tbulk[r, c+1] ←←←←←← Tbulk[r, c+2] ←←←←←← ... ←←←←←← Tbulk[r, c+Δc-1] ←←←←←← Tbulk[r, c+Δc] ←←←←←← Er[r, c+Δc]
    ↓                  ↓                  ↓                    ↓                                  ↓                     ↓                   ↓
    ↓                  ↓                  ↓                    ↓                                  ↓                     ↓                   ↓
    ↓                  ↓                  ↓                    ↓                                  ↓                     ↓                   ↓
    ↓                  ↓                  ↓                    ↓                                  ↓                     ↓                   ↓
    ↓                  ↓                  ↓                    ↓                                  ↓                     ↓                   ↓
 Cld[r, c] →→→→→→→ Ed[r, c] →→→→→→→ Ed[r, c+1] →→→→→→→→→ Ed[r, c+2] →→→→→→→→ ... →→→→→→→→ Ed[r, c+Δc-1] →→→→→→→→ Ed[r, c+Δc] →→→→→→→→ Crd[r, c+Δc]
"""

function correlation_function_horizontal(
    Tbulk_mat::Matrix{Grassmann{Q2, 4}}, 
    Timp1_mat::Matrix{Grassmann{Q1, 4}}, 
    Timp2_mat::Matrix{Grassmann{Q1, 4}}, 
    env::CTMRGEnv{Q2}, 
    r0::Int, c0::Int, 
    dist::Int) where {Q1, Q2}
     
    Q = promote_type(Q1, Q2)

    _, Ly = size(Tbulk_mat)

    # Lenv1[h1, v2, h2] = Clu[h1, v1] * El[v1, v2, h2]
    Lenv1 = contract(env.Clu[r0, c0], env.El[r0, c0], (2, 1); sign_function=global_sign)
    # Lenv2[h1. h2, h3] = Lenv1[h1, v2, h2] * Cld[h3, v2]
    Lenv2 = contract(Lenv1, env.Cld[r0, c0], (2, 2); sign_function=global_sign)

    Lenv_num, _ = left_move(Lenv2, env.Ed[r0, c0], Timp1_mat[r0, c0], env.Eu[r0, c0])
    Lenv_den, _ = left_move(Lenv2, env.Ed[r0, c0], Tbulk_mat[r0, c0], env.Eu[r0, c0])

    corre_vec = Vector{Q}(undef, dist)

    for Δc in 1:dist
        
        c = Nmod(c0 + Δc, Ly)
        # Renv1[h1, v2, h2] = Cru[h1, v1] * Er[v1, v2, h2]
        Renv1 = contract(env.Cru[r0, c], env.Er[r0, c], (2, 1); sign_function=global_sign)
        # Renv2[h1, h2, h3] = Renv1[h1, v2, h2] * Crd[h3, v2]
        Renv2 = contract(Renv1, env.Crd[r0, c], (2, 2); sign_function=global_sign)

        Renv_num, _ = right_move(env.Ed[r0, c], Timp2_mat[r0, c], env.Eu[r0, c], Renv2)
        Renv_den, _ = right_move(env.Ed[r0, c], Tbulk_mat[r0, c], env.Eu[r0, c], Renv2)

        corre_num = contract(Lenv_num, Renv_num, ((1, 2, 3), (1, 2, 3)); sign_function=global_sign)
        corre_den = contract(Lenv_den, Renv_den, ((1, 2, 3), (1, 2, 3)); sign_function=global_sign)
        corre_vec[Δc] = scalar(corre_num)/scalar(corre_den)

        Lenv_num_new, coef = left_move(Lenv_num, env.Ed[r0, c], Tbulk_mat[r0, c], env.Eu[r0, c])
        Lenv_den_new, _ = left_move(Lenv_den, env.Ed[r0, c], Tbulk_mat[r0, c], env.Eu[r0, c])
        Lenv_num = Lenv_num_new/coef
        Lenv_den = Lenv_den_new/coef
    end

    return corre_vec
end 

"""
    # ←←←←← (h1) ←←←←← Eu ←←←←←←←← (h4)          # ←←←←←←←← (h4) 
    #                  ↓                         #
    #                  ↓                         # 
    #                 (v1)                       #
    #                  ↓                         #
    #                  ↓                         #
  Lenv ←←←← (h2) ←←←←← T ←←←←←←←←← (h5)   ===>   # ←←←←←←←← (h5)
    #                  ↓                         #
    #                  ↓                         #
    #                 (v2)                       #
    #                  ↓                         #
    #                  ↓                         #
    # →→→→→ (h3) →→→→→ Ed →→→→→→→→ (h6)          # →→→→→→→→ (h6)
"""

function left_move(
    Lenv::Grassmann{Q1, 3}, 
    Ed::Grassmann{Q2, 3}, 
    T::Grassmann{Q3, 4}, 
    Eu::Grassmann{Q2, 3}) where {Q1, Q2, Q3}

    # out1[h1, h2, h6, v2] = Lenv[h1, h2, h3] * Ed[h3, h6, v2]
    out1 = contract(Lenv, Ed, (3, 1); sign_function=global_sign)
    # out2[h1, h6, h5, v1] = out1[h1, h2, h6, v2] * T[h2, h5, v1, v2]
    out2 = contract(out1, T, ((2, 4), (1, 4)); sign_function=global_sign)
    # out3[h4, h5, h6] <-- out3[h4, h6, h5] = Eu[h1, h4, v1] * out2[h1, h6, h5, v1]
    out3 = contract(Eu, out2, ((1, 3), (1, 4)); perm=(1, 3, 2), sign_function=global_sign)

    coef = norm(out3)

    return out3, coef
end

"""
 (h1) ←←←←←←←←← Eu ←←←← (h4) ←←←← #             (h1) ←←←←←←←← #  
                ↓                 #                           # 
                ↓                 #                           #
               (v1)               #                           #
                ↓                 #                           #  
                ↓                 #                           #
 (h2) ←←←←←←←←  T ←←←← (h5) ←←←← Renv    ===>   (h2) ←←←←←←←← #                
                ↓                 #                           #
                ↓                 #                           #
               (v2)               #                           #
                ↓                 #                           #
                ↓                 #                           #
 (h3) →→→→→→→→ Ed →→→→ (h6) →→→→  #             (h3) →→→→→→→→ # 
"""

function right_move(
    Ed::Grassmann{Q1, 3}, 
    T::Grassmann{Q2, 4}, 
    Eu::Grassmann{Q1, 3}, 
    Renv::Grassmann{Q3, 3}) where {Q1, Q2, Q3}

    # out1[h5, h6, h1, v1] = Renv[h4, h5, h6] * Eu[h1, h4, v1]
    out1 = contract(Renv, Eu, (1, 2); sign_function=global_sign)
    # out2[h6, h1, h2, v2] = out1[h5, h6, h1, v1] * T[h2, h5, v1, v2]
    out2 = contract(out1, T, ((1, 4), (2, 3)); sign_function=global_sign)
    # out3[h1, h2, h3] = out2[h6, h1, h2, v2] * Ed[h3, h6, v2]
    out3 = contract(out2, Ed, ((1, 4), (2, 3)); sign_function=global_sign)

    coef = norm(out3)

    return out3, coef
end

function correlation_function_vertical(    
    Tbulk_mat::Matrix{Grassmann{Q2, 4}}, 
    Timp1_mat::Matrix{Grassmann{Q1, 4}}, 
    Timp2_mat::Matrix{Grassmann{Q1, 4}}, 
    env::CTMRGEnv{Q2}, 
    r0::Int, c0::Int, 
    dist::Int) where {Q1, Q2}

    Q = promote_type(Q1, Q2)

    Lx, _ = size(Tbulk_mat)

    # Uenv1[v1, h, v2] = Clu[dum, v1] * Eu[dum, h, v2]
    Uenv1 = contract(env.Clu[r0, c0], env.Eu[r0, c0], (1, 1); sign_function=global_sign)
    # Uenv2[v1, v2, v3] = Uenv1[v1, dum, v2] * env.Cru[dum, v3]
    Uenv2 = contract(Uenv1, env.Cru[r0, c0], (2, 1); sign_function=global_sign)

    Uenv_num, _ = up_move(Uenv2, env.El[r0, c0], Timp1_mat[r0, c0], env.Er[r0, c0])
    Uenv_den, _ = up_move(Uenv2, env.El[r0, c0], Tbulk_mat[r0, c0], env.Er[r0, c0])

    corre_vec = Vector{Q}(undef, dist)

    for Δr in 1:dist

        r = Nmod(r0 + Δr, Lx)
        # Denv1[v1, h, v2] = Cld[dum, v1] * Ed[dum, h, v2]
        Denv1 = contract(env.Cld[r, c0], env.Ed[r, c0], (1, 1); sign_function=global_sign)
        # Denv2[v1, v2, v3] = Denv1[v1, dum, v2] * Crd[dum, v3]
        Denv2 = contract(Denv1, env.Crd[r, c0], (2, 1); sign_function=global_sign)

        Denv_num, _ = down_move(env.El[r, c0], Timp2_mat[r, c0], env.Er[r, c0], Denv2)
        Denv_den, _ = down_move(env.El[r, c0], T_bulk[r, c0], env.Er[r, c0], Denv2)

        corre_num = contract(Uenv_num, Denv_num, ((1, 2, 3), (1, 2, 3)); sign_function=global_sign)
        corre_den = contract(Uenv_den, Denv_den, ((1, 2, 3), (1, 2, 3)); sign_function=global_sign)
        corre_vec[Δr] = scalar(corre_num)/scalar(corre_den)

        Uenv_num_new, coef = up_move(Uenv_num, env.El[r, c0], Tbulk_mat[r, c0], env.Er[r, c0])
        Uenv_den_new, _ = up_move(Uenv_den, env.El[r, c0], Tbulk_mat[r, c0], env.Er[r, c0])
        Uenv_num = Uenv_num_new/coef
        Uenv_den = Uenv_den_new/coef
    end

    return corre_vec
end

"""
    ############# Uenv #############
    ↓               ↓              ↑
    ↓               ↓              ↑
   [v1]            [v2]           [v3]
    ↓               ↓              ↑
    ↓               ↓              ↑
    El ←←← [h1] ←←← T ←←← [h2] ←←← Er 
    ↓               ↓              ↑
    ↓               ↓              ↑
    ↓               ↓              ↑
    ↓               ↓              ↑
   [v4]            [v5]           [v6]
"""

function up_move(
    Uenv::Grassmann{Q1, 3}, 
    El::Grassmann{Q2, 3}, 
    T::Grassmann{Q3, 4}, 
    Er::Grassmann{Q2, 3}) where {Q1, Q2, Q3}

    # Uenv1[v2, v3, v4, h1] = Uenv[v1, v2, v3] * El[v1, v4, h1]
    Uenv1 = contract(Uenv, El, (1, 1); sign_function=global_sign)
    # Uenv2[v3, v4, h2, v5] = Uenv1[v2, v3, v4, h1] * T[h1, h2, v2, v5]
    Uenv2 = contract(Uenv1, T, ((1, 4), (3, 1)); sign_function=global_sign)
    # out[v4, v5, v6] = Uenv2[v3, v4, h2, v5] * Er[v3, v6, h2]
    out = contract(Uenv2, Er, ((1, 3), (1, 3)); sign_function=global_sign)
  
    coef = norm(out)

    return out, coef
end

"""
   [v4]            [v5]           [v6]
    ↓               ↓              ↑
    ↓               ↓              ↑ 
    ↓               ↓              ↑
    ↓               ↓              ↑
    El ←←← [h1] ←←← T ←←← [h2] ←←← Er
    ↓               ↓              ↑
    ↓               ↓              ↑
   [v1]            [v2]           [v3]
    ↓               ↓              ↑
    ↓               ↓              ↑
    ############# Denv #############
"""

function down_move(
    El::Grassmann{Q1, 3}, 
    T::Grassmann{Q2, 4}, 
    Er::Grassmann{Q1, 3}, 
    Denv::Grassmann{Q3, 3}) where {Q1, Q2, Q3}

    # Denv1[v2, v3, v4, h1] = Denv[v1, v2, v3] * El[v4, v1, h1]
    Denv1 = contract(Denv, El, (1, 2); sign_function=global_sign)
    # Denv2[v3, v4, h2, v5] = Denv1[v2, v3, v4, h1] * T[h1, h2, v5, v2]
    Denv2 = contract(Denv1, T, ((1, 4), (4, 1)); sign_function=global_sign)
    # out[v4, v5, v6] = Denv2[v3, v4, h2, v5] * Er[v6, v3, h2]
    out = contract(Denv2, Er, ((1, 3), (2, 3)); sign_function=global_sign)
  
    coef = norm(out)

    return out, coef
end

function _left_move_keep_open(
    Lenv::Grassmann{Q1, N1},
    Ed::Grassmann{Q2, 3},
    T::Grassmann{Q3, N2},
    Eu::Grassmann{Q2, 3}) where {Q1, Q2, Q3, N1, N2}

    N1 >= 3 || throw(ArgumentError("left environment must have at least three legs"))
    N2 >= 4 || throw(ArgumentError("bulk tensor must have at least four lattice legs"))
    open_env = N1 - 3
    open_bulk = N2 - 4

    # out1[h1, h2, open_env..., h6, v2] = Lenv[h1, h2, h3, open_env...] * Ed[h3, h6, v2]
    out1 = contract(Lenv, Ed, (3, 1); sign_function=global_sign)
    # out2[h1, open_env..., h6, h5, v1, open_bulk...] = out1[h1, h2, open_env..., h6, v2] * T[h2, h5, v1, v2, open_bulk...]
    out2 = contract(out1, T, ((2, open_env + 4), (1, 4)); sign_function=global_sign)
    # out3[h4, open_env..., h6, h5, open_bulk...] = Eu[h1, h4, v1] * out2[...]
    out3 = contract(Eu, out2, ((1, 3), (1, open_env + 4)); sign_function=global_sign)

    nout = 3 + open_env + open_bulk
    perm_dst = (1, open_env + 3, open_env + 2, (2:(open_env + 1))..., ((open_env + 4):nout)...)

    return permutedims(out3, perm_dst; sign_function=global_sign), norm(out3)
end

function _up_move_keep_open(
    Uenv::Grassmann{Q1, N1},
    El::Grassmann{Q2, 3},
    T::Grassmann{Q3, N2},
    Er::Grassmann{Q2, 3}) where {Q1, Q2, Q3, N1, N2}

    N1 >= 3 || throw(ArgumentError("upper environment must have at least three legs"))
    N2 >= 4 || throw(ArgumentError("bulk tensor must have at least four lattice legs"))
    open_env = N1 - 3
    open_bulk = N2 - 4

    # Uenv1[v2, v3, open_env..., v4, h1] = Uenv[v1, v2, v3, open_env...] * El[v1, v4, h1]
    Uenv1 = contract(Uenv, El, (1, 1); sign_function=global_sign)
    # Uenv2[v3, open_env..., v4, h2, v5, open_bulk...] = Uenv1[v2, v3, open_env..., v4, h1] * T[h1, h2, v2, v5, open_bulk...]
    Uenv2 = contract(Uenv1, T, ((1, open_env + 4), (3, 1)); sign_function=global_sign)
    # out[open_env..., v4, v5, open_bulk..., v6] = Uenv2[v3, open_env..., v4, h2, v5, open_bulk...] * Er[v3, v6, h2]
    out = contract(Uenv2, Er, ((1, open_env + 3), (1, 3)); sign_function=global_sign)

    nout = 3 + open_env + open_bulk
    perm_dst = (open_env + 1, open_env + 2, nout, (1:open_env)..., ((open_env + 3):(nout - 1))...)

    return permutedims(out, perm_dst; sign_function=global_sign), norm(out)
end
