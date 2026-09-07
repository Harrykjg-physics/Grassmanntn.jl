# Grassmann version of Corner Transfer Matrix Renormalization Group (GCTMRG) for contracting general 2D Grassmann tensor networks
# The truncation scheme is based on PRL. 113, 046402 (2014), P.Corboz and M.Troyer
# The goal of GCTMRG is to iteratively optimize the environment tensors for each bulk tensor in a unit cell

# const global_sign = auto_sign --- Z2 symmetric fermionic model
# const global_sign = trivial_sign --- Z2 symmetric bosonic/spin model

"""
In CTMRG, the X/Y-axis is defined as :

O →→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→ Y 
↓
↓              ↓               ↓
↓              ↓               ↓
↓     ←←←← T[x-1, y] ←←←← T[x-1, y+1] ←←←←
↓              ↓               ↓
↓              ↓               ↓
↓     ←←←←  T[x, y]  ←←←← T[x, y+1] ←←←←         
↓              ↓               ↓
↓              ↓               ↓
↓
X

Positive direction in X axis <---> down direction
Negative direction in X axis <---> up direction
Positive direction in Y axis <---> right direction
Negative direction in Y axis <---> left direction

The indices order of each kind of tensor is consistently defined as :

                                 [v1]
                                  ↓
                                  ↓ 
(1) rank-4 bulk tensor  [h1] ←←←← T ←←←← [h2]  : T[h1, h2, v1, v2]
                                  ↓
                                  ↓
                                 [v2]

                          [v1]                                       [v1]
                           ↓                                          ↑
                           ↓                                          ↑
(2) rank-3 edge tensors :  L ←←←←←← [h] : L[v1, v2, h]     [h] ←←←←←← R  :  R[v1, v2, h]
                           ↓                                          ↑   
                           ↓                                          ↑ 
                          [v2]                                       [v2]

               [h1] ←←←←←← U ←←←←←← [h2]  : U[h1, h2, v]             [v]   :  D[h1, h2, v]
                           ↓                                          ↓
                           ↓                                          ↓
                          [v]                             [h1] →→→→→→ D →→→→→→ [h2]
                
(3) rank-2 corner matrices : 

                          Clu ←←←←←← [h]                   [h] ←←←←←← Cru
                           ↓              :  Clu[h, v]                 ↑  :  Cru[h, v]
                           ↓                                           ↑
                          [v]                                         [v]

                          [v]                                         [v]
                           ↓              :  Cld[h, v]                 ↑  :  Crd[h, v]
                           ↓                                           ↑
                          Cld →→→→→→ [h]                   [h] →→→→→→ Crd

Arguments: 
    `T_bulk`: a unit cell of Lx × Ly Grassmann bulk tensors 
    `T_imp`: a unit cell of Lx × Ly Grassmann impurity tensors 
    `ctmrg_env`: a custom Structure containing CTMRG environment tensors 
    `χ`: the bond dimension of environment tensors 
    `ctmrg_iter` and `ctmrg_tol` control the number of CTMRG iterations
    `start` : count the CTMRG iterations from `start`+1 (default=0)
    `average_trunc` set whether to equally truncate the even and odd sectors, i.e. set χe = χo = χ/2 (default=false)
    `verbosity` controls the printing level (default=0)
    `save_iter` determines the number of steps to save the CTMRG environment tensors (default=0)
    `save_filename` specifies the filename to store the CTMRG environment tensors (default="ctmrg_env")
"""

function run_GCTMRG!(
    T_bulk::Matrix{Grassmann{Q, 4}}, 
    T_imp::Matrix{Grassmann{Q, 4}}, 
    ctmrg_env::CTMRGEnv, 
    χ::Int; 
    ctmrg_iter::Int=100, 
    ctmrg_tol::Float64=1e-12, 
    start::Int=0, 
    average_trunc::Bool=false, 
    verbosity::Int=0, 
    save_iter::Int=0, 
    save_filename::String="ctmrg_env") where {Q}
 
    Lx, Ly = size(T_bulk)

    coef_iter = ones(Float64, 4, Lx, Ly)
    coef = similar(coef_iter)

    Λd_iter, Λu_iter, Λl_iter, Λr_iter = prepare_Λ(Lx, Ly, χ)

    expval_avg_tmp = one(Q)
    count = 0

    for iter = (start+1):(start+ctmrg_iter)

        ti = time()

        coef[1, :, :], trunc_err_d, Λd = down_move!(T_bulk, ctmrg_env, χ; average_trunc=average_trunc)
        max_trunc_err_d = maximum(trunc_err_d)
        max_Λ_err_d = maximum(compare_weights(Λd, Λd_iter))

        coef[2, :, :], trunc_err_u, Λu = up_move!(T_bulk, ctmrg_env, χ; average_trunc=average_trunc)
        max_trunc_err_u = maximum(trunc_err_u)
        max_Λ_err_u = maximum(compare_weights(Λu, Λu_iter))

        coef[3, :, :], trunc_err_l, Λl = left_move!(T_bulk, ctmrg_env, χ; average_trunc=average_trunc)
        max_trunc_err_l = maximum(trunc_err_l)
        max_Λ_err_l = maximum(compare_weights(Λl, Λl_iter))

        coef[4, :, :], trunc_err_r, Λr = right_move!(T_bulk, ctmrg_env, χ; average_trunc=average_trunc)
        max_trunc_err_r = maximum(trunc_err_r)
        max_Λ_err_r = maximum(compare_weights(Λr, Λr_iter))

        tf = time()

        if verbosity > 0

            @info @sprintf " Down move of CTMRG Iterations %i ==>  
            max_trunc_err_d :  %.6e   max_Λ_err_d : %.6e  " iter max_trunc_err_d max_Λ_err_d

            @info @sprintf " Up move of CTMRG Iterations %i ==>  
            max_trunc_err_u :  %.6e   max_Λ_err_u : %.6e  " iter max_trunc_err_u max_Λ_err_u

            @info @sprintf " Left move of CTMRG Iterations %i ==>  
            max_trunc_err_l :  %.6e   max_Λ_err_l : %.6e  " iter max_trunc_err_l max_Λ_err_l

            @info @sprintf " Right move of CTMRG Iterations %i ==>  
            max_trunc_err_r :  %.6e   max_Λ_err_r : %.6e  " iter max_trunc_err_r max_Λ_err_r
        end

        max_coef_err = maximum(abs.((coef - coef_iter)))/maximum(coef_iter)
        max_trunc_err = max(max(max_trunc_err_d, max_trunc_err_u), max(max_trunc_err_l, max_trunc_err_r))
        max_Λ_err = max(max(max_Λ_err_d, max_Λ_err_u), max(max_Λ_err_l, max_Λ_err_r))
        
        @info @sprintf "        "
        @info @sprintf " CTMRG Iterations %i ==>  Δt : %.4f   
        max_coef_err :  %.6e   max_trunc_err : %.6e  max_Λ_err : %.6e  " iter (tf-ti) max_coef_err max_trunc_err max_Λ_err
        @info @sprintf "        "

        # Whether to calculate and print the expectation value after each iteration
        if verbosity > 1  

            ti = time()
            _, expval = compute_exp_site(T_bulk, T_imp, ctmrg_env)
            tf = time()

            expval_avg = sum(expval)/(Lx*Ly)
            expval_diff = abs((expval_avg - expval_avg_tmp) / expval_avg_tmp)

            @info @sprintf " Expectation Calculation of CTMRG Iterations %i ==>  
            Δt : %.4f    Exp_avg : %.8f    Exp_diff : %.6e  " iter (tf-ti)  expval_avg expval_diff 
            @info @sprintf "        "

            # the convergence in the expectation value
            if (expval_diff < ctmrg_tol) && (count == 10)
                savestr = "χ$χ"*"iter$iter"
                save(ctmrg_env, save_filename, savestr)
                break
            elseif (expval_diff < ctmrg_tol) && (count < 10)
                expval_avg_tmp = copy(expval_avg)
                count += 1
            else
                expval_avg_tmp = copy(expval_avg)
            end
        end

        if save_iter > 0 && mod(iter, save_iter) == 0
            savestr = "χ$χ"*"iter$iter"
            save(ctmrg_env, save_filename, savestr)
        end

        if max_Λ_err < ctmrg_tol
            savestr = "χ$χ"*"iter$iter"
            save(ctmrg_env, save_filename, savestr)
            break
        else
            copyto!(coef_iter, coef)
            Λd_iter = copy(Λd)
            Λu_iter = copy(Λu)
            Λl_iter = copy(Λl)
            Λr_iter = copy(Λr)
        end 
    end
end

# Monitor the energy convergence
function run_GCTMRG!(
    peps::Square_GPEPS{Q1},
    T_bulk::Matrix{Grassmann{Q1, 4}}, 
    H_bond::Grassmann{Q2, 4}, 
    ctmrg_env::CTMRGEnv{Q1}, 
    χ::Int; 
    ctmrg_iter::Int=100, 
    ctmrg_tol::Float64=1e-12, 
    start::Int=0, 
    average_trunc::Bool=false, 
    verbosity::Int=0, 
    save_iter::Int=0, 
    save_filename::String="ctmrg_env") where {Q1, Q2}
 
    Lx, Ly = size(T_bulk)

    coef_iter = ones(Float64, 4, Lx, Ly)
    coef = similar(coef_iter)

    Λd_iter, Λu_iter, Λl_iter, Λr_iter = prepare_Λ(Lx, Ly, χ)

    Q = promote_type(Q1, Q2)
    expval_avg_tmp = one(Q)
    count = 0

    for iter = (start+1):(start+ctmrg_iter)

        ti = time()

        coef[1, :, :], trunc_err_d, Λd = down_move!(T_bulk, ctmrg_env, χ; average_trunc=average_trunc)
        max_trunc_err_d = maximum(trunc_err_d)
        max_Λ_err_d = maximum(compare_weights(Λd, Λd_iter))

        coef[2, :, :], trunc_err_u, Λu = up_move!(T_bulk, ctmrg_env, χ; average_trunc=average_trunc)
        max_trunc_err_u = maximum(trunc_err_u)
        max_Λ_err_u = maximum(compare_weights(Λu, Λu_iter))

        coef[3, :, :], trunc_err_l, Λl = left_move!(T_bulk, ctmrg_env, χ; average_trunc=average_trunc)
        max_trunc_err_l = maximum(trunc_err_l)
        max_Λ_err_l = maximum(compare_weights(Λl, Λl_iter))

        coef[4, :, :], trunc_err_r, Λr = right_move!(T_bulk, ctmrg_env, χ; average_trunc=average_trunc)
        max_trunc_err_r = maximum(trunc_err_r)
        max_Λ_err_r = maximum(compare_weights(Λr, Λr_iter))

        tf = time()

        if verbosity > 0

            @info @sprintf " Down move of CTMRG Iterations %i ==>  
            max_trunc_err_d :  %.6e   max_Λ_err_d : %.6e  " iter max_trunc_err_d max_Λ_err_d

            @info @sprintf " Up move of CTMRG Iterations %i ==>  
            max_trunc_err_u :  %.6e   max_Λ_err_u : %.6e  " iter max_trunc_err_u max_Λ_err_u

            @info @sprintf " Left move of CTMRG Iterations %i ==>  
            max_trunc_err_l :  %.6e   max_Λ_err_l : %.6e  " iter max_trunc_err_l max_Λ_err_l

            @info @sprintf " Right move of CTMRG Iterations %i ==>  
            max_trunc_err_r :  %.6e   max_Λ_err_r : %.6e  " iter max_trunc_err_r max_Λ_err_r
        end

        max_coef_err = maximum(abs.((coef - coef_iter)))/maximum(coef_iter)
        max_trunc_err = max(max(max_trunc_err_d, max_trunc_err_u), max(max_trunc_err_l, max_trunc_err_r))
        max_Λ_err = max(max(max_Λ_err_d, max_Λ_err_u), max(max_Λ_err_l, max_Λ_err_r))
        
        @info @sprintf "        "
        @info @sprintf " CTMRG Iterations %i ==>  Δt : %.4f   
        max_coef_err :  %.6e   max_trunc_err : %.6e  max_Λ_err : %.6e  " iter (tf-ti) max_coef_err max_trunc_err max_Λ_err
        @info @sprintf "        "

        # Whether to calculate and print the expectation value after each iteration
        if verbosity > 1  

            ti = time()
            _, Eh = compute_exp_hbond(T_bulk, peps, H_bond, ctmrg_env)
            _, Ev = compute_exp_vbond(T_bulk, peps, H_bond, ctmrg_env)
            tf = time()

            expval_avg = (sum(Eh) + sum(Ev))/(size(Eh, 1) * size(Eh, 2))
            expval_diff = abs((expval_avg - expval_avg_tmp) / expval_avg_tmp)

            @info @sprintf " Expectation Calculation of CTMRG Iterations %i ==>  
            Δt : %.4f    Exp_avg : %.8f    Exp_diff : %.6e  " iter (tf-ti)  expval_avg expval_diff 
            @info @sprintf "        "

            # the convergence in the expectation value
            if (expval_diff < ctmrg_tol) && (count == 10)
                savestr = "χ$χ"*"iter$iter"
                save(ctmrg_env, save_filename, savestr)
                break
            elseif (expval_diff < ctmrg_tol) && (count < 10)
                expval_avg_tmp = copy(expval_avg)
                count += 1
            else
                expval_avg_tmp = copy(expval_avg)
            end
        end

        if save_iter > 0 && mod(iter, save_iter) == 0
            savestr = "χ$χ"*"iter$iter"
            save(ctmrg_env, save_filename, savestr)
        end

        if max_Λ_err < ctmrg_tol
            savestr = "χ$χ"*"iter$iter"
            save(ctmrg_env, save_filename, savestr)
            break
        else
            copyto!(coef_iter, coef)
            Λd_iter = copy(Λd)
            Λu_iter = copy(Λu)
            Λl_iter = copy(Λl)
            Λr_iter = copy(Λr)
        end 
    end
end

#################################### CTMRG core functions  ####################################

function down_move!(
    Tbulk::Matrix{Grassmann{T, 4}}, 
    env::CTMRGEnv, 
    χ::Int; 
    average_trunc::Bool=true) where {T}

    Lx, Ly = size(Tbulk)

    coef = Matrix{Float64}(undef, Lx, Ly)
    trunc_err = Matrix{Float64}(undef, Lx, Ly)
    Λ = Matrix{GrassmannMatrix{Float64}}(undef, Lx, Ly)

    for x = Lx:-1:1
        up = Nmod(x - 1, Lx)
        P = Vector{Grassmann{T, 3}}(undef, Ly)
        Q = Vector{Grassmann{T, 3}}(undef, Ly)
        for y = 1:Ly 
            rt = Nmod(y + 1, Ly)
            P[rt], Q[rt], Λ[x, y], trunc_err[x, y] = generate_projector_dn(
                Tbulk[up, y], Tbulk[up, rt], Tbulk[x, y], Tbulk[x, rt], 
                env.El[up, y], env.El[x, y], env.Er[up, rt], env.Er[x, rt], 
                env.Eu[up, y], env.Eu[up, rt], env.Ed[x, y], env.Ed[x, rt], 
                env.Clu[up, y], env.Cru[up, rt], env.Cld[x, y], env.Crd[x, rt], χ; 
                average_trunc=average_trunc
                )
        end
        for y = 1:Ly 
            rt = Nmod(y + 1, Ly)
            env.Cld[up, y], env.Ed[up, y], env.Crd[up, y], coef[x, y] = do_truncation_dn(
                env.El[x, y], Tbulk[x, y], env.Er[x, y], 
                env.Cld[x, y], env.Ed[x, y], env.Crd[x, y], 
                P[y], Q[y], P[rt], Q[rt]
                )
        end
    end

    return coef, trunc_err, Λ
end

"""
 C1 ←←←←←←←←←← U1 ←←←← (h1)           (h3p) ←←← U2 ←←←←←←←←←← C2
 ↓             ↓                                ↓             ↑
 ↓             ↓                                ↓             ↑
 ↓             ↓                                ↓             ↑
 ↓             ↓                                ↓             ↑
 L1 ←←←←←←←←←← T1 ←←←← (h2)           (h4p) ←←← T2 ←←←←←←←←←← R1
 ↓             ↓                                ↓             ↑
 ↓             ↓                                ↓             ↑
 ↓             ↓                                ↓             ↑
 ↓             ↓                                ↓             ↑
 L2 ←←←←←←←←←← T3 ←←←← (h5)   *****   (h5p) ←←← T4 ←←←←←←←←← R2
 ↓             ↓                                ↓             ↑
 ↓             ↓                                ↓             ↑
 ↓             ↓                                ↓             ↑
 ↓             ↓                                ↓             ↑
 C3 →→→→→→→→→→ D1 →→→ (h6p)   *****   (h6) →→→→ D2 →→→→→→→→→→ C4
"""

function generate_projector_dn(
    T1::Grassmann{T, 4}, T2::Grassmann{T, 4}, T3::Grassmann{T, 4}, T4::Grassmann{T, 4}, 
    L1::Grassmann{T, 3}, L2::Grassmann{T, 3}, R1::Grassmann{T, 3}, R2::Grassmann{T, 3}, 
    U1::Grassmann{T, 3}, U2::Grassmann{T, 3}, D1::Grassmann{T, 3}, D2::Grassmann{T, 3}, 
    C1::Grassmann{T, 2}, C2::Grassmann{T, 2}, C3::Grassmann{T, 2}, C4::Grassmann{T, 2}, 
    χ::Int; average_trunc::Bool=true) where {T}

    LU = generate_LU(C1, U1, L1, T1)
    LD = generate_LD(L2, C3, D1, T3)
    RU = generate_RU(C2, R1, U2, T2)
    RD = generate_RD(R2, C4, D2, T4)

    LUD = generate_LUD(LU, LD)
    RUD = generate_RUD(RU, RD)

    # RUD_perm[(h5p, h6), (h3p, h4p)] <-- RUD[(h3p, h4p), (h5p, h6)]
    RUD_perm = permutedims(RUD, (3, 4, 1, 2); sign_function=global_sign)
    # RUD_perm_t1[(h5p, h6), (h3p, h4p)] = (RUD_perm × (-1)^(h5 × h6))[(h5p, h6), (h3p, h4p)]
    RUD_perm_t1 = add_perm_sign(RUD_perm, (2, 1, 3, 4); sign_function=global_sign)
    # RUD_perm_t2[(h5p, h6p), (h3p, h4p)] = (RUD_perm_t1 × (-1)^h6)[(h5p, h6), (h3p, h4p)]
    RUD_perm_t2 = add_parity_sign(index_conjugation(RUD_perm_t1, 2), 2; sign_function=global_sign)
    # LUD_t[(h1, h2), (h5, h6)] <-- LUD[(h1, h2), (h5, h6p)]
    LUD_t = index_conjugation(LUD, 4)

    # W[(h1, h2), (h3p, h4p)] = LUD_t[(h1, h2), (dum1, dum2)]  * RUD_perm_t2[(dum1, dum2), (h3p, h4p)]
    # Time —— O(χ³D³) —— O(d¹²), Memory —— O(χ²D²) —— O(d⁸)
    W = contract(LUD_t, RUD_perm_t2, ((3, 4), (1, 2)))
    # W[(h1, h2), (h3p, h4p)] --> U[(h1, h2), x], S[xp, y], Vdag[yp, (h3p, h4p)] --> V[(h3, h4), y]
    U, S, V, trunc_err = gsvd(W, (1, 2), (3, 4), χ; average_trunc=average_trunc)

    invsqrtS = inv(sqrt(S))

    # P1[(h5p, h6p), y] = RUD_perm_t2[(h5p, h6p), (h3p, h4p)] * V[(h3, h4), y]
    P1 = contract(RUD_perm_t2, V, ((3, 4), (1, 2)))
    # P2[(h5p, h6p), y] = P1[(h5p, h6p), dum] * invsqrtS[dum, y]
    P2 = contract(P1, invsqrtS, (3, 1))
    # P3[(h5p, h6p), yp] <-- P2[(h5p, h6p), y]
    P3 = index_conjugation(P2, 3)
    # P4[(h5p, h6), yp] = (P3 × (-1)^h6p)[(h5p, h6p), yp]
    P4 = add_parity_sign(index_conjugation(P3, 2), 2; sign_function=global_sign)
    # P[(h5p, h6), yp] = (P4 × (-1)^(h5 × h6))[(h5p, h6), yp]
    P = add_perm_sign(P4, (2, 1, 3); sign_function=global_sign)

    # Q1[xp, (h1p, h2p)] = invsqrtS[xp, dum] * conj(U)[(h1p, h2p), dum]
    Q1 = contract(invsqrtS, U, (2, 3); cj=(false, true))
    # Q2[xp, (h5, h6)] = Q1[xp, (h1p, h2p)] * LUD_t[(h1, h2), (h5, h6)]
    Q2 = contract(Q1, LUD_t, ((2, 3), (1, 2)))
    # Q3[x, (h5, h6)] = (Q2×(-1)^xp)[xp, (h5, h6)]
    Q3 = add_parity_sign(index_conjugation(Q2, 1), 1; sign_function=global_sign)
    # Q[x, (h5, h6p)] <-- Q3[x, (h5, h6)]
    Q = index_conjugation(Q3, 3)

    return P, Q, S/maximum(S), trunc_err
end

"""
(v1)                                                            (v1)                                                            (v1)
 ↓                                                               ↓                                                                ↑
 ↓                                                               ↓                                                                ↑
 ↓                                                               ↓                                                                ↑
 ↓                                                               ↓                                                                ↑
 L ←←← (h2) ←←←                                     ←←← (h2) ←←← Tb ←←← (h4) ←←←                                     ←←← (h2) ←←← R 
 ↓              ↖                                 ↙              ↓              ↖                                 ↙              ↑
 ↓                ↖                             ↙                ↓                ↖                             ↙                ↑
(v2)                P1 →→ (h1) →→   →→ (h1) →→ Q1               (v2)                P2 →→ (h6) →→   →→ (h1) →→ Q2               (v2)
 ↓                 ↗                             ↘               ↓                ↗                             ↘                ↑
 ↓               ↗                                 ↘             ↓              ↗                                 ↘              ↑
 C3 →→→ (h3) →→→                                     →→→ (h3) →→→ D →→→ (h5) →→→                                     →→→ (h3) →→→ C4
"""

function do_truncation_dn(
    L::Grassmann{T, 3}, Tb::Grassmann{T, 4}, R::Grassmann{T, 3}, 
    C3::Grassmann{T, 2}, D::Grassmann{T, 3}, C4::Grassmann{T, 2}, 
    P1::Grassmann{T, 3}, Q1::Grassmann{T, 3}, 
    P2::Grassmann{T, 3}, Q2::Grassmann{T, 3}) where {T}

    # C̃3a[v1, h2, h3] = L[v1, v2, h2] * C3[h3, v2]
    C̃3a = contract(L, C3, (2, 2); sign_function=global_sign)
    # C̃3[h1, v1] <-- C̃3[v1, h1] = C̃3a[v1, h2, h3] * P1[h2, h3, h1]
    C̃3 = contract(C̃3a, P1, ((2, 3), (1, 2)); perm=(2, 1), sign_function=global_sign)

    # D̃a[h1, h3, h4, v1, v2] = Q1[h1, h2, h3] * Tb[h2, h4, v1, v2]
    D̃a = contract(Q1, Tb, (2, 1); sign_function=global_sign)
    # D̃b[h3, h4, v2, h6] <-- D̃b[h3, v2, h4, h6] = D[h3, h5, v2] * P2[h4, h5, h6]
    D̃b = contract(D, P2, (2, 2); perm=(1, 3, 2, 4), sign_function=global_sign)
    # D̃[h1, h6, v1] <-- D̃[h1, v1, h6] = D̃a[h1, h3, h4, v1, v2] * D̃b[h3, h4, v2, h6]
    D̃ = contract(D̃a, D̃b, ((2, 3, 5), (1, 2, 3)); perm=(1, 3, 2), sign_function=global_sign)

    # C̃4a[h1, h3, v1, v2] = Q2[h1, h2, h3] * R[v1, v2, h2]
    C̃4a = contract(Q2, R, (2, 3); sign_function=global_sign)
    # C̃4[h1, v1] = C̃4a[h1, h3, v1, v2] * C4[h3, v2]
    C̃4 = contract(C̃4a, C4, ((2, 4), (1, 2)); sign_function=global_sign)

    coefC3 = maximum(abs(C̃3))
    coefD = maximum(abs(D̃))
    coefC4 = maximum(abs(C̃4))
    coef = coefC3 * coefD * coefC4

    C̃3 /= coefC3
    D̃ /= coefD
    C̃4 /= coefC4

    return C̃3, D̃, C̃4, coef
end

function up_move!(
    Tbulk::Matrix{Grassmann{T, 4}}, 
    env::CTMRGEnv, 
    χ::Int; 
    average_trunc::Bool=true) where {T}

    Lx, Ly = size(Tbulk)
    
    coef = Matrix{Float64}(undef, Lx, Ly)
    trunc_err = Matrix{Float64}(undef, Lx, Ly)
    Λ = Matrix{GrassmannMatrix{Float64}}(undef, Lx, Ly)

    for x = 1:Lx
        dn = Nmod(x + 1, Lx)
        P = Vector{Grassmann{T, 3}}(undef, Ly)
        Q = Vector{Grassmann{T, 3}}(undef, Ly)
        for y = 1:Ly
            rt = Nmod(y + 1, Ly)
            P[rt], Q[rt], Λ[x, y], trunc_err[x, y] = generate_projector_up(
                Tbulk[x, y], Tbulk[x, rt], Tbulk[dn, y], Tbulk[dn, rt], 
                env.El[x, y], env.El[dn, y], env.Er[x, rt], env.Er[dn, rt], 
                env.Eu[x, y], env.Eu[x, rt], env.Ed[dn, y], env.Ed[dn, rt], 
                env.Clu[x, y], env.Cru[x, rt], env.Cld[dn, y], env.Crd[dn, rt], χ; 
                average_trunc=average_trunc
                )
        end
        for y = 1:Ly
            rt = Nmod(y + 1, Ly)
            env.Clu[dn, y], env.Eu[dn, y], env.Cru[dn, y], coef[x, y] = do_truncation_up(
                env.El[x, y], Tbulk[x, y], env.Er[x, y], 
                env.Clu[x, y], env.Eu[x, y], env.Cru[x, y], 
                P[y], Q[y], P[rt], Q[rt]
                )
        end
    end

    return coef, trunc_err, Λ
end

"""
 C1 ←←←←←←←←←← U1 ←←←← (h1)   *****   (h1p) ←←← U2 ←←←←←←←←←← C2
 ↓             ↓                                ↓             ↑
 ↓             ↓                                ↓             ↑
 ↓             ↓                                ↓             ↑
 ↓             ↓                                ↓             ↑
 L1 ←←←←←←←←←← T1 ←←←← (h2)   *****   (h2p) ←←← T2 ←←←←←←←←←← R1
 ↓             ↓                                ↓             ↑
 ↓             ↓                                ↓             ↑
 ↓             ↓                                ↓             ↑
 ↓             ↓                                ↓             ↑
 L2 ←←←←←←←←←← T3 ←←←← (h3)           (h5p) ←←← T4 ←←←←←←←←←← R2
 ↓             ↓                                ↓             ↑
 ↓             ↓                                ↓             ↑
 ↓             ↓                                ↓             ↑
 ↓             ↓                                ↓             ↑
 C3 →→→→→→→→→→ D1 →→→ (h4p)           (h6) →→→→ D2 →→→→→→→→→→ C4
"""

function generate_projector_up(
    T1::Grassmann{T, 4}, T2::Grassmann{T, 4}, T3::Grassmann{T, 4}, T4::Grassmann{T, 4}, 
    L1::Grassmann{T, 3}, L2::Grassmann{T, 3}, R1::Grassmann{T, 3}, R2::Grassmann{T, 3}, 
    U1::Grassmann{T, 3}, U2::Grassmann{T, 3}, D1::Grassmann{T, 3}, D2::Grassmann{T, 3}, 
    C1::Grassmann{T, 2}, C2::Grassmann{T, 2}, C3::Grassmann{T, 2}, C4::Grassmann{T, 2}, 
    χ::Int; 
    average_trunc::Bool=true) where {T}

    LU = generate_LU(C1, U1, L1, T1)
    LD = generate_LD(L2, C3, D1, T3)
    RU = generate_RU(C2, R1, U2, T2)
    RD = generate_RD(R2, C4, D2, T4)

    LUD = generate_LUD(LU, LD)
    RUD = generate_RUD(RU, RD)

    # LUD_perm[(h3, h4p), (h1, h2)] <-- LUD[(h1, h2), (h3, h4p)] 
    LUD_perm = permutedims(LUD, (3, 4, 1, 2); sign_function=global_sign)
    # RUD_t[(h1p, h2p), (h5p, h6)] = (RUD × (-1)^(h1u × h2u))[(h1p, h2p), (h5p, h6)]
    RUD_t = add_perm_sign(RUD, (2, 1, 3, 4); sign_function=global_sign)

    # W[(h3, h4p), (h5p, h6)] = LUD_perm[(h3, h4p), (dum1, dm2)] * RUD_t[(dum1, dum2), (h5p, h6)]
    W = contract(LUD_perm, RUD_t, ((3, 4), (1, 2)))
    # W[(h3, h4p), (h5p, h6)] --> U[(h3, h4p), x], S[xp, y], Vdag[yp, (h5p, h6)] --> V[(h5, h6p), y]
    U, S, V, trunc_err = gsvd(W, (1, 2), (3, 4), χ; average_trunc=average_trunc)

    invsqrtS = inv(sqrt(S))

    # P1[(h1p, h2p), y] = RUD_t[(h1p, h2p), (h5p, h6)] * V[(h5, h6p), y]
    P1 = contract(RUD_t, V, ((3, 4), (1, 2)))
    # P2[(h1p, h2p), y] = P1[(h1p, h2p), dum] * invsqrtS[dum, y]
    P2 = contract(P1, invsqrtS, (3, 1))
    # P[(h1p, h2p), y] = (P2 × (-1)^(h1 × h2)[(h1p, h2p), y]
    P = add_perm_sign(P2, (2, 1, 3); sign_function=global_sign)

    # Q1[xp, (h3p, h4)] = invsqrtS[xp, dum] * conj(U)[(h3p, h4), dum]
    Q1 = contract(invsqrtS, U, (2, 3); cj=(false, true))
    # Q[xp, (h1, h2)] = Q1[xp, (h3p, h4)] * LUD_perm[(h3, h4p), (h1, h2)]
    Q = contract(Q1, LUD_perm, ((2, 3), (1, 2)))

    return P, Q, S/maximum(S), trunc_err
end

"""
 C1 ←← (h2) ←←                                ←← (h2) ←← U ←← (h4) ←←                                 ←← (h2) ←← C2 
 ↓             ↖                           ↙             ↓            ↖                            ↙             ↑
 ↓               ↖                       ↙               ↓              ↖                        ↙               ↑
(v2)               P1 ←← (h1)   (h1) ←← Q1              (v2)              P2 ←← (h6)    (h1) ←← Q2               (v2)
 ↓               ↙                        ↖              ↓               ↙                        ↖              ↑
 ↓             ↙                            ↖            ↓             ↙                            ↖            ↑
 L  ←← (h3) ←←                                ←← (h3) ←← Tb ←← (h5) ←←                                 ←← (h3) ←← R
 ↓                                                        ↓                                                       ↑
 ↓                                                        ↓                                                       ↑
 ↓                                                        ↓                                                       ↑
 ↓                                                        ↓                                                       ↑
(v1)                                                    (v1)                                                    (v1)
"""

function do_truncation_up(
    L::Grassmann{T, 3}, Tb::Grassmann{T, 4}, R::Grassmann{T, 3}, 
    C1::Grassmann{T, 2}, U::Grassmann{T, 3}, C2::Grassmann{T, 2}, 
    P1::Grassmann{T, 3}, Q1::Grassmann{T, 3}, 
    P2::Grassmann{T, 3}, Q2::Grassmann{T, 3}) where {T}

    # C̃1a[h2, v1, h3] = C1[h2, v2] * L[v2, v1, h3]
    C̃1a = contract(C1, L, (2, 1); sign_function=global_sign)
    # C̃1[h1, v1] <-- C̃1[v1, h1] = C̃1a[h2, v1, h3] * P1[h2, h3, h1]
    C̃1 = contract(C̃1a, P1, ((1, 3), (1, 2)); perm=(2, 1), sign_function=global_sign)

    # Ũa[h1, h3, h4, v2] = Q1[h1, h2, h3] * U[h2, h4, v2]
    Ũa = contract(Q1, U, (2, 1); sign_function=global_sign)
    # Ũb[h3, h4, v2, v1, h6] <-- Ũb[h3, v2, v1, h4, h6] = Tb[h3, h5, v2, v1] * P2[h4, h5, h6]
    Ũb = contract(Tb, P2, (2, 2); perm=(1, 4, 2, 3, 5), sign_function=global_sign)
    # Ũ[h1, h6, v1] <-- Ũ[h1, v1, h6] = Ũa[h1, h3, h4, v2] * Ũb[h3, h4, v2, v1, h6]
    Ũ = contract(Ũa, Ũb, ((2, 3, 4), (1, 2, 3)); perm=(1, 3, 2), sign_function=global_sign)

    # C̃2a[h2, v1, h3] = C2[h2, v2] * R[v2, v1, h3]
    C̃2a = contract(C2, R, (2, 1); sign_function=global_sign)
    # C̃2[h1, v1] <-- C̃2[v1, h1] = C̃2a[h2, v1, h3] * Q2[h1, h2, h3]
    C̃2 = contract(C̃2a, Q2, ((1, 3), (2, 3)); perm=(2, 1), sign_function=global_sign)

    coefC1 = maximum(abs(C̃1))
    coefU = maximum(abs(Ũ))
    coefC2 = maximum(abs(C̃2))
    coef = coefC1 * coefU * coefC2

    C̃1 /= coefC1
    Ũ /= coefU
    C̃2 /= coefC2

    return C̃1, Ũ, C̃2, coef
end

function left_move!(
    Tbulk::Matrix{Grassmann{T, 4}}, 
    env::CTMRGEnv, 
    χ::Int; 
    average_trunc::Bool=true) where {T}

    Lx, Ly = size(Tbulk)

    coef = Matrix{Float64}(undef, Lx, Ly)
    trunc_err = Matrix{Float64}(undef, Lx, Ly)
    Λ = Matrix{GrassmannMatrix{Float64}}(undef, Lx, Ly)

    for y = 1:Ly
        rt = Nmod(y + 1, Ly)
        P = Vector{Grassmann{T, 3}}(undef, Lx)
        Q = Vector{Grassmann{T, 3}}(undef, Lx)
        for x = 1:Lx
            dn = Nmod(x + 1, Lx)
            P[dn], Q[dn], Λ[x, y], trunc_err[x, y] =  generate_projector_left(
                Tbulk[x, y], Tbulk[x, rt], Tbulk[dn, y], Tbulk[dn, rt],
                env.El[x, y], env.El[dn, y], env.Er[x, rt], env.Er[dn, rt],
                env.Eu[x, y], env.Eu[x, rt], env.Ed[dn, y], env.Ed[dn, rt],
                env.Clu[x, y], env.Cru[x, rt], env.Cld[dn, y], env.Crd[dn, rt], χ; 
                average_trunc=average_trunc
                )
        end
        for x = 1:Lx
            dn = Nmod(x + 1, Lx)
            env.Clu[x, rt], env.El[x, rt], env.Cld[x, rt], coef[x, y] = do_truncation_left(
                env.Eu[x, y], Tbulk[x, y], env.Ed[x, y], 
                env.Clu[x, y], env.El[x, y], env.Cld[x, y], 
                P[x], Q[x], P[dn], Q[dn]
                )
        end
    end

    return coef, trunc_err, Λ
end

"""
 C1 ←←←←←←←←←← U1 ←←←←←←←←←←←←←←←←←←←←←←←←←←←←← U2 ←←←←←←←←←← C2
 ↓             ↓                                ↓             ↑
 ↓             ↓                                ↓             ↑
 ↓             ↓                                ↓             ↑
 L1 ←←←←←←←←←← T1 ←←←←←←←←←←←←←←←←←←←←←←←←←←←←← T2 ←←←←←←←←←← R1
 ↓             ↓                                ↓             ↑
 ↓             ↓                                ↓             ↑
(v1p)        (v2p)                            (v3p)          (v4)
 *             *
 *             *
 *             *
(v1)          (v2)                             (v5)          (v6p)
 ↓             ↓                                ↓              ↑
 ↓             ↓                                ↓              ↑
 L2 ←←←←←←←←←← T3 ←←←←←←←←←←←←←←←←←←←←←←←←←←←←← T4 ←←←←←←←←←← R2
 ↓             ↓                                ↓              ↑
 ↓             ↓                                ↓              ↑
 ↓             ↓                                ↓              ↑
 ↓             ↓                                ↓              ↑
 C3 →→→→→→→→→→ D1 →→→→→→→→→→→→→→→→→→→→→→→→→→→→→ D2 →→→→→→→→→→ C4
"""

function generate_projector_left(
    T1::Grassmann{T, 4}, T2::Grassmann{T, 4}, T3::Grassmann{T, 4}, T4::Grassmann{T, 4}, 
    L1::Grassmann{T, 3}, L2::Grassmann{T, 3}, R1::Grassmann{T, 3}, R2::Grassmann{T, 3}, 
    U1::Grassmann{T, 3}, U2::Grassmann{T, 3}, D1::Grassmann{T, 3}, D2::Grassmann{T, 3}, 
    C1::Grassmann{T, 2}, C2::Grassmann{T, 2}, C3::Grassmann{T, 2}, C4::Grassmann{T, 2}, 
    χ::Int; 
    average_trunc::Bool=true) where {T}

    LU = generate_LU(C1, U1, L1, T1)
    RU = generate_RU(C2, R1, U2, T2)
    LD = generate_LD(L2, C3, D1, T3)
    RD = generate_RD(R2, C4, D2, T4)

    ULR = generate_ULR(LU, RU)
    DLR = generate_DLR(LD, RD)

    # ULR_perm[(v3p, v4), (v1p, v2p)] <-- ULR[(v1p, v2p), (v3p, v4)]
    ULR_perm = permutedims(ULR, (3, 4, 1, 2); sign_function=global_sign)
    # ULR_perm_t2[(v3p, v4), (v1, v2)] < -- ULR_perm[(v3p, v4), (v1p, v2p)]
    ULR_perm_t1 = index_conjugation(ULR_perm, 3)
    ULR_perm_t2 = index_conjugation(ULR_perm_t1, 4)

    # DLR_t1[(v1, v2), (v5, v6p)] = (DLR × (-1)^(v1 × v2))[(v1, v2), (v5, v6p)]
    DLR_t1 = add_perm_sign(DLR, (2, 1, 3, 4); sign_function=global_sign)
    # DLR_t3[(v1p, v2p), (v5, v6p)] = (DLR_t1 × (-1)^(v1 + v2))[(v1, v2), (v5, v6p)]
    DLR_t2 = add_parity_sign(index_conjugation(DLR_t1, 1), 1; sign_function=global_sign)
    DLR_t3 = add_parity_sign(index_conjugation(DLR_t2, 2), 2; sign_function=global_sign)

    # W[(v3p, v4), (v5, v6p)] = ULR_perm_t2[(v3p, v4), (v1, v2)] * DLR_t3[(v1p, v2p), (v5, v6p)]
    W = contract(ULR_perm_t2, DLR_t3, ((3, 4), (1, 2)))
    # W[(v3p, v4), (v5, v6p)] --> U[(v3p, v4), x], S[xp, y], Vdag[yp, (v5, v6p)] --> V[(v5p, v6), y]
    U, S, V, trunc_err = gsvd(W, (1, 2), (3, 4), χ; average_trunc=average_trunc)
    
    invsqrtS = inv(sqrt(S))

    # P1[(v1p, v2p), y] = DLR_t3[(v1p, v2p), (dum1, dum2)] * V[(dum1, dum2), y]
    P1 = contract(DLR_t3, V, ((3, 4), (1, 2)))
    # P2[(v1p, v2p), y] = P1[(v1p, v2p), dum] * invsqrtS[dum, y]
    P2 = contract(P1, invsqrtS, (3, 1))
    # P4[(v1, v2), y] = (P2 × (-1)^(v1p + v2p))[(v1p, v2p), y]
    P3 = add_parity_sign(index_conjugation(P2, 1), 1; sign_function=global_sign)
    P4 = add_parity_sign(index_conjugation(P3, 2), 2; sign_function=global_sign)
    # P5[(v1, v2), yp] <-- P4[(v1, v2), y]
    P5 = index_conjugation(P4, 3)
    # P[(v1, v2), yp] = (P5 × (-1)^(v1 × v2))[(v1, v2), yp]
    P = add_perm_sign(P5, (2, 1, 3); sign_function=global_sign)

    # Q1[xp, (v3, v4p)] = invsqrtS[xp, dum] * conj(U)[(v3, v4p), dum]
    Q1 = contract(invsqrtS, U, (2, 3); cj=(false, true))
    # Q2[xp, (v1, v2)] = Q1[xp, (dum1, dum2)] * ULR_perm_t2[(dum1, dum2), (v1, v2)]
    Q2 = contract(Q1, ULR_perm_t2, ((2, 3), (1, 2)))
    # Q3[x, (v1, v2)] = (Q2 × (-1)^xp)[xp, (v1, v2)]
    Q3 = add_parity_sign(index_conjugation(Q2, 1), 1; sign_function=global_sign)
    # Q[x, (v1p, v2p)] <-- Q3[x, (v1, v2)]
    Q4 = index_conjugation(Q3, 2)
    Q = index_conjugation(Q4, 3)

    return P, Q, S/maximum(S), trunc_err
end

"""
 C1  ←← (h2) ←← U ←←←←←← (h1)  
 ↓              ↓    
 ↓              ↓ 
(v2)           (v3)  
   ↘          ↙
     ↘      ↙
       ↘  ↙
        P1           
        ↓   
        ↓
       (v1) 
    
       (v1)
        ↓
        ↓
        Q1
     ↙     ↘
   ↙         ↘
(v2)         (v3)
 ↓            ↓
 ↓            ↓
 L ←← (h2) ←← T ←←←←←← (h1)
 ↓            ↓
 ↓            ↓
(v4)         (v5)
   ↘        ↙
     ↘    ↙
        P2
        ↓
        ↓
       (v6)

       (v1)
        ↓
        ↓
        Q2
     ↙     ↘
   ↙         ↘
 ↙             ↘
(v2)           (v3)
 ↓               ↓
 ↓               ↓
 C3 →→ (h2) →→→  D →→→→→→ (h1)  
"""

function do_truncation_left(
    U::Grassmann{T, 3}, Tb::Grassmann{T, 4}, D::Grassmann{T, 3}, 
    C1::Grassmann{T, 2}, L::Grassmann{T, 3}, C3::Grassmann{T, 2}, 
    P1::Grassmann{T, 3}, Q1::Grassmann{T, 3}, P2::Grassmann{T, 3}, 
    Q2::Grassmann{T, 3}) where {T}

    # C̃1a[v2, h1, v3] = C1[h2, v2] * U[h2, h1, v3]
    C̃1a = contract(C1, U, (1, 1); sign_function=global_sign)
    # C̃1[h1, v1] = C̃1a[v2, h1, v3] * P1[v2, v3, v1]
    C̃1 = contract(C̃1a, P1, ((1, 3), (1, 2)); sign_function=global_sign)

    # L̃1[v2, h2, v5, v6] = L[v2, v4, h2] * P2[v4, v5, v6]
    L̃1 = contract(L, P2, (2, 1); sign_function=global_sign)
    # L̃2[v1, v2, h2, h1, v5] = Q1[v1, v2, v3] * Tb[h2, h1, v3, v5]
    L̃2 = contract(Q1, Tb, (3, 3); sign_function=global_sign)
    # L̃[v1, v6, h1] <-- L̃[v6, v1, h1] = L̃1[v2, h2, v5, v6] * L̃2[v1, v2, h2, h1, v5] 
    L̃ = contract(L̃1, L̃2, ((1, 2, 3), (2, 3, 5)); perm=(2, 1, 3), sign_function=global_sign)

    # C̃3a[v2, h1, v3] = C3[h2, v2] * D[h2, h1, v3]
    C̃3a = contract(C3, D, (1, 1); sign_function=global_sign)
    # C̃3[h1, v1] = C̃3a[v2, h1, v3] * Q2[v1, v2, v3]
    C̃3 = contract(C̃3a, Q2, ((1, 3), (2, 3)); sign_function=global_sign)

    coefC1 = maximum(abs(C̃1))
    coefL = maximum(abs(L̃))
    coefC3 = maximum(abs(C̃3))
    coef = coefC1 * coefL * coefC3

    C̃1 /= coefC1
    L̃ /= coefL
    C̃3 /= coefC3

    return C̃1, L̃, C̃3, coef
end

function right_move!(
    Tbulk::Matrix{Grassmann{T, 4}}, 
    env::CTMRGEnv, 
    χ::Int; 
    average_trunc::Bool=true) where {T}

    Lx, Ly = size(Tbulk)

    coef = Matrix{Float64}(undef, Lx, Ly)
    trunc_err = Matrix{Float64}(undef, Lx, Ly)
    Λ = Matrix{GrassmannMatrix{Float64}}(undef, Lx, Ly)

    for y = Ly:-1:1
        left = Nmod(y - 1, Ly)
        P = Vector{Grassmann{T, 3}}(undef, Lx)
        Q = Vector{Grassmann{T, 3}}(undef, Lx)
        for x = 1:Lx
            dn = Nmod(x + 1, Lx)
            P[dn], Q[dn], Λ[x, y], trunc_err[x, y] = generate_projector_right(
                Tbulk[x, left], Tbulk[x, y], Tbulk[dn, left], Tbulk[dn, y], 
                env.El[x, left], env.El[dn, left], env.Er[x, y], env.Er[dn, y], 
                env.Eu[x, left], env.Eu[x, y], env.Ed[dn, left], env.Ed[dn, y], 
                env.Clu[x, left], env.Cru[x, y], env.Cld[dn, left], env.Crd[dn, y], χ; 
                average_trunc=average_trunc
                )
        end
        for x = 1:Lx
            dn = Nmod(x + 1, Lx)
            env.Cru[x, left], env.Er[x, left], env.Crd[x, left], coef[x, y] = do_truncation_right(
                env.Eu[x, y], Tbulk[x, y], env.Ed[x, y], 
                env.Cru[x, y], env.Er[x, y], env.Crd[x, y], 
                P[x], Q[x], P[dn], Q[dn]
                )
        end
    end

    return coef, trunc_err, Λ
end

"""
 C1 ←←←←←←←←←← U1 ←←←←←←←←←←←←←←←←←←←←←←←←←←←←← U2 ←←←←←←←←←← C2
 ↓             ↓                                ↓             ↑
 ↓             ↓                                ↓             ↑
 ↓             ↓                                ↓             ↑
 L1 ←←←←←←←←←← T1 ←←←←←←←←←←←←←←←←←←←←←←←←←←←←← T2 ←←←←←←←←←← R1
 ↓             ↓                                ↓             ↑
 ↓             ↓                                ↓             ↑
(v1p)        (v2p)                            (v5p)          (v6)
                                                *             *
                                                *             *
                                                *             *
(v3)          (v4)                             (v5)          (v6p)
 ↓             ↓                                ↓              ↑
 ↓             ↓                                ↓              ↑
 L2 ←←←←←←←←←← T3 ←←←←←←←←←←←←←←←←←←←←←←←←←←←←← T4 ←←←←←←←←←← R2
 ↓             ↓                                ↓              ↑
 ↓             ↓                                ↓              ↑
 ↓             ↓                                ↓              ↑
 ↓             ↓                                ↓              ↑
 C3 →→→→→→→→→→ D1 →→→→→→→→→→→→→→→→→→→→→→→→→→→→→ D2 →→→→→→→→→→ C4
"""

function generate_projector_right(
    T1::Grassmann{T, 4}, T2::Grassmann{T, 4}, T3::Grassmann{T, 4}, T4::Grassmann{T, 4}, 
    L1::Grassmann{T, 3}, L2::Grassmann{T, 3}, R1::Grassmann{T, 3}, R2::Grassmann{T, 3}, 
    U1::Grassmann{T, 3}, U2::Grassmann{T, 3}, D1::Grassmann{T, 3}, D2::Grassmann{T, 3}, 
    C1::Grassmann{T, 2}, C2::Grassmann{T, 2}, C3::Grassmann{T, 2}, C4::Grassmann{T, 2}, 
    χ::Int; 
    average_trunc::Bool=true) where {T}

    LU = generate_LU(C1, U1, L1, T1)
    RU = generate_RU(C2, R1, U2, T2)
    LD = generate_LD(L2, C3, D1, T3)
    RD = generate_RD(R2, C4, D2, T4)

    ULR = generate_ULR(LU, RU)
    DLR = generate_DLR(LD, RD)

    # DLR_perm[(v5, v6p), (v3, v4)] <-- DLR[(v3, v4), (v5, v6p)]
    DLR_perm = permutedims(DLR, (3, 4, 1, 2); sign_function=global_sign)
    # DLR_perm_t1[(v5, v6p), (v3, v4)] = (DLR_perm × (-1)^(v5 × v6p))[(v5, v6p), (v3, v4)]
    DLR_perm_t1 = add_perm_sign(DLR_perm, (2, 1, 3, 4); sign_function=global_sign)
    # DLR_perm_t2[(v5p, v6p), (v3, v4)] = (DLR_perm_t1 × (-1)^v5)[(v5, v6p), (v3, v4)]
    DLR_perm_t2 = add_parity_sign(index_conjugation(DLR_perm_t1, 1), 1; sign_function=global_sign)
    
    # ULR_t[(v1p, v2p), (v5, v6)] <-- ULR[(v1p, v2p), (v5p, v6)]
    ULR_t = index_conjugation(ULR, 3)

    # W[(v1p, v2p), (v3, v4)] = ULR_t[(v1p, v2p), (dum1, dum2)] * DLR_perm_t2[(dum1, dum2), (v3, v4)]
    W = contract(ULR_t, DLR_perm_t2, ((3, 4), (1, 2)))
    # W[(v1p, v2p), (v3, v4)] --> U[(v1p, v2p), x], S[xp, y], Vdag[yp, (v3, v4)] --> V[(v3p, v4p), y]
    U, S, V, trunc_err = gsvd(W, (1, 2), (3, 4), χ; average_trunc=average_trunc)

    invsqrtS = inv(sqrt(S))

    # P1[(v5p, v6p), y] = DLR_perm_t2[(v5p, v6p), (dum1, dum2)] * V[(dum1, dum2), y]
    P1 = contract(DLR_perm_t2, V, ((3, 4), (1, 2)))
    # P2[(v5p, v6p), y] = P1[(v5p, v6p), dum] * invsqrtS[dum, y]
    P2 = contract(P1, invsqrtS, (3, 1))
    # P3[(v5p, v6p), y] = (P2 × (-1)^(v5p × v6p))[(v5p, v6p), y]
    P3 = add_perm_sign(P2, (2, 1, 3); sign_function=global_sign)
    # P[(v5, v6p), y] = (P3 × (-1)^v5)[(v5p, v6p), y]
    P = add_parity_sign(index_conjugation(P3, 1), 1; sign_function=global_sign)

    # Q1[xp, (v1, v2)] = invsqrtS[xp, dum] * conj(U)[(v1, v2), dum]
    Q1 = contract(invsqrtS, U, (2, 3); cj=(false, true))
    # Q2[xp, (v5, v6)] = Q1[xp, (dum1, dum2)] * ULR_t[(dum1, dum2), (v5, v6)]
    Q2 = contract(Q1, ULR_t, ((2, 3), (1, 2)))
    # Q2[xp, (v5p, v6)] <-- Q2[xp, (v5, v6)]
    Q = index_conjugation(Q2, 2)

    return P, Q, S/maximum(S), trunc_err
end

"""
(h1) ←←←← U  ←← (h2) ←← C2  
          ↓             ↑     
          ↓             ↑
         (v2)          (v3)  
             ↘       ↙
               ↘   ↙
                 P1           
                 ↑  
                 ↑ 
                (v1)
            
                (v1)
                 ↑
                 ↑
                 Q1
              ↙     ↖
            ↙         ↖
         (v2)         (v3)
           ↓            ↑
           ↓            ↑
(h1) ←←←←  T ←← (h2) ←← R 
           ↓            ↑
           ↓            ↑
          (v4)         (v5)
             ↘        ↗
               ↘    ↗
                 P2
                 ↑
                 ↑
                (v6)

                (v1)
                 ↑
                 ↑
                 Q2
                ↗  ↖
              ↗      ↖
            ↗          ↖
         (v2)           (v3)
          ↓              ↑
          ↓              ↑
(h1) →→→→ D  →→ (h2) →→ C4
"""

function do_truncation_right(
    U::Grassmann{T, 3}, Tb::Grassmann{T, 4}, D::Grassmann{T, 3}, 
    C2::Grassmann{T, 2}, R::Grassmann{T, 3}, C4::Grassmann{T, 2}, 
    P1::Grassmann{T, 3}, Q1::Grassmann{T, 3}, 
    P2::Grassmann{T, 3}, Q2::Grassmann{T, 3}) where {T}

    # C̃2a[h1, v2, v3] <-- C̃2a[v3, h1, v2] = C2[h2, v3] * U[h1, h2, v2]
    C̃2a = contract(C2, U, (1, 2); perm=(2, 3, 1), sign_function=global_sign)
    # C̃2[h1, v1] = C̃2a[h1, v2, v3] * P1[v2, v3, v1]
    C̃2 = contract(C̃2a, P1, ((2, 3), (1, 2)); sign_function=global_sign)

    # R̃a[v1, v3, h1, h2, v4] = Q1[v1, v2, v3] * Tb[h1, h2, v2, v4]
    R̃a = contract(Q1, Tb, (2, 3); sign_function=global_sign)
    # R̃b[v3, h2, v4, v6] = R[v3, v5, h2] * P2[v4, v5, v6]
    R̃b = contract(R, P2, (2, 2); sign_function=global_sign)
    # R̃[v1, v6, h1] <-- R̃[v1, h1, v6] = R̃a[v1, v3, h1, h2, v4] * R̃b[v3, h2, v4, v6]
    R̃ = contract(R̃a, R̃b, ((2, 4, 5), (1, 2, 3)); perm=(1, 3, 2), sign_function=global_sign)

    # C̃4a[h1, v2, v3] <-- C̃4a[v3, h1, v2] = C4[h2, v3] * D[h1, h2, v2]
    C̃4a = contract(C4, D, (1, 2); perm=(2, 3, 1), sign_function=global_sign)
    # C̃4[h1, v1] = C̃4a[h1, v2, v3] * Q2[v1, v2, v3]
    C̃4 = contract(C̃4a, Q2, ((2, 3), (2, 3)); sign_function=global_sign)

    coefC2 = maximum(abs(C̃2))
    coefR = maximum(abs(R̃))
    coefC4 = maximum(abs(C̃4))
    coef = coefC2 * coefR * coefC4

    C̃2 /= coefC2
    R̃ /= coefR
    C̃4 /= coefC4

    return C̃2, R̃, C̃4, coef
end

############################## Generate useful tensors #############################

"""
 C1 ←← (h3) ←← U1 ←← (h4)   
 ↓             ↓                      
 ↓             ↓                      
(v3)          (v4)               
 ↓             ↓                 
 ↓             ↓                      
 L1 ←← (h1) ←← T1 ←← (h2)   
 ↓             ↓                        
 ↓             ↓                 
(v1)          (v2)            

Time cost : O(χ²D⁴) ~ O(d¹²) for D = d² and χ = d²
Memory cost : O(χ²D²) ~ O(d⁸)
"""

function generate_LU(C1::Grassmann{T, 2}, U1::Grassmann{T, 3}, L1::Grassmann{T, 3}, T1::Grassmann{T, 4}) where {T}

    # LU1[v3, h4, v4] = C1[h3, v3] * U1[h3, h4, v4]
    # Time : χ³D, Mem : χ²D
    LU1 = contract(C1, U1, (1, 1); sign_function=global_sign)
    # LU2[h4, v1, h1, v4] <-- LU2[h4, v4, v1, h1] = LU1[v3, h4, v4] * L1[v3, v1, h1]
    # Time :  χ³D², Mem : χ²D²
    LU2 = contract(LU1, L1, (1, 1); perm=(1, 3, 4, 2), sign_function=global_sign)
    # LU[h4, h2, v1, v2] <-- LU[h4, v1, h2, v2] = LU2[h4, v1, h1, v4] * T1[h1, h2, v4, v2]
    # Time : χ²D⁴, Mem : χ²D²
    LU = contract(LU2, T1, ((3, 4), (1, 3)); perm=(1, 3, 2, 4), sign_function=global_sign)
end

"""
(v1)          (v2)                
 ↓             ↓                      
 ↓             ↓                         
 L2 ←← (h1) ←← T3 ←← (h3)   
 ↓             ↓                   
 ↓             ↓          
(v3)          (v4)      
 ↓             ↓                       
 ↓             ↓                        
C3 →→ (h2) →→  D1 →→ (h4)   
"""

function generate_LD(L2::Grassmann{T, 3}, C3::Grassmann{T, 2}, D1::Grassmann{T, 3}, T3::Grassmann{T, 4}) where {T}

    # LD1[v1, h1, h2] = L2[v1, v3, h1] * C3[h2, v3]
    LD1 = contract(L2, C3, (2, 2); sign_function=global_sign)
    # LD2[v1, h1, h4, v4] = LD1[v1, h1, h2] * D1[h2, h4, v4]
    LD2 = contract(LD1, D1, (3, 1); sign_function=global_sign)
    # LD[h3, h4, v1, v2] <-- LD[v1, h4, h3, v2] = LD2[v1, h1, h4, v4] * T3[h1, h3, v2, v4]
    LD = contract(LD2, T3, ((2, 4), (1, 4)); perm=(3, 2, 1, 4), sign_function=global_sign)
end

"""
 (h1) ←← U2 ←← (h3) ←← C2
         ↓             ↑          
         ↓             ↑     
        (v3)          (v4)
         ↓             ↑
         ↓             ↑
 (h2) ←← T2 ←← (h4) ←← R1
         ↓             ↑      
         ↓             ↑                      
        (v1)          (v2)    
"""

function generate_RU(C2::Grassmann{T, 2}, R1::Grassmann{T, 3}, U2::Grassmann{T, 3}, T2::Grassmann{T, 4}) where {T}

    # RU1[h3, v2, h4] = C2[h3, v4] * R1[v4, v2, h4]
    RU1 = contract(C2, R1, (2, 1); sign_function=global_sign)
    # RU2[v2, h4, h1, v3] = RU1[h3, v2, h4] * U2[h1, h3, v3]
    RU2 = contract(RU1, U2, (1, 2); sign_function=global_sign)
    # RU[h1, h2, v1, v2] <-- RU[v2, h1, h2, v1] = RU2[v2, h4, h1, v3] * T2[h2, h4, v3, v1]
    RU = contract(RU2, T2, ((2, 4), (2, 3)); perm=(2, 3, 4, 1), sign_function=global_sign)
end

"""
        (v1)          (v2)
         ↓             ↑                      
         ↓             ↑                          
 (h1) ←← T4 ←← (h3) ←← R2
         ↓             ↑                     
         ↓             ↑                    
        (v3)          (v4)  
         ↓             ↑                        
         ↓             ↑    
 (h2) →→ D2 →→ (h4) →→ C4
"""

function generate_RD(R2::Grassmann{T, 3}, C4::Grassmann{T, 2}, D2::Grassmann{T, 3}, T4::Grassmann{T, 4}) where {T}

    # RD1[v2, h3, h4] = R2[v2, v4, h3] * C4[h4, v4]
    RD1 = contract(R2, C4, (2, 2); sign_function=global_sign)
    # RD2[v2, h3, h2, v3] = RD1[v2, h3, h4] * D2[h2, h4, v3]
    RD2 = contract(RD1, D2, (3, 2); sign_function=global_sign)
    # RD[h1, h2, v1, v2] <-- RD[v2, h2, h1, v1] = RD2[v2, h3, h2, v3] * T4[h1, h3, v1, v3]
    RD = contract(RD2, T4, ((2, 4), (2, 4)); perm=(3, 2, 4, 1), sign_function=global_sign)
end

"""
 C1 ←←←←←←←←←← U1 ←←←←←←←←← (h1)  
 ↓             ↓           
 ↓             ↓                               
 ↓             ↓                
 ↓             ↓                     
 L1 ←←←←←←←←←← T1 ←←←←←←←←← (h2)  
 ↓             ↓                 
 ↓             ↓ 
 ↓             ↓              
(v1)          (v2)
 ↓             ↓                                                                                      
 ↓             ↓                  
 ↓             ↓               
 L2 ←←←←←←←←←← T3 ←←←←←←←←← (h3)  
 ↓             ↓                    
 ↓             ↓                                   
 ↓             ↓                     
 ↓             ↓              
C3 →→→→→→→→→→→ D1 →→→→→→→→→ (h4)   

Time cost : O(χ³D³) ~ O(d¹²) for D = d² and χ = d²
Memory cost : O(χ²D²) ~ O(d⁸)
"""

function generate_LUD(LU::Grassmann{T, 4}, LD::Grassmann{T, 4}) where {T}

    # LUD[(h1, h2), (h3, h4)] = LU[(h1, h2), (v1, v2)] * LD[(h3, h4), (v1, v2)]
    # Time : χ³D³, Mem : χ²D²
    LUD = contract(LU, LD, ((3, 4), (3, 4)); sign_function=global_sign)
    LUD /= maximum(abs(LUD))
end

"""
 (h1) ←←←←←←←←←← U2 ←←←←←←←←←← C2
                 ↓             ↑          
                 ↓             ↑     
                 ↓             ↑
                 ↓             ↑
 (h2) ←←←←←←←←←← T2 ←←←←←←←←←← R1
                 ↓             ↑      
                 ↓             ↑                      
                (v1)          (v2)    
                 ↓             ↑                      
                 ↓             ↑                          
 (h3) ←←←←←←←←←← T4 ←←←←←←←←←← R2
                 ↓             ↑                     
                 ↓             ↑                    
                 ↓             ↑                        
                 ↓             ↑    
 (h4) →→→→→→→→→→ D2 →→→→→→→→→→ C4
"""

function generate_RUD(RU::Grassmann{T, 4}, RD::Grassmann{T, 4}) where {T}

    # RUD[(h1, h2), (h3, h4)] = RU[(h1, h2), (v1, v2)] * RD[(h3, h4), (v1, v2)]
    RUD = contract(RU, RD, ((3, 4), (3, 4)); sign_function=global_sign)
    RUD /= maximum(abs(RUD))
end

"""
 C1 ←←←←←←←←←← U1 ←←←← (h4) ←←←← U2 ←←←←←←←←←← C2
 ↓             ↓                 ↓             ↑
 ↓             ↓                 ↓             ↑
 ↓             ↓                 ↓             ↑
 ↓             ↓                 ↓             ↑
 L1 ←←←←←←←←←← T1 ←←←← (h3) ←←←← T2 ←←←←←←←←←← R1
 ↓             ↓                 ↓             ↑
 ↓             ↓                 ↓             ↑
 ↓             ↓                 ↓             ↑
 ↓             ↓                 ↓             ↑
(v1)         (v2)               (v3)          (v4)
"""

function generate_ULR(LU::Grassmann{T, 4}, RU::Grassmann{T, 4}) where {T}

    # ULR[(v1, v2), (v3, v4)] = LU[(h4, h3), (v1, v2)] * RU[(h4, h3), (v3, v4)]
    ULR = contract(LU, RU, ((1, 2), (1, 2)); sign_function=global_sign)
    ULR /= maximum(abs(ULR))
end

"""
(v1)          (v2)              (v3)          (v4)
 ↓             ↓                 ↓             ↑
 ↓             ↓                 ↓             ↑
 ↓             ↓                 ↓             ↑
 ↓             ↓                 ↓             ↑
 L2 ←←←←←←←←←← T3 ←←←← (h1) ←←←← T4 ←←←←←←←←←← R2
 ↓             ↓                 ↓             ↑
 ↓             ↓                 ↓             ↑
 ↓             ↓                 ↓             ↑
 ↓             ↓                 ↓             ↑
C3 →→→→→→→→→→→ D1 →→→→ (h2) →→→→ D2 →→→→→→→→→→ C4
"""

function generate_DLR(LD::Grassmann{T, 4}, RD::Grassmann{T, 4}) where {T}

    # DLR[(v1, v2), (v3, v4)] = LD[(h1, h2), (v1, v2)] * RD[(h1, h2), (v3, v4)]
    DLR = contract(LD, RD, ((1, 2), (1, 2)); sign_function=global_sign)
    DLR /= maximum(abs(DLR))
end

####################################### auxiliary functions #######################################

function prepare_Λ(Lx::Int, Ly::Int, χ::Int)

    χe = div(χ, 2)

    Λd = Matrix{GrassmannMatrix{Float64}}(undef, Lx, Ly)
    Λu = Matrix{GrassmannMatrix{Float64}}(undef, Lx, Ly)
    Λl = Matrix{GrassmannMatrix{Float64}}(undef, Lx, Ly)
    Λr = Matrix{GrassmannMatrix{Float64}}(undef, Lx, Ly)
    for r in 1:Lx, c in 1:Ly
        Λd[r, c] = Grassmann(prepare_bond_weight(χ, χe), (χ, χ), (χe, χe), (:out, :in))
        Λu[r, c] = Grassmann(prepare_bond_weight(χ, χe), (χ, χ), (χe, χe), (:out, :in))
        Λl[r, c] = Grassmann(prepare_bond_weight(χ, χe), (χ, χ), (χe, χe), (:out, :in))
        Λr[r, c] = Grassmann(prepare_bond_weight(χ, χe), (χ, χ), (χe, χe), (:out, :in))
    end

    return Λd, Λu, Λl, Λr
end

# Find the lastest environment tensor given χ
function read_CTMRG_env(env_file::String, χ::Int)

    prefix = "χ$(χ)iter"

    h5open(env_file*".h5", "r") do fid
        iter_vec = Int[]

        for key in keys(fid)
            key_str = String(key)
            startswith(key_str, prefix) || continue
            iter_str = key_str[nextind(key_str, lastindex(prefix)):end]
            push!(iter_vec, parse(Int, iter_str))
        end

        isempty(iter_vec) && return "random"

        return "χ$χ"*"iter$(maximum(iter_vec))"
    end
end