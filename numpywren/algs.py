from numpywren.matrix import BigMatrix 

def SimpleTestLinear(A:BigMatrix, B:BigMatrix, N:int):
    for i in range(N):
        for j in range(i+1, N):
            A[j, i] = identity(A[i,j])

    for z in range(N):
        for k in range(N):
            B[z,k] = identity(A[z,k])

def SimpleTestLinear2(A:BigMatrix, B:BigMatrix, N:int):
    for i in range(N):
        for j in range(i+1, N):
            A[j+1, i+j] = identity(A[i,j])

    for z in range(N):
        for k in range(N):
            B[z,k] = identity(A[z,k])

def SimpleTestNonLinear(A:BigMatrix, B: BigMatrix, N:int):
    for i in range(N):
        N_tree = ceiling(log(N - i)/log(2))
        for level in range(0,ceiling(log(N - i)/log(2))):
            for k in range(0, N, 2**(level+1)):
                A[N_tree - level - 1, i, k] = add(A[N_tree - level, i, k], A[N_tree - level, i, k + 2**(level)])

        B[i] = identity(A[1, i, 0])

def TSQR(A:BigMatrix, Vs:BigMatrix, Ts:BigMatrix, Rs:BigMatrix, N:int):
    for j in range(0, N):
        Vs[0, j], Ts[0, j], Rs[0, j] = qr_factor(A[j, 0])

    for level in range(0, ceiling(log(N)/log(2))):
        for j in range(0, N, 2**(level + 1)):
            Vs[level+1, j], Ts[level+1, j], Rs[level+1, j] = qr_factor(Rs[level, j], Rs[level, j + 2**(level)])

def BDFAC(I:BigMatrix, V_QR:BigMatrix, T_QR:BigMatrix, S_QR:BigMatrix, R_QR:BigMatrix, V_LQ:BigMatrix, T_LQ:BigMatrix, S_LQ:BigMatrix, L_LQ:BigMatrix, N:int, truncate:int):
    b_fac = 2
    # I is the input matrix of dimension N * N.
    #
    # Dimensions for T_QR, V_QR, V_LQ, T_LQ, R_QR and L_LQ are N * N_tree_full * N
    # Each array contains all outputs of all column and row updates at each recursion level.
    # The first index represents which compute stage we are at i.e. how many times we've updated
    # the trailing matrix and done a block column/row update.
    # The second index represents how deep in the TSQR/LU recursion we are.
    # The third index represents the index of the block under consideration at the current recursion
    # for QR it is the block row and for LQ it is the block column.
    # 
    # Dimensions for S_QR and S_LQ are N * N_tree_full * N * N
    # S_QR and S_LQ are the intermediate representations of our matrix after each low rank update.
    # S_QR contains all matrices after a QR low rank update, while S_LQ contains all matrices after an LQ
    # low rank update.
    # The first index represents which compute stage we are at i.e. how many times we've updated
    # the trailing matrix and done a block column/row update.
    # The second index represent the trailing update happening after the recursion depth of the given index.
    # The third and fourth indices are the indices of the matrix at that stage.

    # We take the convention that i is the computation stage, j is an index into rows,
    # k is an index into columns, level is a recursion depth.

    #
    # Step 0: QR column update + trailing matrix updates.
    # 

    # TODO: add an output matrix right now the output is sharded across S_QR and S_LU.
    N_tree_QR_full = ceiling(log(N)/log(2))
    # Do the first recursion step of the column update + corresponding trailing matrix update.
    for j in range(0, N):
        # Perform the QR, reading from the input matrix.
#0
        V_QR[0, 0, j], T_QR[0, 0, j], R_QR[0, 0, j] = qr_factor(I[j, 0])
        # Update the trailing row.
        for k in range(1, N):
#1
            S_QR[0, 0, j, k] = qr_leaf(V_QR[0, 0, j], T_QR[0, 0, j], I[j, k])

    # Now run the rest of the recursion for the column + trailing matrix updates.
    for level in range(1, N_tree_QR_full + 1):
        for j in range(0, N, 2 ** level):
            # Perform the QR, reading from the previous recursion depth.
#2
            V_QR[0, level, j], T_QR[0, level, j], R_QR[0, level, j] = qr_factor(R_QR[0, level - 1, j], R_QR[0, level - 1, j + 2 ** (level - 1)])
            # Update the trailing row.
            for k in range(1, N):
#3
                S_QR[0, level, j, k], S_QR[0, N_tree_QR_full, j + 2 ** (level - 1), k] = qr_trailing_update(V_QR[0, level, j], T_QR[0, level, j], S_QR[0, level - 1, j, k], S_QR[0, level - 1, j + 2 ** (level - 1), k])

    #
    # Step 0: LQ row update + trailing matrix updates.
    # 

    N_tree_LQ_full = ceiling(log(N - 1)/log(2))
    # Do the first recursion step of the row update + corresponding trailing matrix update.
    for k in range(1, N):
#4
        # Perform the LQ, reading from the final in the previous QR.
        V_LQ[0, 0, k], T_LQ[0, 0, k], L_LQ[0, 0, k] = lq_factor(S_QR[0, N_tree_QR_full, 0, k])
        # Update the trailing column.
        for j in range(1, N):
#5
            # TODO: I'm not sure this is correct. Need to think about transposes.
            S_LQ[0, 0, j, k] = lq_leaf(V_LQ[0, 0, k], T_LQ[0, 0, k], S_QR[0, N_tree_QR_full, j, k])

    # Now run the rest of the recursion for the row + trailing matrix updates.
    for level in range(1, N_tree_LQ_full + 1):
        for k in range(1, N, 2 ** level):
#6
            V_LQ[0, level, k], T_LQ[0, level, k], L_LQ[0, level, k] = lq_factor(L_LQ[0, level - 1, k], L_LQ[0, level - 1, k + 2 ** (level - 1)])
            # Update the trailing column.
            for j in range(1, N):
                # TODO: I'm not sure this is correct. Need to think about transposes.
#7
                S_LQ[0, level, j, k], S_LQ[0, N_tree_LQ_full, j, k + 2 ** (level - 1)] = lq_trailing_update(V_LQ[0, level, k], T_LQ[0, level, k], S_LQ[0, level - 1, j, k], S_LQ[0, level - 1, j, k + 2 ** (level - 1)])

    #
    # Run the rest of the steps except for the last step.
    # 
    for i in range(1, N - 1 - truncate):
        #
        # Step i: QR column update + trailing matrix updates.
        # 

        N_tree_QR = ceiling(log(N - i) / log(2))
        prev_N_tree_LQ = ceiling(log(N - i) / log(2))
        # Do the first recursion step of the column update + corresponding trailing matrix update.
        for j in range(i, N):
            # Perform the QR, reading from the final matrix in the previous LQ.
#8
            V_QR[i, 0, j], T_QR[i, 0, j], R_QR[i, 0, j] = qr_factor(S_LQ[i - 1, prev_N_tree_LQ, j, i])
            # Update the trailing row.
            for k in range(i + 1, N):
#9
                S_QR[i, 0, j, k] = qr_leaf(V_QR[i, 0, j], T_QR[i, 0, j], S_LQ[i - 1, prev_N_tree_LQ, j, k])

        # Now run the rest of the recursion for the column + trailing matrix updates.
        for level in range(1, N_tree_QR + 1):
            # TODO: double check we should be starting from i here.
            for j in range(i, N, 2 ** level):
                # Perform the QR, reading from the previous recursion depth.
#10
                V_QR[i, level, j], T_QR[i, level, j], R_QR[i, level, j] = qr_factor(R_QR[i, level - 1, j], R_QR[i, level - 1, j + 2 ** (level - 1)])
                # Update the trailing row.
                for k in range(i + 1, N):
#11
                    S_QR[i, level, j, k], S_QR[i, N_tree_QR, j + 2 ** (level - 1), k] = qr_trailing_update(V_QR[i, level, j], T_QR[i, level, j], S_QR[i, level - 1, j, k], S_QR[i, level - 1, j + 2 ** (level - 1), k])

        #
        # Step i: LQ row update + trailing matrix updates.
        #

        N_tree_LQ = ceiling(log(N - i - 1)/log(2))
        # Do the first recursion step of the row update + corresponding trailing matrix update.
        for k in range(i + 1, N):
#12
            # Perform the LQ, reading from the final matrix in the previous QR.
            V_LQ[i, 0, k], T_LQ[i, 0, k], L_LQ[i, 0, k] = lq_factor(S_QR[i, N_tree_QR, i, k])
            # Update the trailing column.
            for j in range(i + 1, N):
#13
                # TODO: I'm not sure this is correct. Need to think about transposes.
                S_LQ[i, 0, j, k] = lq_leaf(V_LQ[i, 0, k], T_LQ[i, 0, k], S_QR[i, N_tree_QR, j, k])

        # Now run the rest of the recursion for the first row + trailing matrix updates.
        for level in range(1, N_tree_LQ + 1):
            # TODO: double check we should be starting from i + 1 here.
            for k in range(i + 1, N, 2 ** level):
#14
                V_LQ[i, level, k], T_LQ[i, level, k], L_LQ[i, level, k] = lq_factor(L_LQ[i, level - 1, k], L_LQ[i, level - 1, k + 2 ** (level - 1)])
                # Update the trailing column.
                for j in range(i + 1, N):
                    # TODO: I'm not sure this is correct. Need to think about transposes.
#15
                    S_LQ[i, level, j, k], S_LQ[i, N_tree_LQ, j, k + 2 ** (level - 1)] = lq_trailing_update(V_LQ[i, level, k], T_LQ[i, level, k], S_LQ[i, level - 1, j, k], S_LQ[i, level - 1, j, k + 2 ** (level - 1)])

    #
    # Step N - 1: Final QR update. We don't run an LQ.
    #
    V_QR[N - 1, 0, N - 1], T_QR[N - 1, 0, N - 1], R_QR[N - 1, 0, N - 1] = qr_factor(S_LQ[N - 2, 0, N - 1, N - 1])


def QR(I:BigMatrix, Vs:BigMatrix, Ts:BigMatrix, Rs:BigMatrix, S:BigMatrix, N:int, truncate:int):
    b_fac = 2
    # starting code
    N_tree_full = ceiling(log(N)/log(2))
    for j in range(0, N):
        #0
        Vs[j, 0, N_tree_full], Ts[j, 0, N_tree_full], Rs[j, 0, N_tree_full] = qr_factor(I[j, 0])
    for level in range(0, N_tree_full):
        for j in range(0, N, 2**(level + 1)):
            #1
            Vs[j, 0, N_tree_full - level - 1], Ts[j, 0, N_tree_full - level - 1], Rs[j, 0, N_tree_full - level - 1] = qr_factor_triangular(Rs[j, 0, N_tree_full - level], Rs[j + 2**(level), 0, N_tree_full - level])

    # flat trailing matrix update
    for j in range(0, N):
        for k in range(1, N):
            #2
            S[j, k, 1, N_tree_full] = qr_leaf(Vs[j, 0, N_tree_full], Ts[j, 0, N_tree_full], I[j,k])


    for k in range(1, N):
        for level in range(0, N_tree_full):
            for j in range(0, N, 2**(level + 1)):
                #3
                S[j, k, 1, N_tree_full - 1 - level], S[j + 2**level, k, 1, 0]  = qr_trailing_update(Vs[j, 0, N_tree_full - 1 - level], Ts[j, 0, N_tree_full - 1 - level], S[j, k, 1, N_tree_full - level], S[j + 2**level, k, 1, N_tree_full - level])

    for k in range(1, N):
        #4
        Rs[0, k, 0]  = identity(S[0, k, 1, 0])
    # rest
    for i in range(1, N):
        N_tree = ceiling(log(N - i)/log(2))
        for j in range(i, N):
            #5
            Vs[j, i, N_tree], Ts[j, i, N_tree], Rs[j, i, N_tree] = qr_factor(S[j, i, i, 0])
        for level in range(0, N_tree):
            for j in range(i, N, 2**(level + 1)):
                #6
                Vs[j, i, N_tree - level - 1], Ts[j, i, N_tree - level - 1], Rs[j, i, N_tree - level - 1] = qr_factor_triangular(Rs[j, i, N_tree - level], Rs[j + 2**(level), i, N_tree - level])
        # flat trailing matrix update
        for j in range(i, N):
            for k in range(i+1, N):
                #7
                S[j, k, i+1, N_tree] = qr_leaf(Vs[j, i, N_tree], Ts[j, i, N_tree], S[j, k, i, 0])

        for k in range(i+1, N):
            for level in range(0, N_tree):
                for j in range(i, N, 2**(level + 1)):
                    #8
                    S[j, k, i+1, N_tree - 1 - level], S[j + 2**level, k, i+1, 0]  = qr_trailing_update(Vs[j, i, N_tree - 1 - level], Ts[j, i, N_tree - 1 - level], S[j, k, i+1, N_tree - level], S[j + 2**level, k, i +1, N_tree - level])

        for k in range(i+1, N):
            #9
            Rs[i, k, 0]  = identity(S[i, k, i+1, 0])

def CHOLESKY(O:BigMatrix, I:BigMatrix, S:BigMatrix,  N:int, truncate:int):
    # handle first loop differently
    O[0,0] = chol(I[0,0])
    for j in range(1,N - truncate):
        O[j,0] = trsm(O[0,0], I[j,0])
        for k in range(1,j+1):
            S[1,j,k] = syrk(I[j,k], O[j,0], O[k,0])

    for i in range(1,N - truncate):
        O[i,i] = chol(S[i,i,i])
        for j in range(i+1,N - truncate):
            O[j,i] = trsm(O[i,i], S[i,j,i])
            for k in range(i+1,j+1):
                S[i+1,j,k] = syrk(S[i,j,k], O[j,i], O[k,i])

def GEMM(A:BigMatrix, B:BigMatrix, M:int, N:int, K:int, Temp:BigMatrix, Out:BigMatrix):
    tree_depth = ceiling(log(K)/log(4))
    for i in range(0, M):
        for j in range(0, N):
            for k in range(0, K):
                Temp[i, j, k, 0] = gemm(A[i, k], B[k, j])

    for i in range(0, M):
        for j in range(0, N):
            for level in range(0, tree_depth):
                for k in range(0,K,4**(level+1)):
                    Temp[i, j, k, level+1] = add_matrices(Temp[i, j, k, level], Temp[i, j, k + 4**(level), level], Temp[i, j, k + 2 * 4**(level), level], Temp[i, j, k + 3 * 4**(level), level])

    for  i in range(0, M):
        for j in range(0, N):
            Out[i, j] = identity(Temp[i, j, 0, tree_depth])




