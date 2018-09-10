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
            Vs[j, 0, N_tree_full - level - 1], Ts[j, 0, N_tree_full - level - 1], Rs[j, 0, N_tree_full - level - 1] = qr_factor(Rs[j, 0, N_tree_full - level], Rs[j + 2**(level), 0, N_tree_full - level])

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
                Vs[j, i, N_tree - level - 1], Ts[j, i, N_tree - level - 1], Rs[j, i, N_tree - level - 1] = qr_factor(Rs[j, i, N_tree - level], Rs[j + 2**(level), i, N_tree - level])
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




