from numpy import matrix, zeros, random, finfo, argmax, array, identity, diag, roots, linalg, float32


def sortSecond(val):
    return val[1]

# MATRIX MULTIPLICATION ________________________________________________________________________________________________

def prod(A, B):
    if A.shape == B.shape:
        C = 0
        if A.shape[0] == 1:
            for k in xrange(A.shape[1]):
                C +=  A[0, k] * B[0, k]
            return C
        elif A.shape[1] == 1:
            for k in xrange(A.shape[0]):
                C +=  A[k, 0] * B[k, 0]
        return C
    raise Exception("MatrixShapeError in function prod")


def mult(A, B):
    if A.shape[1] != B.shape[0]:
        raise Exception("MatrixShapeError in function mult")

    C = zeros((A.shape[0], B.shape[1]))
    for l in xrange(A.shape[0]):
        for c in xrange(B.shape[1]):
            for k in xrange(A.shape[1]):
                C[l, c] += A[l, k] * B[k, c]
    return C


def getblock(M, lin, col, blocksize):
    n, m = M.shape

    linebegin = lin * blocksize
    lineend = min(n, (lin + 1) * blocksize)
    colbegin = col * blocksize
    colend = min(m, (col + 1) * blocksize)

    if linebegin >= n or colbegin >= m:
        raise Exception("MatrixShapeError in function gebblock")

    return M[linebegin:lineend, colbegin:colend]

def multblock(A, B, blocksize):
    if A.shape[1] != B.shape[0]:
        raise Exception("MatrixShapeError in function mult")

    n = A.shape[0] / blocksize + (A.shape[0] % blocksize > 0)
    m = A.shape[1] / blocksize + (A.shape[1] % blocksize > 0)
    p = B.shape[1] / blocksize + (B.shape[1] % blocksize > 0)

    C = zeros((A.shape[0], B.shape[1]))
    for l in xrange(n):
        for c in xrange(p):
            for k in xrange(m):
                block = getblock(C, l, c, blocksize)
                block += mult(getblock(A, l, k, blocksize), getblock(B, k, c, blocksize))
    return C


# PIVOTING _____________________________________________________________________________________________________________

def partial_pivoting(M, N, i, row_order):
    if abs(M[i, i]) > finfo(float).eps:
        return

    # buscando um elemento maior que M[i,i]
    max_idx = i
    for k in xrange(i+1, M.shape[0]):
        if abs(M[k, i]) > abs(M[max_idx, i]):
            max_idx = k

    # trocando as linhas se existe um elemento maior
    if max_idx != i:
        M[[i, max_idx], :] = M[[max_idx, i], :]

        if N != None:
            N[[i, max_idx], :] = N[[max_idx, i], :]

        if row_order != None:
            row_order[[i, max_idx]] = row_order[[max_idx, i]]


def total_pivoting(M, N, i, col_order=None):
    if abs(M[i, i]) > finfo(float).eps:
        return

    # procurando indice do maior elemento
    max_idx = (i, i)
    for l in xrange(i, M.shape[0]):
        for c in xrange(i+1, M.shape[1]):
            if abs(M[l, c]) > abs(M[max_idx[0], max_idx[1]]):
                max_idx = (l, c)

    # troca linhas se forem diferentes
    if max_idx[0] != i:
        M[[i, max_idx[0]], :] = M[[max_idx[0], i], :]

        if N != None:
            N[[i, max_idx[0]], :] = N[[max_idx[0], i], :]

    # troca colunas da matriz M
    if max_idx[1] != i:
        M[:, [i, max_idx[1]]] = M[:, [max_idx[1], i]]
        # atualiza ordem das variaveis
        if col_order != None:
            col_order[[i, max_idx[1]]] = col_order[[max_idx[1], i]]


# MATRIX ELIMINATION ___________________________________________________________________________________________________

# eliminacao de gauss
def gauss(A, B = None, row_order=None, col_order=None):

    if B.any() != None:
        if A.shape[0] != B.shape[0]:
            raise Exception("MatrixShapeError in function gauss")

    # n = numero de linhas de A
    n = A.shape[0]
    # m = numero de colunas de A
    m = A.shape[1]
    # inicialmente copia de A
    C = A.copy()
    # inicialmente copia de B, se B existir
    D = B.copy() if B.any() != None else None

    for c in xrange(min(n - 1, m)):
        partial_pivoting(C, D, c, row_order)
        total_pivoting(C, D, c, col_order)

        if abs(C[c, c]) > finfo(float).eps:
            for l in xrange((c+1), n):
                alpha = C[l, c] / C[c, c]
                C[l, c] = 0.

                for k in xrange((c+1), m):
                    C[l, k] -= alpha * C[c, k]

                if B.any() != None:
                    for k in xrange(D.shape[1]):
                        D[l, k] -= alpha * D[c, k]
        else:
            break

    return (C, D) if B.any() != None else C

# usado apenas para obtencao da inversa
def jordan(A, B):

    C = A.copy()
    # inicialmente copia de B, se B existir
    D = B.copy()

    for c in reversed(range(A.shape[0])):

        D[c, :] /= C[c, c]
        C[c, c:] /= C[c, c]

        for l in xrange(c):
            D[l, :] -= C[l, c] * D[c, :]
            C[l, c:] -= C[l, c] * C[c, c:]

    return C, D

# MATRIX ATTRIBUTE _____________________________________________________________________________________________________

# rank
def rank(A):
    B = gauss(A)
    for i in xrange(min(A.shape[0], A.shape[1])):
        if not (abs(B[i, i]) > finfo(float).eps):
            return i
    return min(A.shape[0], A.shape[1])

# inv
def inv(A):
    if A.shape[0] != A.shape[1]:
        raise Exception("MatrixShapeError in function inv")

    C = identity(A.shape[0], dtype=float)

    B, C = gauss(A, C)
    B, C = jordan(B, C)

    return C


# LINEAR SISTEM SOLVER _________________________________________________________________________________________________

# resolve sistemas lineares
def solve_ls(A, b):

    # numero de linhas
    n = A.shape[0]
    # guarda a ordem das colunas (variaveis)
    col_order = array(range(A.shape[1]))
    # aplicando eliminacao de gauss
    C, d = gauss(A, b, col_order=col_order)
    # guarda o valor das variaveis
    var_val = zeros(d.shape)

    # replacement
    for l in reversed(range(n)):
        var_sum = 0.
        if abs(C[l, l]) > finfo(float32).eps:
            for c in range(l+1, n):
                var_sum += C[l, c] * var_val[c, 0]
            var_val[l, 0] = (d[l] - var_sum) / C[l, l]
        else:
            var_val[l, 0] = 1.

    sortlist = []
    var_idex = col_order.tolist()
    for i in xrange(n):
        index = var_idex.index(i)
        # print "var_" + str(var_idex[index]) + " = " + str(var_val[index, 0])
        sortlist.append((var_val[index, 0], var_idex[index]))

    sortlist.sort(key= sortSecond)

    return matrix([val[0] for val in sortlist]).transpose()

# MATRIX DECOMPOSITION _________________________________________________________________________________________________

# decomposicao LU
def LU(A):
    if A.shape[0] != A.shape[1]:
        raise Exception("MatrixShapeError in function LU")

    L = identity(A.shape[0], dtype=float)
    U = zeros(A.shape)

    for c in xrange(A.shape[1]):
        for l in xrange(A.shape[0]):
            sum = (L[l, :min(l, c)] * U[:min(l, c), c]).sum()
            if l > c:
                L[l, c] = (A[l, c] - sum) / U[c, c]
            else:
                U[l, c] = A[l, c] - sum

    return L, U


# decomposicao cholesky (apenas para matrizes simetricas)
def cholesky(A):
    if A.shape[0] != A.shape[1]:
        raise Exception("MatrixShapeError in function LU")

    S = zeros(A.shape)

    for c in xrange(A.shape[1]):
        S[c, c] = (A[c, c] - (S[c, :c] * S[c, :c]).sum())**(.5)
        for l in xrange(c+1, A.shape[0]):
            S[l, c] = (A[l, c] - (S[l, :c] * S[c, :c]).sum()) / S[c, c]

    return S

# decomposicao LD
def LD(A):
    if A.shape[0] != A.shape[1]:
        raise Exception("MatrixShapeError in function LU")

    L = identity(A.shape[0], dtype=float)
    D = zeros(A.shape)

    for c in xrange(A.shape[1]):
        D[c, c] = A[c, c] - (L[c, :c] * L[c, :c] * diag(D)[:c]).sum()

        for l in xrange(A.shape[0]):
            L[l, c] = (A[l, c] - (L[l, :c] * L[c, :c] * diag(D)[:c]).sum()) / D[c, c]

    return L, D

# LEAST SQUARES ________________________________________________________________________________________________________

def least_squares(samples, degree):

    A = zeros((len(samples), degree + 1))
    b = zeros((len(samples), 1))

    for s in xrange(len(samples)):
        b[s] = samples[s][1]
        for d in xrange(degree + 1):
            A[s, d] = pow(samples[s][0], degree - d)

    print A
    print b

    print solve_ls(mult(A.transpose(), A), mult(A.transpose(), b))

# EIGENVALUES AND EIGENVECTORS _________________________________________________________________________________________

# autovalores de matrizes 2X2
def eigenvalues(A):
    if A.shape != (2,2):
        raise Exception("MatrixShapeError in function eigenvalues")

    a = 1.
    b = -A[0, 0] - A[1, 1]
    c  = (A[0, 0] * A[1, 1]) - (A[0, 1] * A[1, 0])

    return roots([a, b, c])


def eigen(A):
    eigval = eigenvalues(A)
    b = zeros((2, 1))
    eigpairlist = []

    for i in range(2):
        B = A.copy()
        B[0, 0] -= eigval[i]
        B[1, 1] -= eigval[i]
        # print linalg.solve(B, b)
        eigpairlist.append((eigval[i], solve_ls(B, b)))

    return eigpairlist

# metodo da potencia
def power_method(A, x, epsilon):
    eigvec = x / linalg.norm(x, ord=2)
    x = mult(A, eigvec)
    oldeigval = None
    eigval = prod(eigvec, x)

    while True:
        eigvec = x / linalg.norm(x, ord=2)
        x = mult(A, eigvec)
        oldeigval = eigval
        eigval = prod(eigvec, x)

        if abs(oldeigval - eigval) < epsilon:
            break
    print " "
    return eigval, eigvec/eigvec[1]

def inv_power_method(A, x, epsilon):
    eigval, eigvec = power_method(inv(A), x, epsilon)
    return (1. / eigval), eigvec

def des_power_method(A, x, eps, mi):
    B = A - (identity(A.shape[0]) * mi)
    print B
    eigval, eigvec = inv_power_method(B, x, eps)
    return (eigval + mi), eigvec