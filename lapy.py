from numpy import matrix, zeros, random, finfo, argmax, array


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

def partial_pivoting(M, m, i):
    if abs(M[i, i]) > finfo(float).eps:
        return
    print "aqui"
    n = M.shape[0]
    max_idx = i

    # procurando indice do maior
    for k in xrange(i+1, n):
        if abs(M[k, i]) > abs(M[max_idx, i]):
            max_idx = k

    # realizar troca se houver alguem maior que o pivo
    if max_idx != i:
        M[[i, max_idx], :] = M[[max_idx, i], :]
        m[[i, max_idx], :] = m[[max_idx, i], :]

def total_pivoting(M, i, var_idx):
    if abs(M[i, i]) > finfo(float).eps:
        return

    n = M.shape[0]
    max_idx = (i, i)

    # procurando indice do maior
    for l in xrange(i, n):
        for c in xrange(i+1, n):
            if abs(M[l, c]) > abs(M[max_idx[0], max_idx[1]]):
                max_idx = (l, c)

    if max_idx == (i, i):
        raise Exception("PivotingError in function total_pivoring")

    # troca linhas se forem diferentes
    if max_idx[0] != i:
        M[[i, max_idx[0]], :] = M[[max_idx[0], i], :]
    # troca colunas da matriz
    M[:, [i, max_idx[1]]] = M[:, [max_idx[1], i]]
    # atualiza ordem das variaveis
    var_idx[[i, max_idx[1]]] = var_idx[[max_idx[1], i]]


# precisa adaptar para resolver o problema de encontrar a inversa de A
def gauss(A, b):

    if A.shape[0] != A.shape[1] or A.shape[0] != b.shape[0] or b.shape[1] != 1:
        raise Exception("MatrixShapeError in function gauss")

    n = A.shape[0]
    C = A.copy()
    d = b.copy()
    var_idx = array(range(n))

    for c in xrange(n - 1):
        partial_pivoting(C, d, c)
        total_pivoting(C, c, var_idx)

        for l in xrange((c+1), n):
            alpha = C[l, c] / C[c, c]
            C[l, c] = 0.
            for k in xrange((c+1), n):
                C[l, k] += -1 * alpha * C[c, k]
            d[l] -= alpha * d[c]

    return C, d, var_idx

# resolve sistemas lineares
def solve_ls(A, b):

    C, d, var_idex = gauss(A, b)
    n = C.shape[0]
    var_val = zeros(d.shape)

    for l in reversed(range(n)):
        var_sum = 0.
        for c in range(l+1, n):
            var_sum += C[l, c] * var_val[c, 0]
        var_val[l, 0] = (d[l] - var_sum) / C[l, l]

    var_idex = var_idex.tolist()
    for i in xrange(n):
        index = var_idex.index(i)
        print "var_" + str(var_idex[index]) + " = " + str(var_val[index, 0])

