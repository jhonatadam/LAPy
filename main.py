import lapy
import sys

A = lapy.matrix("[2, 1; 1, 3]", dtype=float)

print lapy.eigen(A)

eigval, eigvec = lapy.power_method(A, lapy.matrix("[1; 1]", dtype=float), lapy.finfo(float).eps)
print eigval
print eigvec
print ""

eigval, eigvec = lapy.inv_power_method(A, lapy.matrix("[1; 1]", dtype=float), lapy.finfo(float).eps)
print eigval
print eigvec
print ""

eigval, eigvec = lapy.des_power_method(A, lapy.matrix("[1; 1]", dtype=float), lapy.finfo(lapy.float32).eps, 1)
print eigval
print eigvec
print ""

print lapy.mult(A, eigvec)
print eigval * eigvec

# for eig in eigens:
#     print lapy.mult(A, eig[1])
    # print eig[0] * eig[1]


# samples = [(1., 1.7), (5.7, 3.), (2.8, 4.2)]
# print lapy.least_squares(samples, 3)

# A = 1. * lapy.random.randint(10, size=(4, 4))
# B = 1. * lapy.random.randint(5, size=(4, 1))
# A = lapy.mult(A, A.transpose())
# L, D = lapy.LD(A)
# print L
# print D
# print lapy.mult(L, lapy.mult(D, L.transpose()))
# print A
