import lapy
import sys


A = 1. * lapy.random.randint(10, size=(4, 4))
B = 1. * lapy.random.randint(5, size=(4, 1))

A = lapy.mult(A, A.transpose())

print A

S = lapy.cholesky(A)

print S
print lapy.mult(S, S.transpose())

# print lapy.mult(A, lapy.inv(A))
