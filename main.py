import lapy
import sys


A = 1. * lapy.random.randint(10, size=(4, 4))
B = 1. * lapy.random.randint(5, size=(4, 1))

# print lapy.mult(A, B)[0,0]

L, U = lapy.LU(A)

print L
print U
print lapy.mult(L, U)

# print lapy.mult(A, lapy.inv(A))
