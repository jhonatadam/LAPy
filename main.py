import lapy
import sys


A = 1. * lapy.random.randint(5, size=(3, 5))
b = 1. * lapy.random.randint(10, size=(3, 1))

A[1, :] = A[0, :]
print A

A, b = lapy.gauss(A, b)


print A
print b

# implementar inversa da matriz