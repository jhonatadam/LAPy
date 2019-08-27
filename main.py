import lapy
import sys


A = 1. * lapy.random.randint(5, size=(6, 6))
b = 1. * lapy.random.randint(10, size=(6, 1))

# A[:, 0] = 0

row_order = lapy.array(range(A.shape[0]))

A, b = lapy.gauss(A, b, row_order=row_order)


print A
print b
print row_order

# consertar pivoteamento total!