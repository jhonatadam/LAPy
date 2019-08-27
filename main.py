import lapy
import sys

A = 1. * lapy.random.randint(10, size=(5, 5))
b = 1. * lapy.random.randint(10, size=(5, 1))

print lapy.mult(A, b)
print lapy.multblock(A, b, 2)
