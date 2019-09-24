import lapy
import sys


A = 1. * lapy.random.randint(10, size=(4, 4))
B = 1. * lapy.random.randint(5, size=(4, 1))

A = lapy.mult(A, A.transpose())

L, D = lapy.LD(A)

print L
print D

print "\n\n"
print lapy.mult(L, lapy.mult(D, L.transpose()))
print A
