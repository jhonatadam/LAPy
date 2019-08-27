import numpy


class matrix:

    _M = None

    def __init__(self, filename = None, shape = None):
        if filename:
            _M = numpy.load(filename)
        elif shape:
            _M = numpy.zeros(shape)


    

    def __mul__(self, other):
        pass