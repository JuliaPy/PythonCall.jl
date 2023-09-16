# unary
"""
    pyneg(x)

Equivalent to `-x` in Python.
"""
pyneg(x) = pynew(errcheck(@autopy x C.PyNumber_Negative(getptr(x_))))
"""
    pypos(x)

Equivalent to `+x` in Python.
"""
pypos(x) = pynew(errcheck(@autopy x C.PyNumber_Positive(getptr(x_))))
"""
    pyabs(x)

Equivalent to `abs(x)` in Python.
"""
pyabs(x) = pynew(errcheck(@autopy x C.PyNumber_Absolute(getptr(x_))))
"""
    pyinv(x)

Equivalent to `~x` in Python.
"""
pyinv(x) = pynew(errcheck(@autopy x C.PyNumber_Invert(getptr(x_))))
"""
    pyindex(x)

Convert `x` losslessly to an `int`.
"""
pyindex(x) = pynew(errcheck(@autopy x C.PyNumber_Index(getptr(x_))))
export pyneg, pypos, pyabs, pyinv, pyindex

# binary
"""
    pyadd(x, y)

Equivalent to `x + y` in Python.
"""
pyadd(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Add(getptr(x_), getptr(y_))))
"""
    pysub(x, y)

Equivalent to `x - y` in Python.
"""
pysub(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Subtract(getptr(x_), getptr(y_))))
"""
    pymul(x, y)

Equivalent to `x * y` in Python.
"""
pymul(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Multiply(getptr(x_), getptr(y_))))
"""
    pymatmul(x, y)

Equivalent to `x @ y` in Python.
"""
pymatmul(x, y) = pynew(errcheck(@autopy x y C.PyNumber_MatrixMultiply(getptr(x_), getptr(y_))))
"""
    pyfloordiv(x, y)

Equivalent to `x // y` in Python.
"""
pyfloordiv(x, y) = pynew(errcheck(@autopy x y C.PyNumber_FloorDivide(getptr(x_), getptr(y_))))
"""
    pytruediv(x, y)

Equivalent to `x / y` in Python.
"""
pytruediv(x, y) = pynew(errcheck(@autopy x y C.PyNumber_TrueDivide(getptr(x_), getptr(y_))))
"""
    pymod(x, y)

Equivalent to `x % y` in Python.
"""
pymod(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Remainder(getptr(x_), getptr(y_))))
"""
    pydivmod(x, y)

Equivalent to `divmod(x, y)` in Python.
"""
pydivmod(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Divmod(getptr(x_), getptr(y_))))
"""
    pylshift(x, y)

Equivalent to `x << y` in Python.
"""
pylshift(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Lshift(getptr(x_), getptr(y_))))
"""
    pyrshift(x, y)

Equivalent to `x >> y` in Python.
"""
pyrshift(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Rshift(getptr(x_), getptr(y_))))
"""
    pyand(x, y)

Equivalent to `x & y` in Python.
"""
pyand(x, y) = pynew(errcheck(@autopy x y C.PyNumber_And(getptr(x_), getptr(y_))))
"""
    pyxor(x, y)

Equivalent to `x ^ y` in Python.
"""
pyxor(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Xor(getptr(x_), getptr(y_))))
"""
    pyor(x, y)

Equivalent to `x | y` in Python.
"""
pyor(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Or(getptr(x_), getptr(y_))))
export pyadd, pysub, pymul, pymatmul, pyfloordiv, pytruediv, pymod, pydivmod, pylshift, pyrshift, pyand, pyxor, pyor

# binary in-place
"""
    pyiadd(x, y)

In-place add. `x = pyiadd(x, y)` is equivalent to `x += y` in Python.
"""
pyiadd(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceAdd(getptr(x_), getptr(y_))))
"""
    pyisub(x, y)

In-place subtract. `x = pyisub(x, y)` is equivalent to `x -= y` in Python.
"""
pyisub(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceSubtract(getptr(x_), getptr(y_))))
"""
    pyimul(x, y)

In-place multiply. `x = pyimul(x, y)` is equivalent to `x *= y` in Python.
"""
pyimul(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceMultiply(getptr(x_), getptr(y_))))
"""
    pyimatmul(x, y)

In-place matrix multiply. `x = pyimatmul(x, y)` is equivalent to `x @= y` in Python.
"""
pyimatmul(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceMatrixMultiply(getptr(x_), getptr(y_))))
"""
    pyifloordiv(x, y)

In-place floor divide. `x = pyifloordiv(x, y)` is equivalent to `x //= y` in Python.
"""
pyifloordiv(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceFloorDivide(getptr(x_), getptr(y_))))
"""
    pyitruediv(x, y)

In-place true division. `x = pyitruediv(x, y)` is equivalent to `x /= y` in Python.
"""
pyitruediv(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceTrueDivide(getptr(x_), getptr(y_))))
"""
    pyimod(x, y)

In-place subtraction. `x = pyimod(x, y)` is equivalent to `x %= y` in Python.
"""
pyimod(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceRemainder(getptr(x_), getptr(y_))))
"""
    pyilshift(x, y)

In-place left shift. `x = pyilshift(x, y)` is equivalent to `x <<= y` in Python.
"""
pyilshift(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceLshift(getptr(x_), getptr(y_))))
"""
    pyirshift(x, y)

In-place right shift. `x = pyirshift(x, y)` is equivalent to `x >>= y` in Python.
"""
pyirshift(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceRshift(getptr(x_), getptr(y_))))
"""
    pyiand(x, y)

In-place and. `x = pyiand(x, y)` is equivalent to `x &= y` in Python.
"""
pyiand(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceAnd(getptr(x_), getptr(y_))))
"""
    pyixor(x, y)

In-place xor. `x = pyixor(x, y)` is equivalent to `x ^= y` in Python.
"""
pyixor(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceXor(getptr(x_), getptr(y_))))
"""
    pyior(x, y)

In-place or. `x = pyior(x, y)` is equivalent to `x |= y` in Python.
"""
pyior(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceOr(getptr(x_), getptr(y_))))
export pyiadd, pyisub, pyimul, pyimatmul, pyifloordiv, pyitruediv, pyimod, pyilshift, pyirshift, pyiand, pyixor, pyior

# power
"""
    pypow(x, y, z=None)

Equivalent to `x ** y` or `pow(x, y, z)` in Python.
"""
pypow(x, y, z=pybuiltins.None) = pynew(errcheck(@autopy x y z C.PyNumber_Power(getptr(x_), getptr(y_), getptr(z_))))
"""
    pyipow(x, y, z=None)

In-place power. `x = pyipow(x, y)` is equivalent to `x **= y` in Python.
"""
pyipow(x, y, z=pybuiltins.None) = pynew(errcheck(@autopy x y z C.PyNumber_InPlacePower(getptr(x_), getptr(y_), getptr(z_))))
export pypow, pyipow
