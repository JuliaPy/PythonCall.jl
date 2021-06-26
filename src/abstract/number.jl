# unary
pyneg(x) = pynew(errcheck(@autopy x C.PyNumber_Negative(getptr(x_))))
pypos(x) = pynew(errcheck(@autopy x C.PyNumber_Positive(getptr(x_))))
pyabs(x) = pynew(errcheck(@autopy x C.PyNumber_Absolute(getptr(x_))))
pyinv(x) = pynew(errcheck(@autopy x C.PyNumber_Invert(getptr(x_))))
pyindex(x) = pynew(errcheck(@autopy x C.PyNumber_Index(getptr(x_))))
export pyneg, pypos, pyabs, pyinv, pyindex

# binary
pyadd(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Add(getptr(x_), getptr(y_))))
pysub(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Subtract(getptr(x_), getptr(y_))))
pymul(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Multiply(getptr(x_), getptr(y_))))
pymatmul(x, y) = pynew(errcheck(@autopy x y C.PyNumber_MatrixMultiply(getptr(x_), getptr(y_))))
pyfloordiv(x, y) = pynew(errcheck(@autopy x y C.PyNumber_FloorDivide(getptr(x_), getptr(y_))))
pytruediv(x, y) = pynew(errcheck(@autopy x y C.PyNumber_TrueDivide(getptr(x_), getptr(y_))))
pymod(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Remainder(getptr(x_), getptr(y_))))
pydivmod(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Divmod(getptr(x_), getptr(y_))))
pylshift(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Lshift(getptr(x_), getptr(y_))))
pyrshift(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Rshift(getptr(x_), getptr(y_))))
pyand(x, y) = pynew(errcheck(@autopy x y C.PyNumber_And(getptr(x_), getptr(y_))))
pyxor(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Xor(getptr(x_), getptr(y_))))
pyor(x, y) = pynew(errcheck(@autopy x y C.PyNumber_Or(getptr(x_), getptr(y_))))
export pyadd, pysub, pymul, pymatmul, pyfloordiv, pytruediv, pymod, pydivmod, pylshift, pyrshift, pyans, pyxor, pyor

# binary in-place
pyiadd(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceAdd(getptr(x_), getptr(y_))))
pyisub(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceSubtract(getptr(x_), getptr(y_))))
pyimul(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceMultiply(getptr(x_), getptr(y_))))
pyimatmul(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceMatrixMultiply(getptr(x_), getptr(y_))))
pyifloordiv(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceFloorDivide(getptr(x_), getptr(y_))))
pyitruediv(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceTrueDivide(getptr(x_), getptr(y_))))
pyimod(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceRemainder(getptr(x_), getptr(y_))))
pyidivmod(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceDivmod(getptr(x_), getptr(y_))))
pyilshift(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceLshift(getptr(x_), getptr(y_))))
pyirshift(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceRshift(getptr(x_), getptr(y_))))
pyiand(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceAnd(getptr(x_), getptr(y_))))
pyixor(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceXor(getptr(x_), getptr(y_))))
pyior(x, y) = pynew(errcheck(@autopy x y C.PyNumber_InPlaceOr(getptr(x_), getptr(y_))))
export pyiadd, pyisub, pyimul, pyimatmul, pyifloordiv, pyitruediv, pyimod, pyidivmod, pyilshift, pyirshift, pyians, pyixor, pyior

# power
pypow(x, y, z=pybuiltins.None) = pynew(errcheck(@autopy x y z C.PyNumber_Power(getptr(x_), getptr(y_), getptr(z_))))
pyipow(x, y, z=pybuiltins.None) = pynew(errcheck(@autopy x y z C.PyNumber_InPlacePower(getptr(x_), getptr(y_), getptr(z_))))
export pypow, pyipow
