for m in ["os", "sys", "pprint", "traceback", "numbers", "math", "collections", "collections.abc", "datetime", "fractions", "io"]
    j = Symbol(:py, replace(m, '.'=>""), :module)
    @eval const $j = PyLazyObject(() -> pyimport($m))
end
