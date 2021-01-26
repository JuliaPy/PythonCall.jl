for (n, m) in [(:OS, "os"), (:Sys, "sys"), (:DateTime, "datetime")]
    p = Symbol(:Py_, n, :Module)
    r = Symbol(p, :__ref)
    @eval const $r = Ref(PyNULL)
    @eval $p(doimport::Bool = true) = begin
        ptr = $r[]
        isnull(ptr) || return ptr
        ptr = doimport ? PyImport_ImportModule($m) : PyImport_GetModule($m)
        isnull(ptr) && return ptr
        $r[] = ptr
    end
end
