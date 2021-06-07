PyFraction_Type() = begin
    ptr = POINTERS.PyFraction_Type
    if isnull(ptr)
        POINTERS.PyFraction_Type = ptr = @pydsl_nojlerror begin
            @py import fractions
            PyPtr(fractions.Fraction)
        end onpyerror=(return PyNULL)
    end
    ptr
end

PyFraction_From(x::Union{Rational,Integer}) = @pydsl_nojlerror begin
    PyPtr((@py externb PyFraction_Type())(numerator(x), denominator(x)))
end onpyerror=(return PyNULL)
