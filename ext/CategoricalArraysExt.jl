module CategoricalArraysExt

using PythonCall

using CategoricalArrays: CategoricalArrays

function PythonCall.Compat.aspandasvector(x::CategoricalArrays.CategoricalArray)
    codes = map(x -> x === missing ? -1 : Int(CategoricalArrays.levelcode(x)) - 1, x)
    cats = CategoricalArrays.levels(x)
    ordered = x.pool.ordered
    pyimport("pandas").Categorical.from_codes(codes, cats, ordered = ordered)
end

end
