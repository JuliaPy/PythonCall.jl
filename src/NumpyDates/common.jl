const NAT = typemin(Int64)

const NAT_STRINGS = ("NaT", "nat", "NAT", "NaN", "nan", "NAN")

function add(x::Integer, y::Integer)
    ans, err = Base.Checked.add_with_overflow(Int64(x), Int64(y))
    (err || ans == NAT) && throw(OverflowError("add"))
    ans
end

function sub(x::Integer, y::Integer)
    ans, err = Base.Checked.sub_with_overflow(Int64(x), Int64(y))
    (err || ans == NAT) && throw(OverflowError("sub"))
    ans
end

function mul(x::Integer, y::Integer)
    ans, err = Base.Checked.mul_with_overflow(Int64(x), Int64(y))
    (err || ans == NAT) && throw(OverflowError("mul"))
    ans
end
