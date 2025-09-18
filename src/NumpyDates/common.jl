const NAT = typemin(Int64)

const NAT_STRINGS = ("NaT", "nat", "NAT", "NaN", "nan", "NAN")

const DatesPeriod = Union{
    Dates.Year,
    Dates.Month,
    Dates.Week,
    Dates.Day,
    Dates.Hour,
    Dates.Minute,
    Dates.Second,
    Dates.Millisecond,
    Dates.Microsecond,
    Dates.Nanosecond,
}

const DatesInstant = Union{Dates.Date,Dates.DateTime}

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

function convert_and_check(::Type{T}, x) where {T}
    ans = T(x)::T
    x2 = typeof(x)(ans)
    x == x2 || throw(InexactError(:convert, T, x))
    ans
end
