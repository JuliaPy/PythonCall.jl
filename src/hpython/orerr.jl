iserr(::Context, x) = false
value(::Context, x) = x

struct VoidOrErr
    val :: Cint
end
VoidOrErr() = VoidOrErr(-1)
iserr(::Context, x::VoidOrErr) = x.val == -1
value(::Context, x::VoidOrErr) = nothing

struct BoolOrErr
    val :: Cint
end
BoolOrErr() = BoolOrErr(-1)
iserr(::Context, x::BoolOrErr) = x.val == -1
value(::Context, x::BoolOrErr) = x.val != 0
