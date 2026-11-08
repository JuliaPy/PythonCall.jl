@testitem "conversion registry, scopes and global ordering" begin
    using PythonCall
    const CV = PythonCall.Convert
    ns = pyimport("types").ModuleType("convert_redesign_test")
    pyimport("sys").modules["convert_redesign_test"] = ns
    pyexec("""class A: pass
class B: pass
class C(A, B): pass
class D(C): pass
class E(B, A): pass
class Protocol:
    __array_interface__ = {}
class Both(Protocol):
    def __array__(self): pass
""", ns.__dict__)
    ids = Int[]
    add(name,T,f;scope=T) = (id=pyconvert_add_rule("convert_redesign_test:"*name,T,f;scope); push!(ids,id); id)
    try
        b = add("B",String, (_,_) -> "B"; scope=Any)
        a = add("A",String, (_,_) -> "A"; scope=Any)
        @test pyconvert(Any, ns.C()) == "B"
        @test pyconvert(Any, ns.D()) == "B"
        @test pyconvert(Any, ns.E()) == "B" # opposite MRO
        selected(x) = [r.id for r in pyconvert_rules(Any,x) if r.eligible]
        @test selected(ns.C()) == selected(ns.D()) == selected(ns.E())
        c = add("C",String, (_,_) -> "C"; scope=Any)
        @test pyconvert(Any,ns.D()) == "C"
        @test pyconvert_replace_rule(c,"convert_redesign_test:C",String,(_,_) -> nothing;scope=Any) == c
        @test pyconvert(Any,ns.C()) == "B"
        duplicate = add("B",String,(_,_) -> "later"; scope=Any)
        @test duplicate > c > a > b
        @test pyconvert(Any,ns.C()) == "B"
        pyconvert_remove_rule(b); filter!(!=(b),ids)
        @test pyconvert(Any,ns.C()) == "later"
        @test_throws KeyError pyconvert_remove_rule(b)
        scoped = add("A",Vector,(_,_) -> [1]; scope=Vector)
        @test pyconvert(Vector,ns.A()) == [1]
        @test pyconvert(Any,ns.A()) == "A"
        @test pytryconvert(AbstractArray,ns.A()) === nothing
        @test pyconvert(Union{Vector,Nothing},ns.A()) == [1]
        @test pytryconvert(Union{AbstractArray,Vector},ns.A()) === nothing
        @test pytryconvert(Union{},ns.A()) === nothing
        @test all(!r.eligible for r in pyconvert_rules(Any,ns.A()) if r.id==scoped)
        @test CV.pyconvert_protocol("<arrayinterface>",ns.Both)
        @test CV.pyconvert_protocol("<array>",ns.Both)
        @test !CV.pyconvert_protocol("<buffer>",ns.Both)
        # Explicit rule outranks its generic virtual protocol parent.
        p = add("Both",String,(_,_) -> "specific";scope=Any)
        @test pyconvert(Any,ns.Both()) == "specific"
        rows = pyconvert_rules(Any,ns.Both())
        @test first(rows).id == p
        @test first(rows).priority >= maximum(r.priority for r in rows)
    finally
        foreach(pyconvert_remove_rule,ids)
        pyimport("sys").modules.pop("convert_redesign_test")
    end
end

@testitem "native containers and preferences" begin
    using PythonCall
    cases = [(Any[],Any), ([1,2],Int), (Any[1,2.5],Real),
             ([1,"two"],Any), ([nothing,1],Union{Nothing,Int})]
    for (values,T) in cases
        result = pyconvert(Any,pylist(values))
        @test result isa Vector{T}
        @test isequal(result,values)
    end
    @test pyconvert(Any,pylist([big(2)^200,1])) isa Vector{<:Integer}
    @test pyconvert(Vector{Int},pylist([1,2])) == [1,2]
    @test pyconvert(Vector{<:Real},pylist(Any[1,2.5])) isa Vector{Real}
    @test typeof.(pyconvert(Any,pylist(Any[1,2.5]))) == [Int,Float64]
    @test pytryconvert(Vector{Int},pylist([1,"bad"])) === nothing
    @test pyconvert(Any,pydict()) isa Dict{Any,Any}
    d = pyconvert(Any,pydict("a"=>1,"b"=>2.5))
    @test d isa Dict{String,Real}
    @test d == Dict("a"=>1,"b"=>2.5)
    @test pyconvert(Any,pylist([pylist([1,2])])) == [[1,2]]
    @test pyconvert(Any,pyiter(pylist([1,2]))) isa PyIterable
    @test eltype(pyconvert(Any,pyiter(pylist([1,2])))) === Any
    pref = ("builtins:list"=>PyList, "builtins:dict"=>PyDict)
    @test pyconvert(Any,pylist([1,2]);prefer=pref) isa PyList
    @test pyconvert(Any,pydict();prefer=pref) isa PyDict
    @test pyconvert(Vector,pylist([1,2]);prefer=pref) isa Vector
    @test pyconvert(Any,pylist([1]);prefer=("builtins:dict"=>PyDict,)) isa Vector
    ctx = PyConvertContext(;prefer=pref)
    @test pyconvert(ctx,Vector,pylist([pylist([1])]))[1] isa PyList
    @test pyconvert(Any,pylist([1]);prefer=("builtins:list"=>Int8,)) == [1]
    @test pyconvert(Any,pylist([1]);prefer=("builtins:list"=>PyList,"builtins:list"=>Vector)) isa PyList
end

@testitem "scopes with parametric bounds and union normalization" begin
    using PythonCall
    CV = PythonCall.Convert
    rule(T,S) = CV.PyConvertRule(1,"builtins:list",T,S,(_,_) -> error("ineligible"))
    @test CV.pyconvert_eligible(rule(Vector,AbstractVector),Vector{Int})
    @test CV.pyconvert_eligible(rule(Vector,AbstractVector),Vector{<:Real})
    @test !CV.pyconvert_eligible(rule(Vector,Vector),AbstractVector)
    @test !CV.pyconvert_eligible(rule(Vector,Vector),Any)
    @test CV.pyconvert_eligible(rule(Vector{<:Real},Any),Vector)
    @test !CV.pyconvert_eligible(rule(Vector{String},Any),Vector{<:Real})
    nested = Vector{Vector{T}} where T<:Real
    @test CV.pyconvert_eligible(rule(nested,nested),Vector{Vector{Int}})
    @test !CV.pyconvert_eligible(rule(nested,nested),Vector{Vector{String}})
    @test CV.pyconvert_eligible(rule(nested,nested),Union{Nothing,Vector{Vector{Int}}})
    @test !CV.pyconvert_eligible(rule(Vector,Vector),Union{AbstractVector,Vector})
    @test !CV.pyconvert_eligible(rule(Union{},Any),Any)
    @test !CV.pyconvert_eligible(rule(Any,Any),Union{})
    @test !CV.pyconvert_eligible(rule(Vector,Union{}),Vector)
end

@testitem "protocol ordering, ABCs and cached scalar overrides" begin
    using PythonCall
    CV=PythonCall.Convert
    ns=pyimport("types").ModuleType("convert_protocol_test")
    pyimport("sys").modules["convert_protocol_test"]=ns
    pyexec("""
import abc
import numpy
class A(metaclass=abc.ABCMeta): pass
class B: pass
A.register(B)
class C(B): pass
class IntSubclass(int): pass
class FloatSubclass(float): pass
class Both(numpy.ndarray): pass
""",ns.__dict__)
    ids=Int[]
    add(source,T,f;scope=Any)=(id=pyconvert_add_rule(source,T,f;scope);push!(ids,id);id)
    try
        add("convert_protocol_test:A",String,(_,_) -> "ABC")
        @test pyconvert(Any,ns.B()) == pyconvert(Any,ns.C()) == "ABC"
        # Prime the scalar cache, then register a stricter source rule.
        @test pyconvert(Int,ns.IntSubclass(2)) == 2
        bigrule=add("convert_protocol_test:IntSubclass",BigInt,(_,_) -> big(7);scope=BigInt)
        @test pyconvert(BigInt,ns.IntSubclass(2)) == big(7)
        @test pyconvert(Any,ns.IntSubclass(2)) === 2
        pyconvert_replace_rule(bigrule,"convert_protocol_test:IntSubclass",BigInt,(_,_) -> nothing;scope=BigInt)
        @test pyconvert(BigInt,ns.IntSubclass(2)) == big(2)
        pyconvert_remove_rule(bigrule);filter!(!=(bigrule),ids)
        i=add("convert_protocol_test:IntSubclass",Int,(_,_) -> 3)
        @test pyconvert(Int,ns.IntSubclass(2)) == 3
        pyconvert_replace_rule(i,"convert_protocol_test:IntSubclass",Int,(_,_) -> 4;scope=Any)
        @test pyconvert(Int,ns.IntSubclass(2)) == 4
        pyconvert_remove_rule(i);filter!(!=(i),ids)
        @test pyconvert(Int,ns.IntSubclass(2)) == 2
        @test pyconvert(Float64,ns.FloatSubclass(2)) == 2.0
        add("convert_protocol_test:FloatSubclass",Float64,(_,_) -> 3.0)
        @test pyconvert(Float64,ns.FloatSubclass(2)) == 3.0
        a=ns.numpy.arange(3).view(ns.Both)
        @test pyconvert(Any,a) isa Vector{<:Integer}
        rows=pyconvert_rules(Any,a)
        protocols=unique(r.source for r in rows if startswith(r.source,"<"))
        @test collect(protocols) == ["<arraystruct>","<arrayinterface>","<array>","<buffer>"]
        i=add("numpy:ndarray",String,(_,_) -> "ndarray")
        @test pyconvert(Any,a) == "ndarray"
        rows=pyconvert_rules(Any,a)
        @test first(rows).id==i
        @test first(rows).priority==maximum(r.priority for r in rows)
        @test pyconvert(Any,pyjl([1,2])) == [1,2]
        @test first(pyconvert_rules(Any,pyjl([1,2]))).source=="juliacall:JlBase"
        @test pyconvert(Any,ns.numpy.arange(2);prefer=("numpy:ndarray"=>PyArray,)) isa PyArray
    finally
        foreach(pyconvert_remove_rule,ids)
        pyimport("sys").modules.pop("convert_protocol_test")
    end
end

@testitem "conversion exceptions and explicit context boundaries" begin
    using PythonCall
    ns=pyimport("types").ModuleType("convert_context_test")
    pyimport("sys").modules["convert_context_test"]=ns
    pyexec("""
class Value:
    child = [1,2]
class Iterator:
    def __iter__(self):
        yield 1
        raise RuntimeError('iterator failed')
class BadArray:
    def __array__(self):
        raise RuntimeError('array failed')
""",ns.__dict__)
    source="convert_context_test:Value"
    id=pyconvert_add_rule(source,Any,(ctx,T,x)->pyconvert(ctx,Any,x.child);scope=Any)
    try
        pref=(source=>String,"builtins:list"=>PyList)
        # Failed preference for String then ordinary conversion propagates context.
        # Restrict the advertised output so the String attempt genuinely declines.
        pyconvert_replace_rule(id,source,AbstractVector,(ctx,T,x)->pyconvert(ctx,Any,x.child);scope=Any)
        @test pyconvert(Any,ns.Value();prefer=pref) isa PyList
        pyconvert_replace_rule(id,source,AbstractVector,(T,x)->pyconvert(Any,x.child);scope=Any)
        @test pyconvert(Any,ns.Value();prefer=pref) isa Vector
        @test_throws "iterator failed" pyconvert(Vector,ns.Iterator())
        @test pyconvert(Any,ns.Iterator()) isa PyIterable
        @test_throws "array failed" pyconvert(Any,ns.BadArray())
        pyconvert_replace_rule(id,source,AbstractVector,(T,x)->error("preference failed");scope=Any)
        @test_throws "preference failed" pyconvert(Any,ns.Value();prefer=(source=>Vector,))
    finally
        pyconvert_remove_rule(id)
        pyimport("sys").modules.pop("convert_context_test")
    end
end

@testitem "rule result convention and context signatures" begin
    using PythonCall
    const CV = PythonCall.Convert
    ns = pyimport("types").ModuleType("convert_result_test")
    pyimport("sys").modules["convert_result_test"] = ns
    pyexec("class Value: pass",ns.__dict__)
    source = "convert_result_test:Value"
    x = ns.Value()
    id = pyconvert_add_rule(source,Any,(_,_) -> Some(nothing);scope=Any)
    try
        @test pytryconvert(Any,x) === Some(nothing)
        @test pyconvert(Any,x) === nothing
        pyconvert_replace_rule(id,source,Any,(_,_) -> missing;scope=Any)
        @test pyconvert(Any,x) === missing
        pyconvert_replace_rule(id,source,Any,(_,_) -> Some(Some(2));scope=Any)
        @test pyconvert(Any,x) === Some(2)
        for value in (CV.PYCONVERT_FAILED, CV.PyConvertEscape(1), Some(nothing))
            pyconvert_replace_rule(id,source,Any,(_,_) -> Some(value);scope=Any)
            @test pyconvert(Any,x) === value
            @test pytryconvert(Any,x) === Some(value)
            @test pyconvert(typeof(value),x) === value
        end
        # Built-in result encoding must preserve ambiguous values exactly once.
        for value in (nothing, missing, Some(1), Some(nothing), Some(Some(1)),
                      [1,2], CV.PYCONVERT_FAILED, CV.PyConvertEscape(1))
            pyconvert_replace_rule(id,source,Any,(_,_) -> CV.pyconvert_return(value);scope=Any)
            @test pyconvert(Any,x) === value
            @test pytryconvert(typeof(value),x) === Some(value)
            @test pyconvert(Any,pyjl(value)) === value
            @test pytryconvert(typeof(value),pyjl(value)) === Some(value)
        end
        @test pyconvert(Any,pylist([pyjl(Some(1)),pyjl(nothing)])) == [Some(1),nothing]
        @test pyconvert(Any,pydict("some"=>pyjl(Some(1)),"none"=>pyjl(nothing))) ==
              Dict("some"=>Some(1),"none"=>nothing)
        pyconvert_replace_rule(id,source,Any,(_,_) -> nothing;scope=Any)
        @test pyconvert(Any,x) === x # implicit fallback
        @test pytryconvert(Int,x) === nothing
        pyconvert_replace_rule(id,source,Any,(_,_) -> error("rule error");scope=Any)
        @test_throws "rule error" pyconvert(Any,x)
        function rule(::Type{Int},x)
            1
        end
        function rule(ctx,::Type{String},x)
            "context"
        end
        function rule(::Type{String},x)
            "legacy"
        end
        pyconvert_replace_rule(id,source,Union{Int,String},rule;scope=Any)
        @test pyconvert(Int,x) == 1
        @test pyconvert(String,x) == "context"
        @test pyconvert(Union{Int,String},x) in (1,"context")
        calls=Any[]
        pyconvert_replace_rule(id,source,Union{Int,String},(T,x)->(push!(calls,T);nothing);scope=Any)
        second=pyconvert_add_rule(source,Union{Int,String},(T,x)->(push!(calls,:second);T===Int ? 1 : "ok");scope=Any)
        try
            @test pyconvert(Union{Int,String},x) in (1,"ok")
            @test Set(calls[1:2]) == Set([Int,String])
            @test calls[3] === :second
        finally
            pyconvert_remove_rule(second)
        end
        @test Some(1) isa CV.pyconvert_returntype(Integer)
        @test Some(nothing) isa CV.pyconvert_returntype(Union{Int,Nothing})
        @test Some(1) isa CV.pyconvert_returntype(Any)
        @test CV.pyconvert_returntype(Int) === Union{Nothing,Some{Int}}
        @test CV.pyconvert_returntype(Union{Int,Nothing}) === Union{Nothing,Some{Int},Some{Nothing}}
        @test (@inferred pyconvert(Int,pyint(1))) === 1
        @test pytryconvert(Nothing,pybuiltins.None) === Some(nothing)
    finally
        pyconvert_remove_rule(id)
        pyimport("sys").modules.pop("convert_result_test")
    end
end

@testitem "scalar dispatch preserves singleton rules and overflow behavior" begin
    using PythonCall
    CV=PythonCall.Convert
    original=only(r for r in CV.PYCONVERT_RULES if r.source=="types:NoneType" && r.type===Nothing)
    x=pybuiltins.None
    @test pyconvert(Nothing,x) === nothing
    calls=Ref(0)
    try
        pyconvert_replace_rule(original.id,original.source,Nothing,(T,x)->(calls[]+=1;Some(nothing));scope=Any)
        @test pyconvert(Nothing,x) === nothing
        @test calls[]==1
        pyconvert_replace_rule(original.id,original.source,Nothing,(T,x)->nothing;scope=Any)
        @test pytryconvert(Nothing,x) === nothing
        @test pyconvert(Nothing,x,:declined) === :declined
        pyconvert_replace_rule(original.id,original.source,Nothing,(T,x)->error("singleton rule");scope=Any)
        @test_throws "singleton rule" pyconvert(Nothing,x)
    finally
        pyconvert_replace_rule(original.id,original.source,original.type,original.func;scope=original.scope)
    end
    @test pytryconvert(Nothing,x) === Some(nothing)
    @test (@inferred pyconvert(Float64,pyfloat(1.5))) === 1.5
    @test (@inferred pyconvert(Int,pyint(10000))) === 10000
    for n in (big(2)^200,-big(2)^200,big(typemax(Int))+1)
        p=pyint(n)
        @test pyconvert(BigInt,p)==n
        @test pyconvert(Integer,p)==n
        @test pyconvert(Any,p)==n
        @test pytryconvert(Int,p) === nothing
    end
    # An exact integer decline must not retry through a rounded float/complex.
    below_min=pyint(big(typemin(Int))-1)
    @test CV.pyconvert_rule_int(Int,below_min) === nothing
    @test pytryconvert(Int,below_min) === nothing
    @test pyconvert(BigInt,pyint(42))==42
    @test pyconvert(UInt,pyint(typemax(UInt)))==typemax(UInt)
    @test pytryconvert(UInt,pyint(-1)) === nothing
end

@testitem "numeric conversion uses source representations" begin
    using PythonCall
    ns = pyimport("types").ModuleType("convert_numeric_source_test")
    pyimport("sys").modules["convert_numeric_source_test"] = ns
    pyexec("""
import numbers
from fractions import Fraction
class Integral(int):
    def __float__(self): raise AssertionError('integral __float__ called')
    def __complex__(self): raise AssertionError('integral __complex__ called')
class Rational(Fraction):
    def __float__(self): raise AssertionError('rational __float__ called')
    def __complex__(self): raise AssertionError('rational __complex__ called')
class Float(float):
    def __float__(self): raise AssertionError('float override called')
    def __complex__(self): raise AssertionError('float complex override called')
class Complex(complex):
    def __complex__(self): raise AssertionError('complex override called')
class ApproximateReal:
    def __float__(self): raise AssertionError('real __float__ called')
    def __complex__(self): raise AssertionError('real __complex__ called')
class ApproximateComplex:
    def __complex__(self): raise AssertionError('complex __complex__ called')
numbers.Real.register(ApproximateReal)
numbers.Complex.register(ApproximateComplex)
""", ns.__dict__)
    try
        n = big(typemin(Int))-1
        for x in (pyint(n), ns.Integral(pyint(n)), ns.Rational(pyint(n),1))
            @test pytryconvert(Int,x) === nothing
            @test pyconvert(BigInt,x) == n
            @test pyconvert(Float64,x) == Float64(n)
            @test pyconvert(ComplexF64,x) == ComplexF64(n)
        end
        @test pyconvert(Int,ns.Integral(3)) === 3
        @test pyconvert(Int,ns.Rational(6,2)) === 3
        @test pyconvert(Rational,ns.Rational(1,3)) == 1//3
        @test pyconvert(Float64,ns.Rational(1,3)) === Float64(1//3)
        @test pytryconvert(Int,ns.Rational(pyint(big(2)^63+1),2)) === nothing
        for x in (ns.ApproximateReal(),ns.ApproximateComplex())
            for T in (Int,Float64,ComplexF64,Number)
                @test pytryconvert(T,x) === nothing
            end
            @test pyconvert(Any,x) === x
        end
        @test pyconvert(Int,pyfloat(3.0)) === 3
        @test pytryconvert(Int,pyfloat(3.5)) === nothing
        @test pyconvert(Int,pycomplex(3,0)) === 3
        @test pytryconvert(Int,pycomplex(3,1)) === nothing
        @test pyconvert(Any,ns.Float(1.5)) === 1.5
        @test pyconvert(ComplexF64,ns.Float(1.5)) === 1.5+0.0im
        @test pyconvert(Any,ns.Complex(1,2)) === 1.0+2.0im
        id = pyconvert_add_rule("convert_numeric_source_test:ApproximateReal",Float64,(_,_) -> 1.5)
        try
            @test pyconvert(Float64,ns.ApproximateReal()) === 1.5
            @test pytryconvert(Number,ns.ApproximateReal()) === nothing
        finally
            pyconvert_remove_rule(id)
        end
        # Supported NumPy scalars use their own representation-reading rules.
        np = pyimport("numpy")
        @test pyconvert(Int,np.int64(3)) === 3
        @test pyconvert(Float64,np.float32(1.5)) === 1.5
        @test pyconvert(ComplexF64,np.complex64(pycomplex(1,2))) === 1.0+2.0im
        @test pytryconvert(Int,np.float64(3.5)) === nothing
    finally
        pyimport("sys").modules.pop("convert_numeric_source_test")
    end
end

@testitem "effective source priorities are monotonic" begin
    using PythonCall
    CV=PythonCall.Convert
    pyconvert_rules(Any,pyint(1))
    active=Set(r.source for r in CV.PYCONVERT_ORDER)
    sources=filter(s->s.name in active || s.name=="builtins:object",CV.PYCONVERT_SOURCES)
    for a in sources, b in sources
        CV.pyconvert_subtype(a,b) || continue
        @test a.priority >= b.priority
    end
end

@testitem "lazy rule resolution" begin
    using PythonCall
    id = pyconvert_add_rule("convert_late_test:Value",String,(_,_) -> "late";scope=Any)
    try
        @test pyconvert(Int,pyint(1)) == 1
        ns = pyimport("types").ModuleType("convert_late_test")
        pyexec("class Value: pass",ns.__dict__)
        pyimport("sys").modules["convert_late_test"] = ns
        @test pyconvert(Any,ns.Value()) == "late"
    finally
        pyconvert_remove_rule(id)
        pyimport("sys").modules.pop("convert_late_test",nothing)
    end
end

@testitem "source names resolve public aliases and built-in shorthand" begin
    using PythonCall
    CV = PythonCall.Convert
    modules = pyimport("sys").modules
    types = pyimport("types")
    impl = types.ModuleType("convert_names_internal")
    api = types.ModuleType("convert_names_public")
    modules["convert_names_internal"] = impl
    pyexec("""
class Base: pass
class Outer:
    class Internal(Base): pass
class Child(Outer.Internal): pass
""", impl.__dict__)
    api.Public = impl.Outer.Internal
    api.Namespace = types.SimpleNamespace(Value=impl.Outer.Internal)
    ids = Int[]
    add(source,T,f;scope=T) = (id=pyconvert_add_rule(source,T,f;scope);push!(ids,id);id)
    try
        base = add("convert_names_internal:Base",String,(_,_) -> "base";scope=Any)
        firstalias = add("convert_names_public:Public",String,(_,_) -> nothing;scope=Any)
        secondalias = add("convert_names_public:Namespace.Value",String,(_,_) -> "alias";scope=Any)
        # Registration and unrelated conversion do not import an optional module.
        @test !pycontains(modules,"convert_names_public")
        @test pyconvert(Int,pyint(1)) === 1
        @test "convert_names_public:Public" in CV.PYCONVERT_PENDING
        modules["convert_names_public"] = api
        @test pyconvert(String,api.Public.__module__) == "convert_names_internal"
        @test pyconvert(String,api.Public.__qualname__) == "Outer.Internal"
        @test pyconvert(Any,impl.Child()) == "alias"
        rows = pyconvert_rules(Any,impl.Child())
        @test [r.id for r in rows if r.id in ids] == [firstalias,secondalias,base]
        @test !("convert_names_public:Public" in CV.PYCONVERT_PENDING)
        # Preference lookup also works for an unregistered public alias.
        api.OtherName = impl.Outer.Internal
        @test pyconvert(Any,impl.Child();prefer=("convert_names_public:OtherName"=>Py,)) isa Py
        @test pyconvert(Any,impl.Child();prefer=("convert_names_missing:Public"=>Py,)) == "alias"
        @test !pycontains(modules,"convert_names_missing")
        # Bare builtins work in preferences, contexts, rule adds and replacements.
        x = pylist([1,2])
        @test pyconvert(Any,x;prefer=("list"=>PyList,)) isa PyList
        ctx = PyConvertContext(;prefer=("list"=>PyList,"dict"=>PyDict))
        @test pyconvert(ctx,Any,pydict()) isa PyDict
        @test pyconvert(ctx,Vector,pylist([x]))[1] isa PyList
        @test pyconvert(Vector,x;prefer=("list"=>PyList,)) == [1,2]
        @test pyconvert(Any,x;prefer=("dict"=>PyDict,)) == [1,2]
        id = add("list",String,(_,_) -> "list rule")
        @test pyconvert(String,x) == "list rule"
        @test only(r.source for r in pyconvert_rules(String,x) if r.id==id) == "builtins:list"
        @test pyconvert(Any,x) == [1,2]
        @test pyconvert_replace_rule(id,"dict",String,(_,_) -> "dict rule") == id
        @test pytryconvert(String,x) === nothing
        @test pyconvert(String,pydict()) == "dict rule"
        @test only(r.source for r in pyconvert_rules(String,pydict()) if r.id==id) == "builtins:dict"
        @test pyis(CV.pyconvert_resolve("list"),pybuiltins.list)
        @test pyis(CV.pyconvert_resolve("NoneType"),pytype(pybuiltins.None))
        # An API alias also retains priorities attached to special source types.
        api.JuliaValue = pyimport("juliacall").JlBase
        wrapid = add("convert_names_public:JuliaValue",String,(_,_) -> nothing)
        rows = pyconvert_rules(String,pyjl([1,2]))
        canonical = only(r.priority for r in rows if r.source=="juliacall:JlBase")
        @test only(r.priority for r in rows if r.id==wrapid) == canonical
    finally
        foreach(pyconvert_remove_rule,ids)
        modules.pop("convert_names_public",nothing)
        modules.pop("convert_names_internal",nothing)
    end
end

@testitem "built-in rules use public source locations" begin
    using PythonCall
    CV = PythonCall.Convert
    @test pyconvert(Any,pybuiltins.None) === nothing
    @test first(pyconvert_rules(Any,pybuiltins.None)).source == "types:NoneType"
    @test pyis(CV.pyconvert_resolve("types:NoneType"),pytype(pybuiltins.None))
    io = pyimport("io")
    privatebase = pyimport("_io")._IOBase
    @test !pyis(io.IOBase,privatebase)
    @test pyconvert(Any,io.BytesIO()) isa PyIO
    @test pyconvert(PyIO,privatebase()) isa PyIO
    if pyis(pyimport("importlib.util").find_spec("pandas"),pybuiltins.None)
        @test_skip false # pandas is an optional development dependency
    else
        pd = pyimport("pandas")
        @test pyconvert(Any,pd.NA) === missing
        @test pytryconvert(Nothing,pd.NA) === Some(nothing)
        @test all(r.source=="pandas:NA.__class__" for r in pyconvert_rules(Any,pd.NA))
        df = pd.DataFrame()
        @test pyconvert(Any,df) isa PyPandasDataFrame
        @test count(r->r.source=="pandas:DataFrame",pyconvert_rules(Any,df)) == 1
        a = pd.array(pylist([1,2]);dtype="Int64")
        @test pyconvert(PyList,a) isa PyList
        @test count(r->r.source=="pandas.api.extensions:ExtensionArray",pyconvert_rules(PyList,a)) == 1
        @test !any(r->startswith(r.source,"pandas.core.") || startswith(r.source,"pandas._libs."),CV.PYCONVERT_RULES)
    end
end

@testitem "None source shorthand" begin
    using PythonCall
    CV = PythonCall.Convert
    x = pybuiltins.None
    pref = ("None"=>Missing,)
    @test pyis(CV.pyconvert_resolve("None"),pytype(x))
    @test pyconvert(Any,x;prefer=pref) === missing
    @test pytryconvert(Any,x;prefer=pref) === Some(missing)
    @test pyconvert(Nothing,x;prefer=pref) === nothing
    @test pyconvert(Any,pyint(1);prefer=pref) === 1
    ctx = PyConvertContext(;prefer=pref)
    @test pyconvert(ctx,Any,x) === missing
    @test isequal(pyconvert(ctx,Vector,pylist([x])),[missing])
    @test pyconvert(Any,x) === nothing
    id = pyconvert_add_rule("None",String,(_,_) -> "none")
    try
        @test pyconvert(String,x) == "none"
        @test only(r.source for r in pyconvert_rules(String,x) if r.id==id) == "types:NoneType"
        @test pyconvert_replace_rule(id,"None",String,(_,_) -> "replaced") == id
        @test pyconvert(String,x) == "replaced"
    finally
        pyconvert_remove_rule(id)
    end
end

@testitem "priority topological sort agrees with grouped sorting" begin
    using PythonCall
    CV = PythonCall.Convert
    # Enumerate DAGs with monotonic priorities; independent tie breaker reference.
    for mask in 0:63
        edges = falses(4,4)
        for (bit,(i,j)) in enumerate([(i,j) for i=1:4 for j=i+1:4])
            edges[i,j] = (mask >> (bit-1)) & 1 == 1
        end
        priorities = [0,10,20,0]
        for _=1:4, i=1:4, j=1:4
            edges[i,j] && (priorities[i]=max(priorities[i],priorities[j]))
        end
        grouped = Int[]
        for p in sort(unique(priorities);rev=true)
            group = findall(==(p),priorities)
            append!(grouped,group[CV.pyconvert_toposort(edges[group,group],zeros(Int,length(group)),group)])
        end
        @test grouped == CV.pyconvert_toposort(edges,priorities,collect(1:4))
    end
end
