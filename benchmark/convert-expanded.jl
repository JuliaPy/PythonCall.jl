# Shared harness for the pre-redesign revision and current v1.
# julia --project=benchmark benchmark/convert-expanded.jl after
# Optional second argument "warm" skips unchanged cold-path experiments.
using PythonCall, BenchmarkTools, SHA
const ExpandedConvert = PythonCall.Convert
const EXPANDED_NEW_API = isdefined(PythonCall, :PyConvertContext)

expanded_decline1(T,x) = ExpandedConvert.pyconvert_unconverted()
expanded_decline2(T,x) = ExpandedConvert.pyconvert_unconverted()
expanded_decline3(T,x) = ExpandedConvert.pyconvert_unconverted()
expanded_success(T,x) = ExpandedConvert.pyconvert_return(42)
struct ExpandedDestination{I} end
struct ExpandedScopedDecliner{I} <: Function end
(::ExpandedScopedDecliner)(T,x) = ExpandedConvert.pyconvert_unconverted()

function expanded_register(source,T,f;scope=T)
    if EXPANDED_NEW_API
        pyconvert_add_rule(source,T,f;scope)
    else
        pyconvert_add_rule(source,T,f)
    end
end
expanded_value(::Val{T},x) where T = pyconvert(T,x,nothing)
function expanded_case(name,::Type{T},x,valid) where T
    tag=Val(T)
    (;name, target=T, run=()->expanded_value(tag,x), valid)
end
function expanded_cache(T)
    if EXPANDED_NEW_API
        ExpandedConvert.pyconvert_rules_cache(typeof(PyConvertContext()),T)
    else
        ExpandedConvert.pyconvert_rules_cache(T)
    end
end
function expanded_invalidate!()
    if EXPANDED_NEW_API
        ExpandedConvert.pyconvert_invalidate!()
    else
        foreach(empty!,values(ExpandedConvert.PYCONVERT_RULES_CACHE))
    end
end

label=isempty(ARGS) ? "after" : first(ARGS)
warm_only=length(ARGS)>=2 && ARGS[2]=="warm"
ns=pyimport("types").ModuleType("convert_expanded_benchmark")
pyimport("sys").modules["convert_expanded_benchmark"]=ns
pyexec("""
import collections.abc
import array
import numpy
class Success: pass
class Declines: pass
class ThreeDeclines: pass
class Scoped: pass
class Preference: pass
class VirtualMapping: pass
collections.abc.Mapping.register(VirtualMapping)
class InterfaceOnly:
    def __init__(self, owner): self.owner = owner
    @property
    def __array_interface__(self): return self.owner.__array_interface__
class ArrayOnly:
    def __init__(self, owner): self.owner = owner
    def __array__(self): return self.owner
""",ns.__dict__)
prefix="convert_expanded_benchmark:"
expanded_register(prefix*"Success",Int,expanded_success;scope=Any)
expanded_register(prefix*"Declines",Int,expanded_decline1;scope=Any)
expanded_register(prefix*"Declines",Int,expanded_success;scope=Any)
for f in (expanded_decline1,expanded_decline2,expanded_decline3,expanded_success)
    expanded_register(prefix*"ThreeDeclines",Int,f;scope=Any)
end
for i=1:16
    # These rules legitimately decline for this value. Old v1 has no scope
    # filtering and must try them; the redesign excludes their owned scopes.
    expanded_register(prefix*"Scoped",Any,ExpandedScopedDecliner{i}();scope=ExpandedDestination{i})
end
expanded_register(prefix*"Scoped",Int,expanded_success;scope=Any)
expanded_register(prefix*"Preference",String,expanded_decline1;scope=Any)
expanded_register(prefix*"Preference",Int,expanded_success;scope=Any)
expanded_register("collections.abc:Mapping",Int,expanded_success)

buffer=ns.array.array("d",pylist(1:1000))
owner=ns.numpy.arange(1000)
interface=ns.InterfaceOnly(owner)
arrayonly=ns.ArrayOnly(owner)
@assert PythonCall.C.PyType_CheckBuffer(pytype(buffer))
@assert !any(pyhasattr(pytype(buffer),a) for a in ("__array_struct__","__array_interface__","__array__"))
@assert !PythonCall.C.PyType_CheckBuffer(pytype(interface))
@assert !PythonCall.C.PyType_CheckBuffer(pytype(arrayonly))
if EXPANDED_NEW_API
    eligible=filter(r->r.eligible,pyconvert_rules(Array,buffer))
    @assert first(eligible).source=="<buffer>"
else
    eligible=filter(r->typeintersect(r.type,Array)!==Union{},
                    ExpandedConvert._pyconvert_get_rules(pytype(buffer)))
    @assert first(eligible).func===PythonCall.Wrap.pyconvert_rule_array
end
validint=x->x===42
validarray=x->x isa Array && size(x)==(1000,) && eltype(x)===Int64 && first(x)==0 && last(x)==999
validbuffer=x->x isa Array && size(x)==(1000,) && eltype(x)===Float64 && first(x)==1.0 && last(x)==1000.0
validwrapper=x->x isa PyArray && size(x)==(1000,) && first(x)==0 && last(x)==999
validbufferwrapper=x->x isa PyArray && size(x)==(1000,) && first(x)==1.0 && last(x)==1000.0
cases=[
    expanded_case("single_rule",Int,ns.Success(),validint),
    expanded_case("decline_then_success",Int,ns.Declines(),validint),
    expanded_case("three_declines_then_success",Int,ns.ThreeDeclines(),validint),
    expanded_case("sixteen_scoped_decliners",Int,ns.Scoped(),validint),
    expanded_case("virtual_mapping_abc",Int,ns.VirtualMapping(),validint),
    expanded_case("buffer_only_native",Array,buffer,validbuffer),
    expanded_case("buffer_only_wrapper",PyArray,buffer,validbufferwrapper),
    expanded_case("interface_only_native",Array,interface,validarray),
    expanded_case("array_method_only_native",Array,arrayonly,validarray),
    expanded_case("ndarray_native",Array,owner,validarray),
    expanded_case("ndarray_wrapper",PyArray,owner,validwrapper),
]
# Verify the dispatch fixtures themselves, outside the measured regions.
for (i,expected) in enumerate(EXPANDED_NEW_API ? (1,2,4,1) : (1,2,4,17))
    case=cases[i]
    # The first fixtures are distinct Python classes, each with the indicated
    # number of eligible rules. This also detects accidental rule deduplication.
    t=(ns.Success,ns.Declines,ns.ThreeDeclines,ns.Scoped)[i]
    if EXPANDED_NEW_API
        @assert count(r->r.eligible,pyconvert_rules(Int,t()))==expected case.name
    else
        @assert length(ExpandedConvert.pyconvert_get_rules(Int,t))==expected case.name
    end
end
preference_names=("two_matching_first_success","two_matching_first_declines",
                  "two_matching_decline_then_ordinary")
preference_cases=if EXPANDED_NEW_API
    x=ns.Preference()
    source=prefix*"Preference"
    [(;name=preference_names[1],run=()->pyconvert(Any,x;prefer=(source=>Int,source=>String))),
     (;name=preference_names[2],run=()->pyconvert(Any,x;prefer=(source=>String,source=>Int))),
     (;name=preference_names[3],run=()->pyconvert(Any,x;prefer=(source=>String,source=>Float64)))]
else
    []
end

open(joinpath(@__DIR__,"convert-expanded-$label.tsv"),"w") do io
    println(io,"case\tmode\tstatus\tmin_ns\tmedian_ns\tbytes\tallocs")
    for case in cases
        f=case.run
        @assert case.valid(f()) case.name
        cache=expanded_cache(case.target)
        for mode in (warm_only ? ("warm",) : ("warm","selection_miss","invalidated"))
            trial=if mode=="warm"
                @benchmark $f() seconds=1.0 samples=5000
            elseif mode=="selection_miss"
                @benchmark $f() setup=(empty!($cache)) seconds=0.4 samples=200 evals=1
            else
                @benchmark $f() setup=(expanded_invalidate!()) seconds=0.4 samples=40 evals=1
            end
            m,d=minimum(trial),median(trial)
            println(io,"$(case.name)\t$mode\tmeasured\t$(m.time)\t$(d.time)\t$(m.memory)\t$(m.allocs)")
        end
        flush(io)
    end
    for case in preference_cases
        f=case.run
        @assert f()===42 case.name
        trial=@benchmark $f() seconds=1.0 samples=5000
        m,d=minimum(trial),median(trial)
        println(io,"$(case.name)\twarm\tmeasured\t$(m.time)\t$(d.time)\t$(m.memory)\t$(m.allocs)")
    end
    if !EXPANDED_NEW_API
        for name in preference_names
            println(io,"$name\twarm\tunsupported\tNA\tNA\tNA\tNA")
        end
    end
end
open(joinpath(@__DIR__,"convert-expanded-$label-environment.txt"),"w") do io
    println(io,"Julia: ",VERSION)
    println(io,"Python: ",pyconvert(String,pyimport("sys").version))
    println(io,"Python executable: ",pyconvert(String,pyimport("sys").executable))
    println(io,"NumPy: ",pyconvert(String,ns.numpy.__version__))
    println(io,"BenchmarkTools: ",pkgversion(BenchmarkTools))
    println(io,"PythonCall source: ",pathof(PythonCall))
    println(io,"New API: ",EXPANDED_NEW_API)
    for file in ("Convert/pyconvert.jl","Convert/rules.jl","Wrap/PyArray.jl")
        println(io,file," SHA256: ",bytes2hex(sha256(read(joinpath(dirname(pathof(PythonCall)),file)))))
    end
end
