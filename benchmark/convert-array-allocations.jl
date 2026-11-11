# Attribution only: run after timing trials, with the same environment/Python.
using PythonCall, Profile
array_value(::Val{T},x) where T = pyconvert(T,x,nothing)
label=isempty(ARGS) ? "after" : only(ARGS)
source_root=dirname(pathof(PythonCall))
np=pyimport("numpy")
array=pyimport("array")
cases=[("ndarray_native",Array,np.arange(1000)),
       ("ndarray_wrapper",PyArray,np.arange(1000)),
       ("buffer_native",Array,array.array("d",pylist(1:1000))),
       ("buffer_wrapper",PyArray,array.array("d",pylist(1:1000)))]
open(joinpath(@__DIR__,"convert-array-allocations-$label.txt"),"w") do io
    println(io,"Julia: ",VERSION,"; PythonCall source: ",pathof(PythonCall))
    for (name,T,x) in cases
        f=let tag=Val(T), x=x
            ()->array_value(tag,x)
        end
        for _=1:3
            f()
        end
        GC.gc()
        Profile.Allocs.clear()
        Profile.Allocs.@profile sample_rate=1.0 f()
        println(io,"\n",name)
        for a in Profile.Allocs.fetch().allocs
            println(io,a.type,": ",a.size," bytes")
            for frame in a.stacktrace
                file=string(frame.file)
                if occursin(source_root,file)
                    println(io,"  ",frame)
                end
            end
        end
    end
end
