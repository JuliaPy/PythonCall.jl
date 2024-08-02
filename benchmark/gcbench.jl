module GCBench

using PythonCall

function append_lots(iters=100 * 1024, size=1596)
    v = pylist()
    for i = 1:iters
        v.append(pylist(rand(size)))
    end
    return v
end

end
