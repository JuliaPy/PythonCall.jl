using Dates

oldver = ARGS[1]
newver = ARGS[2]

function bump(file, oldpat, newpat)
    println("Bumping $file...")
    @assert oldpat != newpat
    oldtext = read(file, String)
    newtext = replace(oldtext, oldpat => newpat)
    @assert newtext != oldtext
    write(file, newtext)
end

function bumpver(file, pattern, oldver, newver)
    @assert oldver != newver
    oldpat = replace(pattern, "{}" => oldver)
    @assert oldpat != pattern
    newpat = replace(pattern, "{}" => newver)
    @assert newpat != pattern
    bump(file, oldver, newver)
end

bumpver("Project.toml", "version = \"{}\"\n", oldver, newver)
bumpver("setup.cfg", "version = {}\n", oldver, newver)
bumpver("pysrc/juliacall/__init__.py", "__version__ = '{}'\n", oldver, newver)
bumpver("pysrc/juliacall/juliapkg.json", "\"version\": \"={}\"", oldver, newver)
bumpver("pysrc/juliacall/juliapkg-dev.json", "\"version\": \"={}\"", oldver, newver)
bumpver("src/PythonCall.jl", "VERSION = v\"{}\"", oldver, newver)
bumpver("src/Core/Core.jl", "VERSION = v\"{}\"", oldver, newver)
bump("docs/src/releasenotes.md", "## Unreleased", "## $newver ($(today()))")
