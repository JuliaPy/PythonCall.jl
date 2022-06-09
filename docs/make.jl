using Documenter, PythonCall, Markdown

# This is a bit of a hack to let us insert documentation blocks with custom content.
# It means we can document Python things directly in the documentation source, and they
# are searchable.
#
# It's a hack because of the `doccat` overload, requiring a special kind of "signature"
# to embed the information we want.
#
# The first line is of the form "name - category", the rest is Markdown documentation.
# For example:
# ```@customdoc
# foo - Function
# Documentation for `foo`.
# ```
struct CustomCat{cat} end
Documenter.Utilities.doccat(::Base.Docs.Binding, ::Type{CustomCat{cat}}) where {cat} = string(cat)
struct CustomDocBlocks <: Documenter.Expanders.ExpanderPipeline end
Documenter.Expanders.Selectors.order(::Type{CustomDocBlocks}) = 20.0
Documenter.Expanders.Selectors.matcher(::Type{CustomDocBlocks}, node, page, doc) = Documenter.Expanders.iscode(node, "@customdoc")
Documenter.Expanders.Selectors.runner(::Type{CustomDocBlocks}, x, page, doc) = begin
    header, rest = split(x.code, "\n", limit=2)
    docstr = Markdown.parse(rest)
    name, cat = split(header, "-", limit=2)
    binding = Docs.Binding(Main, Symbol(strip(name)))
    object = Documenter.Utilities.Object(binding, CustomCat{Symbol(strip(cat))})
    slug = Documenter.Utilities.slugify(strip(name))
    anchor = Documenter.Anchors.add!(doc.internal.docs, object, slug, page.build)
    node = Documenter.Documents.DocsNode(docstr, anchor, object, page)
    page.mapping[x] = node
end

makedocs(
    sitename = "PythonCall & JuliaCall",
    modules = [PythonCall],
    pages = [
        "Home" => "index.md",
        "The Julia module PythonCall" => [
            "Guide" => "pythoncall.md",
            "Reference" => "pythoncall-reference.md",
        ],
        "The Python module JuliaCall" => [
            "Guide" => "juliacall.md",
            "Reference" => "juliacall-reference.md",
        ],
        "Conversion" => [
            "Julia to Python" => "conversion-to-python.md",
            "Python to Julia" => "conversion-to-julia.md",
        ],
        "compat.md",
        "pycall.md",
        "releasenotes.md",
    ]
)

deploydocs(
    repo = "github.com/cjdoris/PythonCall.jl.git",
)
