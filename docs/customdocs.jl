module CustomDocs

import Base: Docs
import Documenter
import Documenter: DocSystem, Markdown, MarkdownAST, doccat
import Documenter.Selectors: order, matcher, runner

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

# help?> Documenter.doccat
#   Returns the category name of the provided Object.
doccat(::Docs.Binding, ::Type{CustomCat{cat}}) where {cat} = string(cat)

abstract type CustomDocExpander <: Documenter.Expanders.ExpanderPipeline end

order(::Type{CustomDocExpander}) = 20.0

function matcher(::Type{CustomDocExpander}, node, page, doc)
    return Documenter.iscode(node, "@customdoc")
end

# source:
# https://github.com/JuliaDocs/Documenter.jl/blob/7d3dc2ceef39a62edf2de7081e2d3aaf9be8d7c3/src/expander_pipeline.jl#L353
function runner(::Type{CustomDocExpander}, node, page, doc)
    @assert node.element isa MarkdownAST.CodeBlock

    block = node.element

    m = match(r"^(.+?)\s*-\s*(.+?)\s*(\n[\s\S]*)$", strip(block.code))

    @assert !isnothing(m) "Invalid header:\n$(block.code)"

    name = Symbol(m[1])
    cat = Symbol(m[2])
    body = strip(something(m[3], ""))

    binding = DocSystem.binding(Main, name)

    docsnodes = MarkdownAST.Node[]

    object = Documenter.Object(binding, CustomCat{cat})

    docstr = Markdown.MD[Markdown.parse(body)]
    results = Docs.DocStr[Docs.docstr(body, Dict{Symbol,Any}(:module => Main, :path => "", :linenumber => 0))]

    docsnode = Documenter.create_docsnode(docstr, results, object, page, doc)

    # Track the order of insertion of objects per-binding.
    push!(get!(doc.internal.bindings, binding, Documenter.Object[]), object)

    doc.internal.objects[object] = docsnode.element

    push!(docsnodes, docsnode)

    node.element = Documenter.DocsNodesBlock(block)
    
    push!(node.children, docsnode)

    return nothing
end

end # module CustomDocs