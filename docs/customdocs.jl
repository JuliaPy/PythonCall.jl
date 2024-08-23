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

    name, cat, body = _parse_docs(block.code)

    binding = DocSystem.binding(Main, name)

    docsnodes = MarkdownAST.Node[]

    object = Documenter.Object(binding, CustomCat{cat})

    docstr = Markdown.parse(body)::Markdown.MD
    result = Docs.docstr(
        body,
        Dict{Symbol,Any}(    # NOTE: Not sure about what to put here.                            
            :module => Main, # This is supposed to be tracking python code.
            :path => "",
            :linenumber => 0,
        ),
    )::Docs.DocStr

    # NOTE: This was modified because the original Documenter.create_docsnode was generating unreachable links
    # Also, the original implementation required docstr, result to be vectors.

    docsnode = _create_docsnode(docstr, result, object, page, doc)

    # Track the order of insertion of objects per-binding.
    push!(get!(doc.internal.bindings, binding, Documenter.Object[]), object)

    doc.internal.objects[object] = docsnode.element

    push!(docsnodes, docsnode)

    node.element = Documenter.DocsNodesBlock(block)

    push!(node.children, docsnode)

    return nothing
end

function _parse_docs(code::AbstractString)
    m = match(r"^(.+?)\s*-\s*(.+?)\s*(\n[\s\S]*)$", strip(code))

    if isnothing(m)
        error("""
              Invalid docstring:
              $(code)
              """)
    end

    name = Symbol(m[1])
    cat = Symbol(m[2])
    body = strip(something(m[3], ""))

    return (name, cat, body)
end

# source:
# https://github.com/JuliaDocs/Documenter.jl/blob/7d3dc2ceef39a62edf2de7081e2d3aaf9be8d7c3/src/expander_pipeline.jl#L959-L960
function _create_docsnode(docstring, result, object, page, doc)
    # Generate a unique name to be used in anchors and links for the docstring.
    # NOTE: The way this is being slugified is causing problems:
    # slug = Documenter.slugify(object) 
    slug = Documenter.slugify(string(object.binding))

    anchor = Documenter.anchor_add!(doc.internal.docs, object, slug, page.build)
    docsnode = Documenter.DocsNode(anchor, object, page)

    # Convert docstring to MarkdownAST, convert Heading elements, and push to DocsNode

    ast = convert(MarkdownAST.Node, docstring)

    doc.user.highlightsig && Documenter.highlightsig!(ast)

    # The following 'for' corresponds to the old dropheaders() function
    for headingnode in ast.children
        headingnode.element isa MarkdownAST.Heading || continue

        boldnode = MarkdownAST.Node(MarkdownAST.Strong())

        for textnode in collect(headingnode.children)
            push!(boldnode.children, textnode)
        end

        headingnode.element = MarkdownAST.Paragraph()

        push!(headingnode.children, boldnode)
    end

    push!(docsnode.mdasts, ast)
    push!(docsnode.results, result)
    push!(docsnode.metas, docstring.meta)

    return MarkdownAST.Node(docsnode)
end

end # module CustomDocs
