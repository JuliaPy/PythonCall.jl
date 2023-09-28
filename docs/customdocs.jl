module CustomDocs

import Base: Docs
import Documenter
import Documenter: Markdown, MarkdownAST, doccat
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

function runner(::Type{CustomDocExpander}, node, page, doc)
    block = node.element

    header, body = split(block.code, "\n", limit=2)

    docstring = Markdown.parse(body)

    name, cat = split(header, "-", limit=2)

    binding = Docs.Binding(Main, Symbol(strip(name)))

    object = Documenter.Object(binding, CustomCat{Symbol(strip(cat))})

    # source:
    # https://github.com/JuliaDocs/Documenter.jl/blob/7d3dc2ceef39a62edf2de7081e2d3aaf9be8d7c3/src/expander_pipeline.jl#L959

    # Generate a unique name to be used in anchors and links for the docstring.
    slug = Documenter.slugify(object)
    anchor = Documenter.anchor_add!(doc.internal.docs, object, slug, page.build)
    docsnode = Documenter.DocsNode(anchor, object, page)

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
    # push!(docsnode.results, result)
    push!(docsnode.metas, docstring.meta)

    node.element = docsnode
end

end # module CustomDocs