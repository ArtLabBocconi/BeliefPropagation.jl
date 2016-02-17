module OO

using Compat
using Base.Meta

export @oo

_dosub!(args::Array{Any}, subs::Dict) = map!(a->_dosub!(a, subs), args)
function _dosub!(a::Expr, subs::Dict)
    #show_sexpr(a); println()
    isexpr(a, [:line, :quote, :local, :global]) && return a
    map!(a->_dosub!(a, subs), a.args)
    return a
end
#_dosub!(a::Symbol, subs::Dict) = (println("a=$a subs=$subs"); get(subs, a, a))
_dosub!(a::Symbol, subs::Dict) = get(subs, a, a)
_dosub!(a, subs::Dict) = a

function _oo_auto(fex)
    @assert isexpr(fex, [:function, :(=)])
    @assert isexpr(fex.args[1], :call)
    @assert length(fex.args) == 2
    subrules = Dict()
    for a in fex.args[1].args[2:end]
        if isa(a, Symbol)
            subrules[a] = a
        end
    end
    for a in fex.args[1].args[2:end]
        if isexpr(a, :(::)) && Base.isleaftype(eval(current_module(), a.args[2]))
            for n in fieldnames(eval(current_module(), a.args[2]))
                if haskey(subrules, n)
                    subrules[n] == n && continue
                    error("duplicate name $n")
                end
                subrules[n] = Expr(:(.), a.args[1], QuoteNode(n))
            end
        else
            @assert isa(a, Symbol)
        end
    end
    _dosub!(fex.args[2], subrules)
    fex
end

function _oo_first(fex)
    @assert isexpr(fex, [:function, :(=)])
    @assert isexpr(fex.args[1], :call)
    @assert length(fex.args) == 2
    subrules = Dict()
    allargs = fex.args[1].args[2:end]
    length(allargs) > 0 || error("@oo: at least one function argument required")
    for a in allargs[2:end]
        if isa(a, Symbol)
            an = a
        else
            @assert isexpr(a, :(::))
            an = a.args[2]
        end
        @assert isa(an, Symbol)
        subrules[an] = an
    end
    a = allargs[1]
    isexpr(a, :(::)) || error("@oo: first argument must be type-annotated")
    Base.isleaftype(eval(current_module(), a.args[2])) || error("@oo: first argument must be composite")
    for n in fieldnames(eval(current_module(), a.args[2]))
        haskey(subrules, n) && continue
        subrules[n] = Expr(:(.), a.args[1], QuoteNode(n))
    end
    _dosub!(fex.args[2], subrules)
    fex
end

macro oo(args...)
    if length(args) == 1
        return esc(_oo_first(args[1]))
    elseif length(args) > 2
        error("@oo must be called with 1 or 2 arguments")
    else
        args[1] == :auto && return esc(_oo_auto(args[2]))
        args[1] == :method && return esc(_oo_first(args[2]))
        error("@oo first argument must be either a function definition, or \"auto\" or \"method\"")
    end
end

end
