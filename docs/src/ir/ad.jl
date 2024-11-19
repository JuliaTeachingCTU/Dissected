# ## Source-to-Source Automatic Differentiation with IRCode
# 
# The intention of this document is to show the reader the main 
# mechanisms and building blocks of source-to-source differentiation 
# in Julia relying on IRCode. The second purpose is to show manipulation
# of IR representation and possibly lower bariers of people to investigate it.
# Note that the emphasis is put on the didactic level to illustrate the mechanisms
# rather than on full functionality. Nevertheless the project is using ChainRules.jl
# to provide rules.  For fully flegged source-to-source AD built on 
# similar ideas look at Mooncake.jl or Enzyme.jl. This project is inspired by 
# Petite Zygote. Many ideas are taken from Mooncake.jl. 
# 
# The project is using Julia's internals, which are not part of the official interface
# and are therefore subject to change. The project was developed using Julia 1.11.1.
# The project is not intended for production use. 

# ## Introduction
# The overall idea behind the construction is as follows.
# We assume to be given a function which we want to differentiate and for 
# which there is no `rrule`. We therefore use Julia compiler to provide as 
# with lowered and typed code (note that Zygote used code before typing).
# This has the advantage that the code is in static single assignment form,
# stripped from all syntactic sugar, and it is typed. We can also ask compiler
# to provide certain optimization like SROA, inlining, constant propagation and
# dead code elimination. This means that we can start with relatively lean code.

# For a given function, we automatically generate a function (functor) which will behave 
# like `rrule`. That means that it will take the same arguments, as the original
# function, but the output will be the evaluation of the function and the `pullback`.
# This `rrule` will be generated autotomatically from IRCode of the function to differentiate 
# as follows:
#   1. In the forward, we will replace each call to the function with a call to `rrule`. Since `rrule`
#       return tuple (the evaluation of the function and the pullback), we extract the output 
#       of the original function and store the pullback.
#   2. We will construct the pullback. We will iterate over the pullbacks we store from the forward
#       in reverse order. Each pullback returns gradient with respect to the arguments, which need to 
#      be correctly accumulate it.
#  This is the core of a simple AD without any conditionals and loops. The rest is some plumbing.


# ## Simple AD without control flow

# Before we start, we will define few functions which are needed to interact with IRCode.
# As always, we start by importing few libraries.
import Core.Compiler as CC
using ChainRules
using Core: SSAValue, GlobalRef, ReturnNode

# Let' start by defining a simple function to differentate.

function foo(x,y) 
  z = x * y 
  z + sin(x)
end

# The first step is to get IRCode of the function. We can do this by calling `code_ircode` from Base.
# Note that when calling `Base.code_ircode` we need to provide the types of the arguments, not their values.
# This is because the compiler does not care about values, but about types.

ir, _ = only(Base.code_ircode(foo, (Float64, Float64); optimize_until = "compact 1"))

# The returned IRCode will looks like this.
# ```julia
# 2 1 ─ %1 = (_2 * _3)::Float64                                                                                                      │
# 3 │   %2 = Main.sin(_2)::Float64                                                                                                   │
#   │   %3 = (%1 + %2)::Float64                                                                                                      │
#   └──      return %3                                                                                                               │
#    => Float64
# ```
# Recall that the forward part needs to replace each call of the function by a call to `rrule` and stode pullbacks.
# So in pseudocode, we want something like 
# ```julia
# (%1, %2) = rrule(*, _2, _3)
# (%3, %4) = rrule(Main.sin, _2)
# (%5, %6) = rrule(Base._, %1, %3)
# return(%5, tuple(%2, %4,%6))
# ```
# In the above pseudocode, `%1, %3, %5` are ouputs of function `*, sin, +` respectively, and `%2, %4, %6` 
# are their pullbacks. The function therefore return the correct value and information for the pullback.
# We emphasize that the above is not a valid IRCode, since SSA for allows to assigne only one variable. But
# for dydactic purposes is fine. 

# To implement the code performing the above transformation, we initiate few variables
pullbacks = []          # storage for pullbacks
new_insts = Any[]       # storate for instructions
new_line = Int32[]      # Index of instruction we are differentiating
ssamap = Dict{SSAValue,SSAValue}() # this maps old SSA values to new SSA values, since they need to be linearly ordered.

  for (i, stmt) in enumerate(ir.stmts)
   inst = stmt[:inst]
   if inst isa Expr && inst.head == :call
      new_inst = Expr(:call, GlobalRef(ChainRules, :rrule), remap(ssamap, inst.args)...)
      push!(new_insts, new_inst)
      push!(new_line, stmt[:line])
      rrule_ssa = SSAValue(length(new_insts))


      push!(new_insts, Expr(:call, GlobalRef(Base, :getindex), rrule_ssa, 1))
      push!(new_line, stmt[:line])
      val_ssa = SSAValue(length(new_insts))
      ssamap[SSAValue(i)] = val_ssa

      push!(new_insts, Expr(:call, GlobalRef(Base, :getindex), rrule_ssa, 2))
      pullback_ssa = SSAValue(length(new_insts))
      push!(new_line, stmt[:line])
      push!(pullbacks, (;old_ssa = i, inst = inst, pullback_ssa))
      continue
   end

   if inst isa ReturnNode
      push!(new_insts, Expr(:call, GlobalRef(Main, :Pullback), map(x -> x[end], pullbacks)...))
      pullback_ssa = SSAValue(length(new_insts))
      push!(new_line, stmt[:line])

      # construct returned tuple
      push!(new_insts, Expr(:call, GlobalRef(Base, :tuple), remap(ssamap, inst.val), pullback_ssa))
      returned_tuple = SSAValue(length(new_insts))
      push!(new_line, stmt[:line])

      push!(new_insts, ReturnNode(returned_tuple))
      push!(new_line, stmt[:line])
      continue
     end
   error("unknown node $(i)")
  end

  # this nightmare construct the IRCode with absolutely useless type information
  argtypes = Any[Tuple{}, ir.argtypes[2:end]...]
  new_ir = ircode(new_insts, argtypes)
  # new_ir = infer_ir!(new_ir)
  (new_ir, pullbacks)


function ircode(
    insts::Vector{Any}, argtypes::Vector{Any}, sptypes::Vector{CC.VarState}=CC.VarState[]
)
    cfg = CC.compute_basic_blocks(insts)
    # insts = __line_numbers_to_block_numbers!(insts, cfg)
    stmts = __insts_to_instruction_stream(insts)
    linetable = [CC.LineInfoNode(Main, :ircode, :ir_utils, Int32(1), Int32(0))]
    meta = Expr[]
    return CC.IRCode(stmts, cfg, linetable, argtypes, meta, CC.VarState[])
end

function __insts_to_instruction_stream(insts::Vector{Any})
    return CC.InstructionStream(
        insts,
        fill(Any, length(insts)),
        fill(CC.NoCallInfo(), length(insts)),
        fill(Int32(1), length(insts)),
        fill(CC.IR_FLAG_REFINED, length(insts)),
    )
end

function infer_ir!(ir::CC.IRCode)
    return __infer_ir!(ir, CC.NativeInterpreter(), __get_toplevel_mi_from_ir(ir, Main))
end

# Given some IR, generates a MethodInstance suitable for passing to infer_ir!, if you don't
# already have one with the right argument types. Credit to @oxinabox:
# https://gist.github.com/oxinabox/cdcffc1392f91a2f6d80b2524726d802#file-example-jl-L54
_type(x::Type) = x
_type(x::CC.Const) = _typeof(x.val)
_type(x::CC.PartialStruct) = x.typ
_type(x::CC.Conditional) = Union{_type(x.thentype), _type(x.elsetype)}
_type(::CC.PartialTypeVar) = TypeVar

function __get_toplevel_mi_from_ir(ir, _module::Module)
    mi = ccall(:jl_new_method_instance_uninit, Ref{Core.MethodInstance}, ());
    mi.specTypes = Tuple{map(_type, ir.argtypes)...}
    mi.def = _module
    return mi
end

# Run type inference and constant propagation on the ir. Credit to @oxinabox:
# https://gist.github.com/oxinabox/cdcffc1392f91a2f6d80b2524726d802#file-example-jl-L54
function __infer_ir!(ir, interp::CC.AbstractInterpreter, mi::CC.MethodInstance)
    method_info = CC.MethodInfo(#=propagate_inbounds=#true, nothing)
    min_world = world = CC.get_inference_world(interp)
    max_world = Base.get_world_counter()
    irsv = CC.IRInterpretationState(
        interp, method_info, ir, mi, ir.argtypes, world, min_world, max_world
    );
    rt = CC._ir_abstract_constant_propagation(interp, irsv)
    return ir
end


struct Pullback{T<:Tuple}
  data::T
end

Pullback(args...) = Pullback(tuple(args...))
Base.getindex(p::Pullback, i) = p.data[i]

remap(d, args::Tuple) = map(a -> remap(d,a), args) 
remap(d, args::Vector) = map(a -> remap(d,a), args) 
remap(d, r::ReturnNode) = ReturnNode(remap(d, r.val))
remap(d, x::SSAValue) = d[x] 
remap(d, x) = x


function forward(ir, T = Any)
  pullbacks = []
  new_insts = Any[]
  new_line = Int32[]
  ssamap = Dict{SSAValue,SSAValue}()
  for (i, stmt) in enumerate(ir.stmts)
   inst = stmt[:inst]
   if inst isa Expr && inst.head == :call
      new_inst = Expr(:call, GlobalRef(ChainRules, :rrule), remap(ssamap, inst.args)...)
      push!(new_insts, new_inst)
      push!(new_line, stmt[:line])
      rrule_ssa = SSAValue(length(new_insts))


      push!(new_insts, Expr(:call, GlobalRef(Base, :getindex), rrule_ssa, 1))
      push!(new_line, stmt[:line])
      val_ssa = SSAValue(length(new_insts))
      ssamap[SSAValue(i)] = val_ssa

      push!(new_insts, Expr(:call, GlobalRef(Base, :getindex), rrule_ssa, 2))
      pullback_ssa = SSAValue(length(new_insts))
      push!(new_line, stmt[:line])
      push!(pullbacks, (;old_ssa = i, inst = inst, pullback_ssa))
      continue
   end

   if inst isa ReturnNode
      push!(new_insts, Expr(:call, GlobalRef(Main, :Pullback), map(x -> x[end], pullbacks)...))
      pullback_ssa = SSAValue(length(new_insts))
      push!(new_line, stmt[:line])

      # construct returned tuple
      push!(new_insts, Expr(:call, GlobalRef(Base, :tuple), remap(ssamap, inst.val), pullback_ssa))
      returned_tuple = SSAValue(length(new_insts))
      push!(new_line, stmt[:line])

      push!(new_insts, ReturnNode(returned_tuple))
      push!(new_line, stmt[:line])
      continue
     end
   error("unknown node $(i)")
  end

  # this nightmare construct the IRCode with absolutely useless type information
  argtypes = Any[Tuple{}, ir.argtypes[2:end]...]
  new_ir = ircode(new_insts, argtypes)
  # new_ir = infer_ir!(new_ir)
  (new_ir, pullbacks)
end

function construct_pullback(pullbacks, ::Type{<:Tuple{R,P}}) where {R,P}
  diffmap = Dict{Any,Any}() # this will hold the mapping where is the gradient with respect to SSA.
  # the argument of the pullback we are defining is a gradient with respect to the argument of return
  # which we assume to be the last of insturction in `inst`
  
  diffmap[SSAValue(length(pullbacks))] = Core.Argument(3)

  reverse_inst = []
  # now we iterate over pullbacks and execute one by one with correct argument
  for pull_id in reverse(axes(pullbacks,1))
    ssa_no, inst, _ = pullbacks[pull_id]

    # first we extract the pullback from a tuple of pullbacks
    push!(reverse_inst, Expr(:call, GlobalRef(Base, :getindex), Core.Argument(2), pull_id))
    
    #then we call the pullback with a correct argument
    push!(reverse_inst, Expr(:call, SSAValue(length(reverse_inst)), diffmap[SSAValue(ssa_no)]))
    arg_grad = SSAValue(length(reverse_inst))

    # then we extract gradients with respect to the argument of the instruction and 
    # record all the calls
    for (i, a) in enumerate(inst.args)
      i == 1 && continue # we omit gradient with respect to the name of the function and rrule
      if haskey(diffmap, a)  # we need to perform addition
        push!(reverse_inst, Expr(:call, GlobalRef(Base, :getindex), arg_grad, i))
        new_val = SSAValue(length(reverse_inst))
        old_val = diffmap[a]
        push!(reverse_inst, Expr(:call, GlobalRef(Base, :+), old_val, new_val))
        diffmap[a] = SSAValue(length(reverse_inst))
      else
        push!(reverse_inst, Expr(:call, GlobalRef(Base, :getindex), arg_grad, i))
        diffmap[a] = SSAValue(length(reverse_inst))
      end
    end
  end

  # we create a Tuple with return values
  ∇args = collect(filter(x -> x isa Core.Argument, keys(diffmap)) )
  sort!(∇args, by = x -> x.n)
  push!(reverse_inst, Expr(:call, GlobalRef(Base, :tuple), [diffmap[a] for a in ∇args]...))
  returned_tuple = SSAValue(length(reverse_inst))
  push!(reverse_inst, ReturnNode(returned_tuple))

  ircode(reverse_inst, Any[Tuple{},P, R])
end


struct CachedGrad{F, R}
    foc::F
    roc::R
end

function gradient(f::CachedGrad, args...)
    v, pull_struct = f.foc(args...)
    v, f.roc(pull_struct, one(eltype(v)))
end

totype(x::DataType) = x
totype(x::Type) = x
totype(x) = typeof(x)

function prepare_gradient(f, args...)
    args = tuple(map(totype, args)...)
    ir, _ = only(Base.code_ircode(f, args; optimize_until = "compact 1"))
    forward_ir, pullbacks = forward(ir)
    rt = Base.Experimental.compute_ir_rettype(forward_ir)
    reverse_ir = construct_pullback(pullbacks, rt)
    infer_ir!(reverse_ir)
   
    foc = Core.OpaqueClosure(forward_ir; do_compile = true)
    roc = Core.OpaqueClosure(reverse_ir; do_compile = true)
    CachedGrad(foc, roc)
end

function foo(x,y) 
  z = x * y 
  z + sin(x)
end

bar(x) = 5 * x 


# handling phi nodes
leakyrelu(x) = x > 0 ? x : 0.01 * x

function poorpow(x::Float64, n)
    r = 1.0
    while n > 0
        n -= 1
        r *= x
    end
    return r
end

function uselesspow(x::Float64, n)
    r = 1.0
    while r > n
        r *= x
    end
    return r
end

function test()

    cached = prepare_gradient(foo, 1.0, 1.0)
    gradient(cached, 1.0, 1.0)


    cached = prepare_gradient(bar, 1.0)
    gradient(cached, 1.0)


    # handling phi nodes


    prepare_gradient(leakyrelu, 1)
end