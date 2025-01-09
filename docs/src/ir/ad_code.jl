import Core.Compiler as CC
using ChainRules
using Core: SSAValue, GlobalRef, ReturnNode

# We also define a remap function which will be used to map old SSA values to new SSA values. 
remap(d, args::Tuple) = map(a -> remap(d,a), args) 
remap(d, args::Vector) = map(a -> remap(d,a), args) 
remap(d, r::ReturnNode) = ReturnNode(remap(d, r.val))
remap(d, x::SSAValue) = d[x] 
remap(d, x) = x

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

function construct_forward(ir, T = Any)
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
  new_ir = infer_ir!(new_ir)
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
  new_ir = ircode(reverse_inst, Any[Tuple{},P, R])
  new_ir = infer_ir!(new_ir)
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