# The idea behind the construction is as follows.
# The core part will implement the construction of
# rrule, which means that it returns forward and reverse functions.
# The functions will be actually functors wrapped in a struct, which 
# will make it nicely callable. 
# Any captured variables will be stored in the struct, which can be passed
# to the pullback, which can then pass it as an additional argument to the 
# pullback.


import Core.Compiler as CC
using ChainRules
using Core: SSAValue, GlobalRef, ReturnNode

function foo(x,y) 
  z = x * y 
  z + sin(x)
end

(ir, rt) = only(Base.code_ircode(foo, (Float64, Float64), optimize_until = "compact 1"))

struct Pullback{T}
  data::T
end

Pullback(args...) = Pullback(tuple(args...))


argtype(ir::CC.IRCode, a::Core.Argument) = ir.argtypes[a.n]
argtype(ir::CC.IRCode, a::Core.SSAValue) = ir.stmts.type[a.id]

"""
  type_of_pullback(ir, inst)

  infer type of the pullback
"""
function type_of_pullback(ir, inst, optimize_until = "compact 1")
  inst.head != :call && error("inferrin return type of calls is supported")
  params = tuple([argtype(ir, inst.args[i]) for i in 2:length(inst.args)]...)
  (ir, rt) = only(Base.code_ircode(inst.args[1], params, optimize_until))
  rt
end

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


      push!(new_insts, Expr(:call, :getindex, rrule_ssa, 1))
      push!(new_line, stmt[:line])
      val_ssa = SSAValue(length(new_insts))
      ssamap[SSAValue(i)] = val_ssa

      push!(new_insts, Expr(:call, :getindex, rrule_ssa, 2))
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
      push!(new_insts, Expr(:call, :tuple, remap(ssamap, inst.val), pullback_ssa))
      returned_tuple = SSAValue(length(new_insts))
      push!(new_line, stmt[:line])

      push!(new_insts, ReturnNode(returned_tuple))
      push!(new_line, stmt[:line])
      continue
     end
   error("unknown node $(i)")
  end

  # this nightmare construct the IRCode with absolutely useless type information
  is = CC.InstructionStream(
   new_insts,                               # inst::Vector{Any}
   fill(Any, length(new_insts)),            # type::Vector{Any}
   fill(CC.NoCallInfo(), length(new_insts)),   # info::Vector{CallInfo}
   new_line,                               # line::Vector{Int32}
   fill(UInt8(0), length(new_insts)),       # flag::Vector{UInt8}
  )
  cfg = CC.CFG([CC.BasicBlock(CC.StmtRange(1,length(new_insts)))], Int64[])
  new_ir = CC.IRCode(is, cfg, ir.linetable, ir.argtypes, ir.meta, ir.sptypes)
  (new_ir, pullbacks)
end

function construct_pullback(pullbacks)
  diffmap = Dict{Any,Any}() # this will hold the mapping where is the gradient with respect to SSA.
  # the argument of the pullback we are defining is a gradient with respect to the argument of return
  # which we assume to be the last of insturction in `inst`
  
  diffmap[SSAValue(length(pullbacks))] = Core.Argument(3)

  reverse_inst = []
  # now we iterate over pullbacks and execute one by one with correct argument
  for pull_id in reverse(axes(pullbacks,1))
    ssa_no, inst, _ = pullbacks[pull_id]

    # first we extract the pullback from a tuple of pullbacks
    push!(reverse_inst, Expr(:call, :getindex, Core.Argument(2), pull_id))
    
    #then we call the pullback with a correct argument
    push!(reverse_inst, Expr(:call, SSAValue(length(reverse_inst)), diffmap[SSAValue(ssa_no)]))
    arg_grad = SSAValue(length(reverse_inst))

    # then we extract gradients with respect to the argument of the instruction and 
    # record all the calls
    for (i, a) in enumerate(inst.args)
      i == 1 && continue # we omit gradient with respect to the name of the function and rrule
      if haskey(diffmap, a)  # we need to perform addition
        push!(reverse_inst, Expr(:call, :getindex, arg_grad, i))
        new_val = SSAValue(length(reverse_inst))
        old_val = diffmap[a]
        push!(reverse_inst, Expr(:call, :+, old_val, new_val))
        diffmap[a] = SSAValue(length(reverse_inst))
      else
        push!(reverse_inst, Expr(:call, :getindex, arg_grad, i))
        diffmap[a] = SSAValue(length(reverse_inst))
      end
    end
  end

  # we create a Tuple with return values
  ∇args = collect(filter(x -> x isa Core.Argument, keys(diffmap)) )
  sort!(∇args, by = x -> x.n)
  push!(reverse_inst, Expr(:call, :tuple, [diffmap[a] for a in ∇args]...))
  returned_tuple = SSAValue(length(reverse_inst))
  push!(reverse_inst, ReturnNode(returned_tuple))

  new_line = fill(0, length(reverse_inst))
  # this nightmare construct the IRCode with absolutely useless type information
  is = CC.InstructionStream(
    reverse_inst,                               # inst::Vector{Any}
    fill(Any, length(reverse_inst)),            # type::Vector{Any}
    fill(CC.NoCallInfo(), length(reverse_inst)),   # info::Vector{CallInfo}
    new_line,                               # line::Vector{Int32}
    fill(UInt8(0), length(reverse_inst)),       # flag::Vector{UInt8}
  )
  cfg = CC.CFG([CC.BasicBlock(CC.StmtRange(1,length(reverse_inst)))], Int64[])
  new_ir = CC.IRCode(is, cfg, ir.linetable, ir.argtypes, ir.meta, ir.sptypes)
  new_ir
end

forward_ir, pullbacks = forward(ir)
reverse_ir = construct_pullback(pullbacks)

v, pull_struct = Core.OpaqueClosure(forward_ir)(1.0,1.0)

oc = Core.OpaqueClosure(reverse_ir)
oc(pull_struct, 1.0)


