# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

using MXNet, Formatting

import Base: getindex


function fc_layer(name, X, n_hidden, act_type, pdropout)
  X = mx.FullyConnected(X, num_hidden=Int(n_hidden), name="$(name)")
  #X = mx.BatchNorm(X)
  X = mx.Activation(X, act_type=act_type, name="$(name)_act")
  return mx.Dropout(X, p=pdropout, name="$(name)_dropout")
end

function conv_layer(name, X, num_filter, filter_length, act_type, pool_size, pool_type, dropout)
  pad = Int(floor(0.5*(filter_length - 1)))
  X = mx.Convolution(X, kernel=(1, filter_length), pad=(0, pad), stride=(1, 1) , num_filter=num_filter, name="$name") # 2D->3D
  #X = mx.BatchNorm(X)
  X = mx.Activation(X, act_type=act_type, name="$(name)_act")
  if pool_type != nothing && pool_size != nothing
      X = mx.Pooling(X, kernel=(1, pool_size), stride=(1, pool_size), pool_type=pool_type, name="$(name)_pool") # 256->64
  end
  X = mx.Dropout(X, p=dropout, name="$(name)_dropout")
  return X
end

function deconv_layer(name, X, num_filter, filter_length, act_type, pool_size, dropout)
  pad = max(Int( floor(filter_length/2) - pool_size/2 ),0) # 2
  if pad < 0
    throw(ArgumentError("upsampling not possible because padding is negative ($pad). Increase filter length or decrease pool size."))
  elseif pad == 0
    deconv_filter = 4  # minimum filter length that can be applied with this configuration
  else
    deconv_filter = filter_length-1
  end
        
  X = mx.Deconvolution(X, kernel=(1, deconv_filter), stride=(1, pool_size), pad=(0, pad), num_filter=num_filter, name="$name")
  #X = mx.BatchNorm(X)
  if act_type != nothing
    X = mx.Activation(X, act_type=act_type, name="$(name)_act")
  end
  X = mx.Dropout(X, p=dropout, name="$(name)_dropout")
  return X
end

export PlotCallback
type PlotCallback <: mx.AbstractEpochCallback
    graphs::Dict{Symbol, Array{Float32,1}}

    PlotCallback() = new(Dict{Symbol, Array{Float32,1}}())
end


function (cb::PlotCallback){T<:Real}(model :: Any, state :: mx.OptimizationState, metric :: Vector{Tuple{Base.Symbol, T}})
  for index in 1:length(metric)
    key = metric[index][1]
    if !haskey(cb.graphs, key)
      cb.graphs[key] = Float32[]
    end

    push!(cb.graphs[key], metric[index][2])
  end
end



export calculate_parameters
function calculate_parameters(model, filepath)
  file = open(filepath, "w")
  total_parameter_count = 0
  for param in model.arg_params
    line = "Parameter $(param[1]) has shape $(size(param[2])) = $(length(param[2]))"
    write(file, line*"\n")
    total_parameter_count += length(param[2])
  end
  lastline = "Total parameter count: $total_parameter_count"
  write(file, lastline*"\n")
  close(file)
 end

 function get_total_parameter_count(model)
   return sum([length(param[2]) for param in model.arg_params])
 end


 function to_context(gpus::Vector{Int})
   if length(gpus) == 0
     return mx.cpu()
   else
     return [mx.gpu(i) for i in gpus]
   end
 end

 type NetworkInfo
   name::String
   dir::AbstractString
   config::Dict
   context
   model
   epoch::Integer # the current state of the model, initialized to 0.
   training_curve::Vector{Float64} # MSE, created during training
   xval_curve::Vector{Float64} # MSE, created on demand
#   second_training_curve::Vector{Float64}

   NetworkInfo(name::String, dir::AbstractString, config::Dict, context) =
      new(name, dir, config, context, nothing, 0, Float64[], Float64[])
 end

Base.getindex(n::NetworkInfo, key::AbstractString) = n.config[key]
Base.setindex!(n::NetworkInfo, value, key::AbstractString) = n.config[key] = value

export save_compatible_checkpoint
function save_compatible_checkpoint(sym :: mx.SymbolicNode, arg_params :: Dict{Base.Symbol, mx.NDArray}, aux_params :: Dict{Base.Symbol, mx.NDArray}, prefix :: AbstractString, epoch :: Int)
  if epoch <= 1
    mx.save("$prefix-symbol.json", sym)
  end
  save_dict = merge(Dict{Base.Symbol, mx.NDArray}(map((x) -> Symbol("arg:$(x[1])") => x[2], arg_params)),
                    Dict{Base.Symbol, mx.NDArray}(map((x) -> Symbol("aux:$(x[1])") => x[2], aux_params)))
  save_filename = format("{1}-{2:04d}.params", prefix, epoch)
  mx.save(save_filename, save_dict)
end

function mx_create_kvstore(kv_type :: Base.Symbol, num_device :: Int, arg_params :: Dict{Base.Symbol,mx.NDArray})
  if num_device == 1 && !ismatch(r"dist", string(kv_type))
    kv = nothing
  else
    if kv_type == :local
      max_size = maximum([prod(size(param)) for (k,param) in arg_params])
      if max_size < 1024 * 1024 * 16
        kv_type = :local_update_cpu
      else
        kv_type = :local_allreduce_cpu
      end
      info("Auto-select kvstore type = $kv_type")
    end
    kv = mx.KVStore(kv_type)
  end

  update_on_kvstore = true
  if isa(kv, Void) || ismatch(r"local_allreduce", string(mx.get_type(kv)))
    update_on_kvstore = false
  end

  return (kv, update_on_kvstore)
end

export train
function train(n::NetworkInfo,
      train_provider, eval_provider;
      weight_init=true, verbosity=3)
  learning_rate = n["learning_rate"]
  epochs = n["epochs"]
  optimizer_name = n["optimizer"]

  if epochs <= n.epoch
    verbosity >= 2 && info("$(n.name): Target epoch already reached.")
    return
  end

  training_curve = PlotCallback()
  eval_curve = Float64[]
  #second_training_curve = Float64[]

  metric = mx.MSE()

  optimizer = set_optimizer(optimizer_name, learning_rate, gamma=0.9)

  # Manual initialization
  if !isdefined(n.model, :arg_params)&&weight_init
    info("Weight initialization.")
    mx.init_model(n.model, mx.UniformInitializer(0.1); overwrite=false, [mx.provide_data(train_provider)..., mx.provide_label(train_provider)...]...)
  end

  # Train each step manually
  for epoch in (n.epoch+1) : epochs
    mx.fit(n.model, optimizer, train_provider,
           n_epoch=1,
           eval_metric=metric,
           callbacks=[training_curve],
           verbosity=0,
           kvstore=:device)
    save_compatible_checkpoint(n.model.arch, n.model.arg_params, n.model.aux_params, joinpath(n.dir,n.name), epoch)
# get validation metric
    eval_mse = eval_pred(n.model, eval_provider, mx.MSE())[1][2]
    push!(eval_curve, eval_mse)
    #train_mse = eval_pred(n.model, train_provider, mx.MSE())[1][2]
    #push!(second_training_curve, train_mse)
    verbosity >= 3 && info("Epoch $epoch / $epochs: MSE = $eval_mse.")
    verbosity >= 3 && flush(STDOUT)
  end

  n.epoch = epochs

  calculate_parameters(n.model, joinpath(n.dir,n.name*"-parameters.txt"))

  append!(n.training_curve, training_curve.graphs[:MSE])
  append!(n.xval_curve, eval_curve)
  #append!(n.second_training_curve, second_training_curve)
  writedlm(joinpath(n.dir,n.name*"-MSE-train.txt"), n.training_curve)
  writedlm(joinpath(n.dir,n.name*"-MSE-xval.txt"), n.xval_curve)
end


export build
function build(n::NetworkInfo, method::Symbol,
    train_provider, eval_provider, build_function;
    verbosity=2, weight_init=true
  )
  target_epoch = n["epochs"]
  slim = n["slim"]

  if(slim > 0 && method != :load)
    verbosity >= 2 && info("$(n.name): slim $(train_provider.sample_count) -> $slim")
    train_provider = slim_provider(train_provider, slim)
    if eval_provider != nothing
      eval_provider = slim_provider(eval_provider, slim)
    end
  end

  if method == :train
    loss, net = build_function(n.config, size(train_provider.data_arrays[1],1))
    n.model = mx.FeedForward(loss, context=n.context)
    train(n, train_provider, eval_provider; verbosity=verbosity, weight_init=weight_init)
    load_network(n, target_epoch; pick_best=true)
  elseif method == :load
    load_network(n, target_epoch)
  elseif method == :refine
    load_network(n, -1; pick_best=false)
    train(n, train_provider, eval_provider; verbosity=verbosity)
    load_network(n, target_epoch)
  else
    throw(ArgumentError("method must be train, load or refine. got $method"))
  end
end

function slim_provider(p::mx.ArrayDataProvider, slim)
  slim = min(slim, p.sample_count)
  return mx.ArrayDataProvider(p.data_names[1] => slim_array(p.data_arrays[1], slim),
      p.label_names[1] => slim_array(p.label_arrays[1], slim); batch_size=p.batch_size)
end

function slim_array(array, slim)
  if(length(size(array)) == 1)
    return array[1:slim]
  else
    return array[:,1:slim]
  end
end


function padded_array_provider(key, data::Matrix{Float32}, batch_size)
  if size(data, 2) < batch_size
    plot_waveforms_padded = zeros(size(data,1), batch_size)
    plot_waveforms_padded[:,1:size(data, 2)] = data
  else
    plot_waveforms_padded = data
  end
  return mx.ArrayDataProvider(key => plot_waveforms_padded, batch_size=batch_size)
end

export eval_pred
function eval_pred(model, provider::mx.ArrayDataProvider, metric::mx.AbstractEvalMetric)
  prediction = mx.predict(model, provider; verbosity=0)
  data = provider.label_arrays[1]
  mx.reset!(metric)

  data_nd = mx.NDArray(data)
  prediction_nd = mx.NDArray(prediction)

  mx.update!(metric, [data_nd], [prediction_nd])
  return mx.get(metric)
end


export load_network
 function load_network(n::NetworkInfo, max_epoch; output_name="loss", delete_unneeded_arguments=true, pick_best=true)
   
   if max_epoch < 0
     max_epoch = last_epoch(n.dir, n.name)
   end

  n.training_curve = readdlm(joinpath(n.dir, "$(n.name)-MSE-train.txt"))[:,1]
  n.xval_curve = readdlm(joinpath(n.dir, "$(n.name)-MSE-xval.txt"))[:,1]

  if pick_best
    epoch = findmin(n.xval_curve)[2]
    info("$(n.name): best epoch is $epoch.")
  else
    epoch = max_epoch
  end

   load_network_checkpoint(n, epoch; output_name=output_name, delete_unneeded_arguments=delete_unneeded_arguments)

   return n
 end

 function load_network_checkpoint(n::NetworkInfo, epoch; output_name="loss", delete_unneeded_arguments=true)
   sym, arg_params, aux_params = mx.load_checkpoint(joinpath(n.dir, n.name), epoch)
   n.model = subnetwork(sym, arg_params, aux_params, output_name, delete_unneeded_arguments, n.context)
   n.epoch = epoch
 end

 function subnetwork(sym, arg_params, aux_params, output_name, delete_unneeded_arguments, context)
   all_layers = mx.get_internals(sym)
   loss = all_layers[output_name*"_output"]

   model = mx.FeedForward(loss, context=context)
   model.arg_params = copy(arg_params)
   model.aux_params = copy(aux_params)

   if delete_unneeded_arguments
     needed_args = mx.list_arguments(loss)
     for (name, array) in model.arg_params
       if !(name in needed_args)
         delete!(model.arg_params, name)
       end
     end
   end

   return model
 end

export subnetwork
 function subnetwork(network::mx.FeedForward, subnetwork::mx.FeedForward)
   subnetwork.arg_params = copy(network.arg_params)
   subnetwork.aux_params = copy(network.aux_params)

   needed_args = mx.list_arguments(subnetwork.arch)
   for (name, array) in subnetwork.arg_params
     if !(name in needed_args)
       delete!(subnetwork.arg_params, name)
     end
   end

   return subnetwork
 end

export last_epoch
 function last_epoch(dir, prefix; start=1)
   i = start
   if !isfile("$dir/$prefix-$(lpad(i,4,0)).params")
     throw(ArgumentError("Model not found: $dir"))
   end

   while isfile("$dir/$prefix-$(lpad(i,4,0)).params")
     i += 1
   end
   return i-1
 end

 function exists_network(dir::AbstractString, name::AbstractString)
   return isfile("$dir/$name-symbol.json")
 end

function decide_best_action(n::NetworkInfo)
  if !exists_network(n.dir, n.name) return :train end
  if n["epochs"] > last_epoch(n.dir, n.name) return :refine end
  return :load
end
