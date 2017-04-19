# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

using MXNet

import Base: getindex


function fc_layer(name, X, n_hidden, act_type, pdropout)
  fc = mx.FullyConnected(X, num_hidden=Int(n_hidden), name="$(name)")
  act = mx.Activation(fc, act_type=act_type, name="$(name)_act")
  return mx.Dropout(act, p=pdropout, name="$(name)_dropout")
end

function conv_layer(name, X, num_filter, filter_length, act_type, pool_size, pool_type, dropout)
  conv = mx.Convolution(X, kernel=(1, filter_length), pad=(0, 4), num_filter=num_filter, name="$name") # 2D->3D
  act = mx.Activation(conv, act_type=act_type, name="$(name)_act")
  pool = mx.Pooling(act, kernel=(1, pool_size), stride=(1, pool_size), pool_type=pool_type, name="$(name)_pool") # 256->64
  dropout = mx.Dropout(pool, p=dropout, name="$(name)_dropout")
  return dropout
end

function deconv_layer(name, X, num_filter, filter_length, act_type, pool_size, dropout)
  pad = Int( floor(filter_length/2) - pool_size/2 )
  if pad < 0
    throw(ArgumentError("upsampling not possible because padding is negative ($pad). Increase filter length or decrease pool size."))
  end
  X = mx.Deconvolution(X, kernel=(1, filter_length-1), stride=(1, pool_size), pad=(0, pad), num_filter=num_filter, name="$name")
  if act_type != nothing
    X = mx.Activation(X, act_type=act_type, name="$(name)_act")
  end
  X = mx.Dropout(X, p=dropout, name="$(name)_dropout")
  return X
end



type PlotCallback <: mx.AbstractEpochCallback
    graphs::Dict{Symbol, Array{Float32,1}}

    PlotCallback() = new(Dict{Symbol, Array{Float32,1}}())
end

function (cb::PlotCallback){T<:Real}(model :: Any, state :: mx.OptimizationState, metric :: Vector{Tuple{Base.Symbol, T}})
  for index in 1:length(metric)
    key = metric[index][1]
    if !haskey(cb.graphs, key)
      info("PlotCallback: creating key $key")
      cb.graphs[key] = Float32[]
    end

    push!(cb.graphs[key], metric[index][2])
  end
end




function calculate_parameters(model, filepath)
  file = open(filepath, "w")
  total_parameter_count = 0
  for param in model.arg_params
    line = "Parameter $(param[1]) has shape $(size(param[2])) = $(length(param[2]))"
    println(line)
    write(file, line*"\n")
    total_parameter_count += length(param[2])
  end
  lastline = "Total parameter count: $total_parameter_count"
  println(lastline)
  write(file, lastline*"\n")
  close(file)
 end


function exists_device(ctx::mx.Context)
 try
   mx.ones(Float32, (1,1), ctx)
   return true
 catch err
   return false
 end
end

 best_device_cache = nothing

 function best_device()
   global best_device_cache
   if best_device_cache == nothing
     gpus = list_xpus(mx.gpu)
     if length(gpus) > 0
       best_device_cache = gpus
       info("$(length(gpus)) GPUs found.")
     else
       best_device_cache = mx.cpu()
       info("No GPUs available, fallback to single CPU.")
     end
   end
   return best_device_cache
 end

export use_gpu
function use_gpu(id)
  global best_device_cache
  if exists_device(mx.gpu(id))
    best_device_cache = mx.gpu(id)
  else
    info("Could not access GPU $id, fallback to CPU")
    best_device_cache = mx.cpu()
  end
end

export list_xpus
function list_xpus(xpu=mx.gpu)
  result = mx.Context[]
  while exists_device(xpu(length(result))) && length(result) < 8
    push!(result, xpu(length(result)))
  end
  return result
end



 type NetworkInfo
   name::String
   dir::AbstractString
   config::Dict
   model
   epoch::Integer # the current state of the model, initialized to 0.

   NetworkInfo(name::String, dir::AbstractString, config::Dict) =
      new(name, dir, config, nothing, 0)
 end

export getindex
 function getindex(n::NetworkInfo, key)
  return n.config[key]
end


function train(n::NetworkInfo,
      train_provider, eval_provider,
      xpu)
  learning_rate = n["learning_rate"]
  epochs = n["epochs"]

  plot_cb = PlotCallback()

  metric = mx.MSE()

  optimizer = mx.ADAM(lr=learning_rate)
  println("Training on device $xpu")
  mx.fit(n.model, optimizer, train_provider,
         n_epoch=epochs-n.epoch,
         eval_metric=metric,
         kvstore=:device,
         callbacks=[plot_cb, mx.do_checkpoint(joinpath(n.dir,n.name))])

  # TODO eval not implemented

  n.epoch = epochs

  calculate_parameters(n.model, joinpath(n.dir,n.name*"-parameters.txt"))
  writedlm(joinpath(n.dir,n.name*"-mse.txt"), plot_cb.graphs[:MSE])
end


function build(n::NetworkInfo, method::Symbol,
    train_provider, eval_provider, build_function;
    xpu=best_device()
  )
  target_epoch = n["epochs"]
  slim = n["slim"]

  if(slim > 0)
    println("$(n.name): slim $(train_provider.sample_count) -> $slim")
    train_provider = slim_provider(train_provider, slim)
    if eval_provider != nothing
      eval_provider = slim_provider(eval_provider, slim)
    end
  end

  if method == :train
    loss, net = build_function(n.config, size(train_provider.data_arrays[1],1))
    n.model = mx.FeedForward(loss, context=xpu)
    train(n, train_provider, eval_provider, xpu)
  elseif method == :load
    load_network(n, target_epoch)
  elseif method == :refine
    load_network(n, -1)
    train(n, train_provider, eval_provider, xpu)
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


function eval(model, provider::mx.ArrayDataProvider, metric::mx.AbstractEvalMetric)
  prediction = mx.predict(model, provider)
  data = provider.data_arrays[1]
  mx.reset!(metric)

  data_nd = mx.NDArray(data)
  prediction_nd = mx.NDArray(prediction)

  mx.update!(metric, [data_nd], [prediction_nd])
  return mx.get(metric)
end



 function load_network(n::NetworkInfo, epoch; output_name="softmax", delete_unneeded_arguments=true)
   if epoch < 0
     epoch = last_epoch(n.dir, n.name)
   end

   sym, arg_params, aux_params = mx.load_checkpoint(joinpath(n.dir, n.name), epoch)
   n.model = subnetwork(sym, arg_params, aux_params, output_name, delete_unneeded_arguments)
   n.epoch = epoch
   return n
 end

 function subnetwork(sym, arg_params, aux_params, output_name, delete_unneeded_arguments; xpu=best_device())
   all_layers = mx.get_internals(sym)
   loss = all_layers[output_name*"_output"]

   model = mx.FeedForward(loss, context=xpu)
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
