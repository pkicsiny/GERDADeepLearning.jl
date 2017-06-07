# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

using JSON

import Base: get, contains

export DLEnv
type DLEnv # immutable
  dir::AbstractString
  config::Dict
  _gpus::Vector{Int}
  _ext_event_libs::Dict{String,DLData}
  _verbosity::Integer # 0: nothing, 1: only errors, 2: default, 3: all

  DLEnv() = DLEnv("config")
  DLEnv(name::AbstractString) = DLEnv(abspath(""), name)
  function DLEnv(dir::AbstractString, name::AbstractString)
    f = open(joinpath(dir,"$name.json"), "r")
    dicttxt = readstring(f)  # file information to string
    dict = JSON.parse(dicttxt)  # parse and transform data
    env = new(dir, dict, Int[], Dict(), get(dict, "verbosity", 2))
    setup(env)
    return env
  end
end

type ConfigurationException <: Exception end

export get_properties
function get_properties(env::DLEnv, name::AbstractString)
  return get(env.config, name, Dict())
end

export set_properties!
function set_properties!(env::DLEnv, name::AbstractString, props::Dict)
  env.config[name] = props
end

export new_properties!
function new_properties!(modifier, env::DLEnv, template_name::AbstractString, new_name::AbstractString)
  d = copy(get_properties(env, template_name))
  d = modifier(d)
  set_properties!(env, new_name, d)
  return d
end

export resolvepath
function resolvepath(env::DLEnv, path::AbstractString...)
  if isabspath(path[1])
    return joinpath(path...)
  else
    return joinpath(env.dir, path...)
  end
end

function Base.joinpath(env::DLEnv, elements::String...)
  return joinpath(env.dir, elements...)
end

function Base.info(env::DLEnv, level::Integer, msg::AbstractString)
  if env._verbosity >= level
    info(msg)
  end
end

export set_verbosity!
function set_verbosity!(env::DLEnv, verbosity::Integer)
  env._verbosity = verbosity
end

export get_verbosity
function get_verbosity(env::DLEnv)
  return env._verbosity
end

function Base.getindex(env::DLEnv, key::String)
  return env.config[key]
end

function Base.setindex!(env::DLEnv, key::String, value)
  env.config[key] = value
end

function Base.haskey(env::DLEnv, key::String)
  return haskey(env.config, key)
end


export _create_h5data
function _create_h5data(env::DLEnv, raw_dir)
  info(env, 2, "Reading original data from $(env.config["path"])")
  keylists = KeyList[]
  for (i,keylist_path) in enumerate(env.config["keylists"])
    if !endswith(keylist_path, ".txt")
      keylist_path = keylist_path*".txt"
    end
    push!(keylists, parse_keylist(resolvepath(env, keylist_path), keylist_path))
  end
  isdir(raw_dir) || mkdir(raw_dir)
  mgdo_to_hdf5(env.config["path"], raw_dir, keylists; verbosity=get_verbosity(env))
end

export getdata
function getdata(env::DLEnv; preprocessed=false, targets::Array{String}=String[])
  if !preprocessed
    return get(env, "raw"; targets=targets) do
      _get_raw_data(env; targets=targets)
    end
  else
    data = _get_raw_data(env; targets=["preprocessed"])
    preprocessed = get(env, "preprocessed"; targets=targets) do
      preprocess(env, data)
    end
    if preprocessed == nothing
      return nothing
    end
    # Else check whether cache is up to date
    steps = env.config["preprocessing"]
    cache_up_to_date = true
    for lib in preprocessed
      cached_steps = lib[:preprocessing]
      if cached_steps != steps
        cache_up_to_date = false
      end
    end
    if cache_up_to_date
      return preprocessed
    else
      info(env, 2, "Refreshing cache of 'preprocessed'.")
      delete!(env, "preprocessed")
      return getdata(env; preprocessed=true, targets=targets)
    end
  end
end

function _get_raw_data(env::DLEnv; targets::Array{String}=String[])
  raw_dir = resolvepath(env, "data", "raw")
  if !isdir(raw_dir)
    _create_h5data(raw_dir)
  end
  return lazy_read_all(raw_dir)
end


export preprocess
function preprocess(env::DLEnv, data::DLData)
  select_channels=parse_detectors(env["detectors"])
  if isa(select_channels,Vector) && length(select_channels) > 0
    data = filter(data, :detector_id, select_channels)
  end

  data = _builtin_filter(env, "test-pulses", data, :isTP, isTP -> isTP == 0)
  data = _builtin_filter(env, "baseline-events", data, :isBL, isBL -> isBL == 0)
  data = _builtin_filter(env, "unphysical-events", data, :E, E -> (E > 0) && (E < 9999))

  data = preprocess_transform(env, data)

  data = split(env, data)

  return data
end


function _builtin_filter(env::DLEnv, ftype_key::String, data::DLData, label, exclude_prededicate)
  # TODO per detector to save memory
  ftype = env.config[ftype_key]
  if ftype == "exclude"
    before = eventcount(data)
    result = filter(data, label, exclude_prededicate)
    info(env, 2, "Excluded $(before-(eventcount(result))) $ftype_key of $before events.")
    return result
  elseif ftype == "include"
    return data
  elseif ftype == "only"
    return filter(data, label, x -> !exclude_prededicate(x))
  else
    throw(ArgumentError("Unknown filter keyword in configuration $ftype_key: $ftype"))
  end
end


function parse_datasets(env::DLEnv)
  result = Dict{AbstractString,Vector{AbstractFloat}}()
  strdict = get_properties(env, "sets")
  requiredlength = length(env["keylists"])
  for (key,value) in strdict
    if isa(value, Vector)
      @assert length(value) == requiredlength
      result[key] = value
    elseif isa(value, AbstractFloat)
      result[key] = fill(value, requiredlength)
    else
      throw(ConfigurationException())
    end
  end
  return result
end

function Base.split(env::DLEnv, data::DLData)
  datasets = parse_datasets(env)
  return split(data, datasets)
end

function setup(env::DLEnv)
  e_mkdir(env.dir)
  e_mkdir(joinpath(env.dir, "data"))
  e_mkdir(joinpath(env.dir, "models"))
  e_mkdir(joinpath(env.dir, "plots"))
end
e_mkdir(dir) = !isdir(dir) && mkdir(dir)

export get
function get(compute, env::DLEnv, lib_name::String; targets::Array{String}=String[])
  if !isempty(targets) && containsall(env, targets)
    info(env, 2, "Skipping retrieval of '$lib_name'.")
    return nothing
  end
  if contains(env, lib_name)
    info(env, 2, "Retrieving '$lib_name' from cache.")
    return get(env, lib_name)
  else
    info(env, 2, "Computing '$lib_name'...")
    data = compute()

    # check type
    if !isa(data, DLData)
      throw(TypeError(Symbol(compute), "get_or_compute", DLData, data))
    end

    push!(env, lib_name, data)
    return data
  end
end

function get(env::DLEnv, lib_name::String)
  _ensure_ext_loaded(env, lib_name)
  return env._ext_event_libs[lib_name]
end

function Base.push!(env::DLEnv, lib_name::String, data::DLData)
  _ensure_ext_loaded(env, lib_name)
  env._ext_event_libs[lib_name] = data
  if env.config["cache"]
    write_all(data, joinpath(env.dir, "data", lib_name))
  end
end

export contains
function contains(env::DLEnv, lib_name::String)
  if haskey(env._ext_event_libs, lib_name)
    return true
  end
  return env.config["cache"] && isdir(_cachedir(env, lib_name))
end

export containsall
function containsall(env::DLEnv, lib_names::Array{String})
  for lib_name in lib_names
    if !contains(env, lib_name) return false end
  end
  return true
end

function _ensure_ext_loaded(env::DLEnv, lib_name::String)
  if !haskey(env._ext_event_libs, lib_name)
    c_dir = _cachedir(env, lib_name)
    if isdir(c_dir)
      env._ext_event_libs[lib_name] = lazy_read_all(c_dir)
    end
  end
end

function Base.delete!(env::DLEnv, lib_name::String)
  if haskey(env._ext_event_libs, lib_name)
    delete!(env._ext_event_libs, lib_name)
  end
  # Delete cached files (even if cache is set to false)
  c_dir = _cachedir(env, lib_name)
  if isdir(c_dir)
    rm(c_dir; recursive=true)
    info(env, 2, "Deleted cached $lib_name")
  end
end

function _cachedir(env::DLEnv, lib_name::String)
  return joinpath(env.dir, "data", lib_name)
end


function network(env::DLEnv, name::String)
    dir = joinpath(env.dir, "models", name)
    isdir(dir) || mkdir(dir)
    return NetworkInfo(name, dir, get_properties(env, name), to_context(env._gpus))
end

export use_gpus
function use_gpus(env, gpus::Int...)
  env._gpus = [gpus...]
end
