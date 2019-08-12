# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

using JSON, Base.Threads, MultiThreadingTools


export DLEnv
type DLEnv # immutable
  dir::AbstractString
  name::AbstractString
  config::Dict
  _gpus::Vector{Int}
  _ext_event_libs::Dict{String,DLData}
  _verbosity::Integer # 0: nothing, 1: only errors, 2: default, 3: all
#constructors
  DLEnv() = DLEnv("config")
  DLEnv(name::AbstractString) = DLEnv(abspath(""), name)

  function DLEnv(dir::AbstractString, name::AbstractString)
    env = new(dir, name, Dict(), Int[], Dict(), 0)
    reload(env; replace=true)
    setup(env)
    return env
  end
end

type ConfigurationException <: Exception end

function Base.reload(env::DLEnv; replace=false)
  f = open(joinpath(env.dir,"$(env.name).json"), "r")
  dicttxt = readstring(f)  # file information to string
  env.config = JSON.parse(dicttxt)  # parse and transform data
  if replace
    env._verbosity = get(env.config, "verbosity", 2)
  end
end

#get keys of subdict
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

  modifier(d)
  set_properties!(env, new_name, d)
  return d
end

export resolvepath #concats the path
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

#verbosity settings
function Base.info(env::DLEnv, level::Integer, msg::AbstractString)
  if env._verbosity >= level
    threadsafe_info(msg)
    flush(STDOUT)
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

export with_verbosity
function with_verbosity(action, env::DLEnv, verb)
    default_verbosity = env._verbosity
    env._verbosity = verb
    result = action()
    env._verbosity = default_verbosity
    return result
end

export verbose
verbose(action, env::DLEnv) = with_verbosity(action, env, 1000)

export silent
silent(action, env::DLEnv) = with_verbosity(action, env, 0)

#dict indexing with functions
function Base.getindex(env::DLEnv, key::String)
  return env.config[key]
end

function Base.setindex!(env::DLEnv, key::String, value)
  env.config[key] = value
end

function Base.haskey(env::DLEnv, key::String)
  return haskey(env.config, key)
end

#detector types, returns subset of dets
export detectors
function detectors(env::DLEnv, dettype::AbstractString)
  if dettype == "BEGe"
    return BEGes_GERDA_II()
  elseif dettype == "coax" || dettype == "semi-coaxial"
    return Coax_GERDA_II()
  elseif dettype == "natural"
    return Natural_GERDA_II()
  elseif dettype == "used"
    return Used_GERDA_II()
  else
    throw(ArgumentError("Unknown detector type: $dettype"))
  end
end

function detectors(env::DLEnv, keywords::AbstractString...)
    sets = [detectors(env, keyword) for keyword in keywords]
    return intersect(sets...)
end

export detectors
function detectors(env::DLEnv) #extended to select detectors
  dettype = env.config["raw"]["detectors"]
  info(length(dettype))
  if length(dettype) == 0
    	return phase2_detectors
  else
	detectors = string.(filter!(sde->sde ≠ false, [sd in phase2_detectors && sd for sd in dettype])) 
  	"BEGe" in dettype&&push!(detectors,BEGes_GERDA_II()...)
  	"coax" in dettype&&push!(detectors,Coax_GERDA_II()...)
  	"natural" in dettype&&push!(detectors,Natural_GERDA_II()...)
  	"used" in dettype&&push!(detectors,Used_GERDA_II()...)
  	if iszero(length(detectors)) throw(ArgumentError("Unknown detector type: $dettype")) else return unique(detectors) end 
  end
end


export _create_h5data
function _create_h5data(env::DLEnv, output_dir)
    info(env, 2, "Reading original data from $(env.config["path"])")
    isdir(output_dir) || mkdir(output_dir) #error if target dir nonnexistent and cannot be created
    if haskey(env.config, "rawformat")
        _seg_to_hdf5(env, output_dir)
    else
        _mgdo_to_hdf5(env, output_dir)
    end
    info(env, 3, "Converted raw data to HDF5 and stored in $output_dir.")
end

function _mgdo_to_hdf5(env::DLEnv, output_dir)
  keylists = KeyList[]
  for (i,keylist_path) in enumerate(env.config["keylists"]) #txt that contains data file names
    if !endswith(keylist_path, ".txt") #if you left the txt from the end, it adds it
      keylist_path = keylist_path*".txt"
    end
#extracts whats in the txt (the datafile names) and append content of each .txt to a KeyList object
    push!(keylists, parse_keylist(resolvepath(env, keylist_path), keylist_path))
  end
    load_root(; verbosity=get_verbosity(env))
#gets keylists (name of .txt, actual datafiles in this txt) as input and reads files
    dets = detectors(env)
    mgdo_to_hdf5(env.config["path"], dets, output_dir, keylists; verbosity=get_verbosity(env)) # added raw to select detectors
end

function _seg_to_hdf5(env::DLEnv, output_dir)
    src_dirs = env.config["path"]
    keylists = Vector{AbstractString}[]
    if env.config["rawformat"] == "SegHDF5"
      for src_dir in src_dirs
          content = readdir(src_dir)
          content = joinpath.(src_dir, content[find(f->endswith(f, ".hdf5"), content)])
          push!(keylists, content)
      end
      segh5_to_hdf5(env.config["rawformat"], keylists, output_dir, get_verbosity(env))
    else
      for src_dir in src_dirs
          content = readdir(src_dir)
          content = src_dir .* "/" .* content[find(f->endswith(f, ".root"), content)]
          push!(keylists, content)
      end
      load_root(; verbosity=get_verbosity(env))
      seg_to_hdf5(env.config["rawformat"], keylists, output_dir, get_verbosity(env))
    end
end

#get data
export getdata
"""
data: DLData object that will be preprocessed if given. Useful when raw data is loaded and manipulated and you want it to preprocess without saving the intermediate state.
raw_dir_name: Defines the folder of the raw data. If exists, data can be loaded from it directly. If doesn't exist, it will be created and raw data from the root files will be loaded in there.
"""
function getdata(env::DLEnv; data::Union{DLData,Void}=nothing, raw_dir_name::String="raw", preprocessing::Union{AbstractString,Void}=nothing, targets::Array{String}=String[])
  if preprocessing==nothing
    return get(env, raw_dir_name; targets=targets) do
      _get_raw_data(env, raw_dir_name; targets=targets)
    end
  elseif preprocessing=="processed"
    return get(env, raw_dir_name; h5_libname="_preprocessed_all")
  else
#gets raw data and puts them to folder defined by preprocessing
    if data == nothing
        data = _get_raw_data(env, raw_dir_name; targets=[preprocessing]) #preprocesses data in *raw_dir_name*
    else
        data = deepcopy(data)
    end
    preprocessed = get(env, preprocessing; targets=targets) do #this is the return value
      preprocess(env, data, preprocessing)
    end
 
    if preprocessed == nothing
      return nothing
    end
    # Else check whether cache is up to date
    steps = get_properties(env, preprocessing)["preprocessing"]
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
      delete!(env, preprocessing)
      return getdata(env; preprocessing=preprocessing, targets=targets)
    end
  end
end

function _get_raw_data(env::DLEnv, raw_dir_name="raw"; targets::Array{String}=String[])
#dir to put data
  raw_dir = resolvepath(env, "data", raw_dir_name)
  if !isdir(raw_dir)
#get the data from the txts
    _create_h5data(env, raw_dir)
  end
  return lazy_read_all(raw_dir)
end


export preprocess
function preprocess(env::DLEnv, data::DLData, config_name) #config name is "pulses" or "noise"
"""
Selects detectors from the detectors key of the dict. Also selects preprocessing configuration settings.
Applies builtin filters: test pulses, baseline events and nonphysical events.
Then makes an empty data object for the events, fills in and saves it to a file.
Then gets the preprocessing steps from the config file.
Then loops over the selected detectors and applies the preprocessing steps.
After the preprocessing it filters the events that are failed during pp.
Then splits the data of the detector into the datasets specified in the config file and writes them into the result file.
"""
  config = env.config[config_name]
  @assert isa(config, Dict)
  select_channels = parse_detectors(config["detectors"])
  #if select_channels[1] in ["BEGe", "coax"]
#	select_channels = detectors(env, select_channels[1])
 # end
  info(env, 3, "Selected channels: $select_channels")
  if isa(select_channels,Vector) && length(select_channels) > 0
    filter!(data, :detector_name, select_channels)
  end
  
  N_dset = length(parse_datasets(env, config["sets"]))  # sets are train val and test with ratios
  result = DLData(fill(EventLibrary(zeros(0,0)), length(data)*N_dset))
  result.dir = _cachedir(env, "tmp-preprocessed-$config_name")
  steps = convert(Array{String}, config["preprocessing"])
  
  for i in 1:length(data)
    lib = data.entries[i]
    _builtin_filter(env, config, "test-pulses", lib, :isTP, isTP -> isTP == 0)
    info(env,3,"filtering non-test pulses: $(eventcount(lib)) events left")
    _builtin_filter(env, config, "baseline-events", lib, :isBL, isBL -> isBL == 0)
    info(env,3,"filtering non-baseline events: $(eventcount(lib)) events left")
    _builtin_filter(env, config, "unphysical-events", lib, :E, E -> (E > 0) && (E < 9999))
    info(env,3,"filtering unphysical events: $(eventcount(lib)) events left")
    _builtin_filter(env, config, "high-multiplicity-events", lib, :multiplicity, multiplicity -> multiplicity == 1)
    info(env,3,"filtering single multiplicity events: $(eventcount(lib)) events left")

    _builtin_filter(env, config, "muon-vetoed-events", lib, :isMuVetoed, isMuVetoed -> isMuVetoed == 0)
    info(env,3,"filtering muon vetoed events: $(eventcount(lib)) events left")

    _builtin_filter(env, config, "pileup-events", lib, :isPileup, isPileup -> isPileup == 0)
    info(env,3,"filtering non-pileup events: $(eventcount(lib)) events left")
    lib_t = preprocess_transform(env, lib, steps; copyf=identity)  # do actual prerpocessing
    #_builtin_filter(env, config, "failed-preprocessing", lib_t, :FailedPreprocessing, fail -> fail == 0)
    #info(env,3,"filtering failed-pp events: $(eventcount(data)) events left")
    part_data = DLData(collect(values(split(env, lib_t, config["sets"]))))
    write_all_sequentially(part_data, result.dir, true)  # write it into file
    info(env,3, "Wrote datasets of $(lib[:name]) and released allocated memory.")
    dispose(lib)
    dispose(lib_t)
    for j in 1:length(part_data)  # usually train, xval, test, so 3
      result.entries[N_dset*(i-1) + j] = part_data.entries[j]  # 1,2,3; 4,5,6: ... puts everything in results dict like train,val,test,train,val,test,...
    end
    @assert length(lib.waveforms) == 0 && length(lib_t.waveforms) == 0
  end

  return result # dict of split datasets
end


function _builtin_filter(env::DLEnv, config::Dict, ftype_key::String, data::EventCollection, label, exclude_prededicate)
  # TODO per detector to save memory
  ftype = config[ftype_key]
  if ftype == "exclude"
    result = filter!(data, label, exclude_prededicate)  
    return result
  elseif ftype == "include"
    return data
  elseif ftype == "only"
    return filter!(data, label, x -> !exclude_prededicate(x))
  else
    throw(ArgumentError("Unknown filter keyword in configuration $ftype_key: $ftype"))
  end
end


#reads datasets and ratios from env (config)
export parse_datasets
function parse_datasets(env::DLEnv, strdict::Dict)
#empty dict
  result = Dict{AbstractString,Vector{AbstractFloat}}()
  if haskey(env.config, "keylists") # keylists contains data file txt names
    requiredlength = length(env["keylists"]) # as many files I read
  else
    requiredlength = 1
  end
  for (key,value) in strdict #loop over train, val and test

    if isa(value, Vector) # I can specify datasets per file, different for each but then the length of the array must match as many files I have
      @assert length(value) == requiredlength
      result[key] = value
    elseif isa(value, Real) #same ratios for all files
      result[key] = fill(value, requiredlength) # creates requiredlength long array with all values being value
    else
      throw(ConfigurationException())
    end
  end
#result contains datasets train, xval and test with their ratios
  return result
end

Base.split(env::DLEnv, data::EventCollection, strdict::Dict) = split(data, parse_datasets(env, strdict))


function setup(env::DLEnv)
  e_mkdir(env.dir)
  e_mkdir(joinpath(env.dir, "data"))
  e_mkdir(joinpath(env.dir, "models"))
  e_mkdir(joinpath(env.dir, "plots"))
end
e_mkdir(dir) = !isdir(dir) && mkdir(dir)

function Base.get(compute, env::DLEnv, lib_name::String; targets::Array{String}=String[], uninitialize=true, h5_libname::String="")
  if !isempty(targets) && containsall(env, targets)
    info(env, 2, "Skipping retrieval of '$lib_name'.")
    return nothing
  end
  if contains(env, lib_name)
    info(env, 2, "Retrieving '$lib_name' from cache.")
    return get(env, lib_name; h5_libname=h5_libname)
  else
    info(env, 2, "Computing '$lib_name'...")
    data = compute()
    info(env, 3, "Computation of '$lib_name' finished.")

    # check type
    if !isa(data, DLData)
      throw(TypeError(Symbol(compute), "get_or_compute", DLData, data))
    end

    if data.dir == nothing
      info(env, 3, "Writing computed data to env as '$lib_name' (uninitialize=$uninitialize).")
      push!(env, lib_name, data; uninitialize=uninitialize)
      return data
    else
      # Rename directory
      targetdir = _cachedir(env, lib_name)
          if targetdir != data.dir
              info(env, 3, "Moving data from $(data.dir) to $targetdir and reloading.")
              mv(data.dir, targetdir)
              return get(env, lib_name)
            else
                info(env, 3, "Data is already in the right location ($(data.dir)).")
                return data
            end
    end
  end
end

function Base.get(env::DLEnv, lib_name::String; h5_libname="")
  _ensure_ext_loaded(env, lib_name; h5_libname=h5_libname)
  return env._ext_event_libs[lib_name]
end

Base.push!(env::DLEnv, lib_name::String, lib::EventLibrary; uninitialize::Bool=false) = push!(env, lib_name, DLData([lib]); uninitialize=uninitialize)

function Base.push!(env::DLEnv, lib_name::String, data::DLData; uninitialize::Bool=false)
  _ensure_ext_loaded(env, lib_name)
  env._ext_event_libs[lib_name] = data
  if env.config["cache"]
    write_all_multithreaded(data, joinpath(env.dir, "data", lib_name), uninitialize)
  end
end

function Base.contains(env::DLEnv, lib_name::String)
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

function _ensure_ext_loaded(env::DLEnv, lib_name::String; h5_libname::String="")
  if !haskey(env._ext_event_libs, lib_name)
    c_dir = _cachedir(env, lib_name)
    if isdir(c_dir)
      env._ext_event_libs[lib_name] = lazy_read_all(c_dir; h5_libname=h5_libname)
      
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

#define directory of env, if nonexistent, then creates the dir
export _cachedir
function _cachedir(env::DLEnv, lib_name::String)
  return joinpath(env.dir, "data", lib_name)
end

export network
function network(env::DLEnv, name::String)
    dir = joinpath(env.dir, "models", name)
    isdir(dir) || mkdir(dir)
    return NetworkInfo(name, dir, get_properties(env, name), to_context(env._gpus))
end

export use_gpus
function use_gpus(env, gpus::Int...)
  env._gpus = [gpus...]
end


