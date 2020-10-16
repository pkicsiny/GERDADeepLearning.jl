using JLD
using Plots, StatsBase, LaTeXStrings, StatPlots
using MXNet
using Primes, Distributions
using Measures

export pyplot_config
"""
Sets a deafult plot style and size.
"""
function pyplot_config()
    pyplot(titlefont=24, tickfont=18, legendfont=14, guidefontsize=22, size=(600,500), w=2, framestyle=:box)
end

export formatter
"""
Utility function for rounding floats. Rounds to 5 decimals and changes to scientific notation with 2 decimals for small numbers.
"""
formatter(x) =  if x<1e-2; @sprintf("%.1e", x) elseif x<0; signif(x, 2) else string(round(x, 2)) end

export get_value
"""
Gets a value of a key. Generalized lookup method for DLData and EventLibrary.
"""
get_value(events::EventCollection, key::Symbol) = if typeof(events) == DLData; events[key][1] else events[key] end

export get_statistics
"""
Returns the mean, median and standard deviation of *x* in a dictionary.
"""
function get_statistics(x::AbstractArray)
    return Dict("mean"=>mean(x), "median"=>median(x), "std"=>std(x))
end

export dir
"""
Recursively create path if it does not exist.
Returns the path as a string.
"""
function dir(args...)
    	path = joinpath(args...)
	isdir(path)||info("Create path: $(path)")
	isdir(path)||try mkdir(path); catch mkdir(joinpath(dir(args[1:end-1]...), args[end])) end
	return path
end

export jld_to_hdf5
"""
Writes data to hdf5 file.
"""
function jld_to_hdf5(lib::EventLibrary, path::String)
    isdir(path)||mkdir(path)
    info("Wrtie $(lib[:detector_name]).")
    write_lib(lib, joinpath(path, "$(lib[:detector_name]).h5"), true)
end

function jld_to_hdf5(data::DLData, path::String)
    for lib in data
        jld_to_hdf5(lib, path)
    end
end

export save_working_data
function save_working_data(env::DLEnv, data; name="my_working_data", to_dir="saved")
    dir = resolvepath(env, "data", to_dir)
    isdir(dir) || mkdir(dir)
    save("data/$(to_dir)/$(name).jld", "name", data)
end

export load_working_data
function load_working_data(name="my_working_data", from_dir="data/saved")
    @assert isdir(from_dir) info(from_dir*" does not exist.")
    return load("$(from_dir)/$(name).jld")["name"]
end

export load_jld_data
"""
    load_jld_data(detector::String, datatype::String, pulse_type::String, ds_labels::String)

Loads saved pulses and noise from .jld file.
*detector* is the detector name. If the file has all detector data, set it to 'all'.
*datatype* is 'cal', 'phy' or 'mix'.
*pulse_type* is 'charge' or 'current'.
*ds_labels* is a string with additional filename labels.
E.g.: '_50-2500_400_long_noisy_cal' means the energy range is between [50,2500] keV, the window length is 400 and
the calibration data samples are mixed with baseline noise.
Returns two DLData files.
"""
function load_jld_data(detector::String, datatype::String, pulse_type::String, ds_labels::String)
    pulses = load_working_data(
         "$(datatype)-87-92/charge/pulses/run0087-0092_$(datatype)_charge_$(detector)_pulses$(ds_labels)")
    noise = load_working_data(
        "$(datatype)-87-92/charge/noise/run0087-0092_$(datatype)_charge_$(detector)_noise$(ds_labels)")
    
    if pulse_type == "current"
        info("Differentiating pulse and noise waveforms.")
        for i in range(1,length(pulses.entries))
            pulses.entries[i] = differentiate(pulses.entries[i])
            noise.entries[i] = differentiate(noise.entries[i])
        end
    end
    info("Loaded data has $(eventcount(pulses)) pulse and $(eventcount(noise)) noise traces.")
    info("Loaded data has $(sum(pulses[:isCal])) calibration and $(eventcount(pulses)-sum(pulses[:isCal])) physics samples.")
    info("Loaded data ranges from $(minimum(pulses[:E])) keV to $(maximum(pulses[:E])) keV")
    return pulses, noise
end

export shift_midpoints
"""
	 shift_midpoints(events::EventLibrary; center_y=0.5, target_length=256, position=0.25)

Aligns the midpoints of the charge signlas to a specified point in a cutoff window of specified length.
*center_y* is the level of the pulses compared to their maximum that shall be aligned. E.g. for 0.5 the half maximum values will be aligned.
*target_length* is an integer to specify the cutoff window length. The pulses are aligned within this window.
*position* is a float that gives the alignment position inside the cutoff window. E.g. for 0.5 the pulses will be aligned to the middle of the window.
Returns an EventLibrary object with the aligned waveforms.
"""
function shift_midpoints(events::EventLibrary;
 			 center_y=0.5,
 			 target_length=400,
 			 position=0.5)
  charges = charge_pulses(events; create_new=true)

  s = sample_size(events) #1000
  before = Int64(round(target_length*position)) #200
  after = Int64(round(target_length*(1 - position))) #200
  rwf = zeros(Float32, target_length, eventcount(events)) #400, #events

  @everythread for i in threadpartition(1:eventcount(events))
    index = findmin(abs.(charges.waveforms[:,i] - center_y))[2] #get y=0.5 index
    if (index < before) || (index > s - after) #if not enough space to cut, too left or right
      events[:FailedPreprocessing][i] = 1
    else
      rwf[:,i] = events.waveforms[(index-before+1) : (index+after) , i] #actual data cut of target_length size, failed pp waveforms stay 0 bc this line is not executed
    end
  end
  events.waveforms = rwf
  return events
end

export lazy_norm
"""
	lazy_norm(events::EventLibrary)

Normalizes events by dividing them by their energy.
"""
function lazy_norm(events::EventLibrary)
  charges = charge_pulses(events; create_new=true)
  energies = events[:E]
  @everythread for i in threadpartition(1:eventcount(events))
    events.waveforms[:,i] *= 1 / energies[i]
  end
  return events
end

export post_filter!
"""
Filters some flags that have not been filtered during preprocessing. E.g. events that failed during preprocessing.
"""
function post_filter!(events::EventLibrary, keys::Array{Symbol,1}, Emin::Int64=0, Emax::Int64=9999; save_folder::Union{String, Void}=nothing)
    filter!(events, :E, E -> E < Emax)
    filter!(events, :E, E -> E > Emin)
#filter according to keys
    if length(keys) > 0 && isa(keys,Array{Symbol,1})
       for (i, key) in enumerate(keys)
		info("Post filtering: $(key).")
       		indices = find(x->x!=0, events[key])
#plot some filtered waveforms
		n = min(50, length(indices))
		plots = smart_subplots(indices[1:n], events, series=true, labels="")
		if save_folder != nothing
			for i in 1:n
        			savefig(plots[i], joinpath(save_folder, "$(key)_$i.png"))
    			end
		end	
#perform filter on data	
		filter!(events,key, x -> x == 0)
       end
    end
end

export flag_abnormal_pulses
"""
	flag_abnormal_pulses(events::EventLibrary; center_y::Float32=0.5)

Custom labeling method for oddly shaped pulses.
Labels pulses which have a peak before main rise. Calculates the average of the first and last 200 samples and filters based on a threshold defined for the maximum and minimum average value.
*center_y* is a float representing the center point of the pulses by which they are to be examined.
Returns *events* with a new label (:abnormal_shape) that flags abnormal pulses.
"""
function flag_abnormal_pulses(events::EventLibrary; center_y::Float64=0.5)
    charges = charge_pulses(events; create_new=true)
    abnormal_shape = zeros(Int8, eventcount(events))
# if this many events fall below 0.5 after first occurence, then abnormal pulse
    extra_peak_threshold = 20
# if there are more than this many samples having value of [0.4,0.6] then abnormal pulse (mostly oddly aligned multi site events)
    width_threshold = 70
# sample must be this smaller than center_y value to count for abnormal pulse
    delta = 0.05
# thresholds for noise trigger
    upper_th = .2
    lower_th = .5
    @everythread for i in threadpartition(1:eventcount(charges))
# coordinate of half maximum
        midpoint = findmin(abs.(charges.waveforms[:,i] - center_y))[2]
# width of plateau between 0.4 and 0.6
	midwidth = length(filter(x -> (x<0.6)&&(x>0.4), waveforms(charges)[:,i]))
# flag noise triggered events by sample average
	avg_first = mean(waveforms(charges)[1:200, i])
	avg_last = mean(waveforms(charges)[201:400, i])
# set abnormal shape label for event
	abnormal_shape[i] = (sum((charges.waveforms[midpoint:end, i].+delta).<charges.waveforms[midpoint, i]) > extra_peak_threshold)||(midwidth>width_threshold)||(avg_last<=lower_th)||(avg_first>=upper_th)
    end
    info("$(sum(abnormal_shape)) abnormal events found")
    put_label!(events, :abnormal_shape, abnormal_shape)
    return events
end

export push_effective_e_label!
"""
	push_effective_e_label!(events::EventCollection, noise_scale::Union{Int64,Float32}=0)

Appends the effecive energy label to the data by dividing the energy by *noise scale*.
Returns *events* with the new label (:E_eff) appended. 
"""
function push_effective_e_label!(events::EventCollection, noise_scale::Union{Int64,Float32}=0)
    if typeof(events) == DLData
	lib = events.entries[1]
    else
	lib = events
    end
    energy_key = if haskey(lib, :E_eff); :E_eff else :E end 
    put_label!(events, :E_eff, convert(Array{Float32,1}, round.(events[energy_key] / (noise_scale + 1), 3)))
    return events
end

export create_mixed_data
"""
	 create_mixed_data(env,
 			   cal_data::DLData,
 			   phy_data::DLData,
 			   cal_data_noise::DLData,
 			   phy_data_noise::DLData;
 			   cut_method::String='cut',
 			   select_method::String='random',
 			   shuffle::Bool=true)

Creates a mixed dataset from cal and phy data in the given ratio.
Calibration data will be flagged with isCal=1, physics data with isCal=0.
"""
function create_mixed_data(env,
                            cal_data::DLData,
                            phy_data::DLData,
                            cal_data_noise::DLData,
                            phy_data_noise::DLData;
                            cut_method::String="cut",
                            select_method::String="random",
                            shuffle::Bool=true)
    result_pulses = DLData(EventLibrary[])
    result_noise = DLData(EventLibrary[])
    for (cd, cn, pd, pn) in zip(cal_data, cal_data_noise, phy_data, phy_data_noise)
         mixed_pulses, mixed_noise = create_mixed_data(env, cd, pd, cn, pn, cut_method, select_method, shuffle)
         push!(result_pulses.entries, mixed_pulses)
         push!(result_noise.entries, mixed_noise)
    end
end

function create_mixed_data(env,
    cal_lib::EventLibrary,
    phy_lib::EventLibrary,
    cal_lib_noise::EventLibrary,
    phy_lib_noise::EventLibrary;
    cut_method::String="cut",
    select_method::String="random",
    shuffle::Bool=true)

# append missing labels to physics dataset (effective energy and abnormal pulse flag) with default values
    for (idx, key) in enumerate(keys(cal_lib))
    if key == :E_eff
        !(key in keys(phy_lib))&&put_label!(phy_lib, key, phy_lib[:E])
            !(key in keys(phy_lib_noise))&&put_label!(phy_lib_noise, key, phy_lib_noise[:E])
    else
            !(key in keys(phy_lib))&&put_label!(phy_lib, key, zeros(Int8, eventcount(phy_lib)))
            !(key in keys(phy_lib_noise))&&put_label!(phy_lib_noise, key, zeros(Int8, eventcount(phy_lib_noise)))
    end  
    end

    total = Int32(env.config["data_ratio"]["total"][1])
    phy_events = Int32(env.config["data_ratio"]["phy"][1]*total)
    cal_events = Int32(env.config["data_ratio"]["cal"][1]*total)

    @assert(phy_events == total - cal_events, "The ratio of physics and calibration data must add up to 1. Check 						       the json file.")
    @assert(cut_method in ["cut", "fill"], "Cut method must be either 'cut' or 'fill'.")
    @assert(select_method in ["random", "first"], "Select method must be either 'random' or 'fill'.")  

    if cut_method == "fill"
            phy_events = eventcount(phy_lib)
            cal_events = Int32(total - phy_events)
            info("There are only $(eventcount(phy_lib)) physics events. Using all $(eventcount(phy_lib)) physics events and filling up the rest from calibration data.")
    elseif cut_method == "cut"
            phy_events = eventcount(phy_lib)
            cal_events = Int32(floor(phy_events/env.config["data_ratio"]["phy"][1]*env.config["data_ratio"]["cal"][1]))
            total = Int32(phy_events + cal_events)
            info("There are only $(eventcount(phy_lib)) physics events. Reducing total event number to keep ratios of phy: $(phy_events/total), cal: $(cal_events/total).")
    end

    if select_method =="random"
        cal_idx = sample(1:eventcount(cal_lib),cal_events, replace = false)
        phy_idx = sample(1:eventcount(phy_lib),phy_events, replace = false)
    elseif select_method =="first"
        cal_idx = collect(1:cal_events)
        phy_idx = collect(1:phy_events)
    end
#filter events by indices
    filtered_cal_pulses = cal_lib[cal_idx]
    put_label!(filtered_cal_pulses, :isCal, convert(Array{Int8,1}, ones(cal_events)))
    filtered_phy_pulses = phy_lib[phy_idx]
    put_label!(filtered_phy_pulses, :isCal, convert(Array{Int8,1}, zeros(phy_events)))
    
    filtered_cal_noise = cal_lib_noise[cal_idx]
    put_label!(filtered_cal_noise, :isCal, convert(Array{Int8,1}, ones(cal_events)))
    filtered_phy_noise = phy_lib_noise[phy_idx]
    put_label!(filtered_phy_noise, :isCal, convert(Array{Int8,1}, zeros(phy_events)))    
#merge phy and cal events into one event library
    cal_and_phy_pulses = flatten(cat([filtered_cal_pulses, filtered_phy_pulses]))
    cal_and_phy_noise = flatten(cat([filtered_cal_noise, filtered_phy_noise]))

#shuffle events
    if shuffle
        shuffled_idx = randperm(total)
        cal_and_phy_pulses = cal_and_phy_pulses[shuffled_idx]
        cal_and_phy_noise = cal_and_phy_noise[shuffled_idx]
    end
    return cal_and_phy_pulses, cal_and_phy_noise
end

export make_2d_hist
"""
	make_2d_hist(pulses, noise, reconst;
        	title::AbstractString=nothing,
        	xmax=nothing, 
        	xmin=nothing,
        	xsteps=nothing,
        	ymax=nothing,
        	ymin=nothing,
        	ysteps=nothing,
        	filename=nothing)

Method for calculating standard deviations of the reconstruction errors and the noise.
Plots the difference between the std error and std noise on a histogram.
"""
function make_2d_hist(pulses::EventCollection, noise::EventCollection, reconst::EventCollection;
	method::String="ratio",
        title::String="",
        xmax=nothing, 
        xmin=nothing,
        xsteps=nothing,
	xlabel=nothing,
        ymax=nothing,
        ymin=nothing,
        ysteps=nothing,
        filename=nothing, datatype="All")
    @assert method in ["diff", "ratio"] "Type of error must be either *diff* or *ratio*."
    pyplot_config()
#calculate errors
    std_reconst, std_noise = calculate_deviation(pulses, reconst, noise)
    if method == "ratio"
    	sigmas = std_reconst./std_noise
#data statistics
    	stats = get_statistics(sigmas)
	center = 1
	y_label = L"$\mathrm{\sigma_{reconst}/\sigma_{noise}}$"
	y_max = center+15*stats["std"]
	y_min = max(0, center-5*stats["std"])
    elseif method == "diff"
    	sigmas = std_reconst.-std_noise
#data statistics
    	stats = get_statistics(sigmas)
	center = 0
	y_label = L"$\mathrm{\sigma_{reconst}-\sigma_{noise}}$"
 	y_max = center+2*stats["std"]
	y_min = center-2*stats["std"]
    end
#set plot window range
    (ymax == nothing) && (ymax = y_max)
    (ymin == nothing) && (ymin = y_min)
    (ysteps == nothing) && (ysteps = 401)
    err_axis=linspace(ymin, ymax, ysteps)
    
    (xmax == nothing) && (xmax = 1.02maximum(pulses[:E_eff]))
    (xmin == nothing) && (xmin = 0)
    (xsteps == nothing) && (xsteps = 401)
    E_axis = linspace(xmin, xmax, xsteps)   
#make 2d histogram
    mse_hist = fit(Histogram{Float64}, (pulses[:E_eff], sigmas), (E_axis, err_axis), closed=:left) #data, edges (bins)
    #broadcast!(x -> x <= 0 ? NaN : log10(x), mse_hist.weights, mse_hist.weights)
    figure = plot(mse_hist, xticks = xmin:(xmax-xmin)/10:xmax, xrotation = 45)
    hline!(figure, [1], w=1, label="", color=:green)
    if xlabel == nothing
    	xaxis!("Energy [$(Int64(round(E_axis.step.hi))) keV/bin]")
    else
	xaxis!(xlabel*" [$(Int64(round(E_axis.step.hi))) keV/bin]")
    end
    yaxis!(y_label)
    (title != nothing) && title!(title) 
    (filename != nothing) && savefig(filename*"_$(method)")
#second 1d histogram
    sigma_dist_plot = plot_datatype_distributions(pulses, std_noise, std_reconst, err_axis, center=center, datatype=datatype)
    joint_fig = plot(figure, sigma_dist_plot, right_margin=10mm, size=(1600, 800), layout=@layout([a{0.7w} b]))
    (filename != nothing) && savefig(filename*"_$(method)_joint.png")
    return joint_fig, sigmas, stats
end

export plot_datatype_distributions
"""
	plot_datatype_distributions(pulses, std_noise, std_reconst,
			         sigmas, err_axis)

Plots the error distribution of the reconstruction errors by physics and calibration data.
"""
function plot_datatype_distributions(pulses, std_noise, std_reconst, err_axis; center = 1, datatype="All")
#make 1d histogram of all data
    mse_hist_1d = fit(Histogram{Float64},std_reconst./std_noise, err_axis, closed=:left)
    mse_hist_1d.weights ./= sum(mse_hist_1d.weights)
#make x and y equal length
    length(mse_hist_1d.weights)!=length(err_axis)&&push!(mse_hist_1d.weights, 0)
    #datatype = if pulses[:isCal][1]==1 "Calibration" else "Physics" end
    sigma_dist_plot = plot(mse_hist_1d.weights, err_axis, label="$datatype events", seriestype=:steppre, legend=:topright, title="", color=:blue, xflip=true)
#fit Gaussian
    std = if center == 1; .2 else .002 end
    mean = convert(Float64, center)
    fit_ = curve_fit(gaussian, err_axis, mse_hist_1d.weights, [.01, mean, std])
    fit_μ = formatter(fit_.param[2])
    fit_σ = formatter(fit_.param[3])
    if (length(string(fit_σ)) == 3)
	fit_σ = string(fit_σ)*"0"
    end
   if (length(string(fit_μ)) == 3)
	fit_μ = string(fit_μ)*"0"
    end
	plot!(sigma_dist_plot, gaussian(err_axis, fit_.param), err_axis,
          label="Gaussian fit\nμ = $(fit_μ) σ = $(fit_σ)", color=:red, legendfontsize=12)
    #yaxis!(grid=true, ytickfontcolor="white")
    yaxis!(L"$\mathrm{\sigma_{reconst}/\sigma_{noise}}$", yticks=([2,4,6]))
    xaxis!("Event ratio "*latexstring("[10^{-4}]"), xticks=(10e-4.*collect(1:1:3), collect(1:1:3)))
    return sigma_dist_plot
end

export group_events
"""
	group_events(reconst)

Groups the indices of the events according to several criteria based on the event labels.
Returns a dictionary where each keys are grouping criteria and values are a list of two elements.
The first one is an array containing the indices of pulses belonging to each group. The second is the array of group labels.
"""
function group_events(reconst::EventCollection)
    feature_groups = Dict()
    #event energy
    if typeof(reconst) == DLData
	lib = reconst.entries[1]
    else
	lib = reconst
    end
    if haskey(lib, :E)
        energy_groups = [find(E->(E<100)&&(E>=50), reconst[:E]),
                         find(E->(E<200)&&(E>=100), reconst[:E]),
                         find(E->(E<400)&&(E>=200), reconst[:E]),
                         find(E->(E<800)&&(E>=400), reconst[:E]),
		      	 find(E->(E>=800), reconst[:E])]
        energy_group_labels = ["50keV ≤ E < 100keV", "100keV ≤ E < 200keV", "200keV ≤ E < 400keV","400keV ≤ E < 800keV", "E ≥ 800keV"]
        feature_groups["energy"] = [energy_groups, energy_group_labels]
    end
  
    #decay slope
    if haskey(lib,:reconst_end_slope)
        slope_groups = [find(x -> x >= 0,reconst[:reconst_end_slope]),
                        find(x -> x < 0,reconst[:reconst_end_slope])]
        slope_group_labels = ["(+)", "(-)"].*" slope"
        feature_groups["slope"] = [slope_groups, slope_group_labels]
    end

        #cal phy
    if haskey(lib,:isCal)
        datatype_groups = [find(x -> x == 1,reconst[:isCal]),
                        find(x -> x == 0,reconst[:isCal])]
        datatype_group_labels = ["cal.", "phy."].*" pulse"
        feature_groups["datatype"] = [datatype_groups, datatype_group_labels]
    end

    if haskey(lib,:SE)
        datatype_groups = [get_ar39(reconst),
                           get_double_beta(reconst)]
        datatype_group_labels = [L"^{39}"*"Ar", "2νββ", "rest"]
        feature_groups["subset"] = [datatype_groups, datatype_group_labels]
    end
        
        #risetime
    if haskey(lib,:risetime)
        σ = std(reconst[:risetime])
        risetime_groups = [find(x ->(x<300)&&(x>=0), reconst[:risetime]),
                           find(x ->(x<600)&&(x>=300), reconst[:risetime]),
                           find(x ->(x>=600), reconst[:risetime])]
        risetime_group_labels = " rise time ".*["t < 300", "300 ≤ t < 600", "t ≥ 600"].*" ns"
        feature_groups["risetime"] = [risetime_groups, risetime_group_labels]
    end
    
        #aoe
    if haskey(lib,:AoE)
        aoe_groups = [find(x -> x>=0,reconst[:AoE]),
                        find(x -> x<0,reconst[:AoE])]
        aoe_group_labels = ["(+)", "(-)"].*" AoE param."
        feature_groups["aoe"] = [aoe_groups, aoe_group_labels]
    end

    if haskey(lib,:augment)
        augment_groups = [find(x -> x==-1,reconst[:augment]),
                        find(x -> x==0,reconst[:augment]),
			find(x -> x==1,reconst[:augment])]
        augment_group_labels = ["data", "slow pulse", "augmented copy"]
        feature_groups["augment"] = [augment_groups, augment_group_labels]
    end
    return feature_groups
end

export quality_cuts
"""
Performs filtering of non-physical, pileup and coincident events.
"""
function quality_cuts(data::EventCollection)
    filter!(data, :E, x->(x>0).&(x<10000))
    filter!(data, :multiplicity, x->(x==1))
    filter!(data, :isPileup, x->(x==0))
    return data
end

export trim_events
"""
	trim_events(lib::EventLibrary, n::Int64=200000)

Keeps the first *n* events of the library and discards the rest.
"""
function trim_events(lib::EventLibrary, n::Int64=200000)
  @assert length(lib) > 0 "Empty lib."
  result = EventLibrary(identity)
  result.initialization_function = nothing 
  # draw n events randomly
  indices = sample(1:eventcount(lib), min(n, eventcount(lib)), replace = false)
  # Trim waveforms
  result.waveforms = waveforms(lib)[:,indices] # this initializes the entries
  # Trim labels
  for key in keys(lib.labels)
      result.labels[key] = lib.labels[key][indices]
  end

  # Adopt shared property values
  for key in keys(lib.prop)
    value = lib.prop[key]
    result.prop[key] = value
  end
  return result
end

export add_noise_to_pulses
function (env::DLEnv, data::DLData; noise_scale::Int64=5, entry::Union{Int64,Void}=1, n::Int64=100000, noise::Union{DLData, Void}=nothing)
    contaminated_data = DLData(EventLibrary[])
    if noise == nothing
   	for (i, lib) in enumerate(data)
		push!(contaminated_data.entries, add_noise_to_pulses(env, lib; noise_scale=noise_scale, n=n))
  	end
	return contaminated_data
    else
	contaminated_noise = DLData(EventLibrary[])
	for (i, lib) in enumerate(data)
		d, n = add_noise_to_pulses(env, lib; noise_scale=noise_scale, n=n, noise=noise.entries[i])
		push!(contaminated_data.entries, d)
		push!(contaminated_data.entries, n)
  	end
	return contaminated_data, contaminated_noise
    end
  	  
end

"""
    add_noise_to_pulses(env, events::EventLibrary;
                             noise_scale::Int64=5,
                             n::Int64=100000)

*events* is an EventLibrary object of pulses and the function mixes baseline noise to the waveforms of it.
*noise_scale* is the scale factor of the baseline noise. 
In general: E_eff=E/(noise_scale+1) (the +1 is bc. of the original baseline noise naturally on the signals)
E.g. mix noise of specific detector to signals of that detector.
If set to nothing, all entries will be used.
*n* specifies the number of data pulses to return with added noise.
Returns a DLData object with 1 entry containing the noise contaminated pulses.
"""
function add_noise_to_pulses(events::EventLibrary;
                             noise_scale::Union{Int64,Float32}=5,
                             n::Int64=100000, noise::Union{EventLibrary, Void}=nothing)

    @assert (n>0)&&(noise_scale>=0)
# load noise data if not specified in input
    remote = "/remote/ceph/user/p/pkicsiny"
    info("Initializing $(events[:detector_name]) events and baseline traces.")
    baseline_traces = initialize(lazy_read_library(joinpath(remote, "baseline/$(events[:detector_name]).h5"), "$(events[:detector_name])"))
    @assert events[:detector_name] == baseline_traces[:detector_name] "Data and baseline detectors don't match."
    info("Using baseline noise of $(baseline_traces[:detector_name]) for contaminating $(events[:detector_name]) pulses.")
    events = events[1:min(n,eventcount(events))]
    baseline_traces = baseline_traces.waveforms[1:size(waveforms(events))[1],:]
# mean of each noise trace
    noise_means = mean(baseline_traces, 1)
# substract mean
    baseline_traces .-= noise_means
# filter abnormal noise traces (full 0 or has a slope)
    baseline_traces = baseline_traces[:,find((mean(baseline_traces,1).!=0).&(std(baseline_traces, 1).<20))]
# sample random noise traces
    noise_idx = sample(1:size(baseline_traces)[2], eventcount(events), replace = false)
#add baseline
    if noise != nothing
	noise_length = size(waveforms(noise))[1]
        noise_event_rms = std(waveforms(noise), 1) # vector of eventcount
	detector_baseline_rms = mean(std(baseline_traces, 1)) # one number
	noise_scale = noise_event_rms./detector_baseline_rms
	info("Adding baseline traces to the $(noise_length) long event noise traces.")
    	noise.waveforms .+= noise_scale.*baseline_traces[1:noise_length, noise_idx] #baseline trace amplitude is scaled to event noise trace amplitude
	events.waveforms .+= noise_scale.*baseline_traces[:, noise_idx]
	push_effective_e_label!(events, 1) # 1x baseline noise added
	push_effective_e_label!(noise, 1) # 1x baseline noise added
	@assert events[:E] == noise[:E] "Energies after noising don't match."
	result = [events, noise]
    else
    	events.waveforms .+= noise_scale*baseline_traces[:, noise_idx]
    	push_effective_e_label!(events, noise_scale)
	result = events
    end
    info("Deallocate baseline data.")
    baseline_traces = nothing
    gc()
    return result
end

export correct_preprocessing_flags!
"""
    correct_preprocessing_flags!(pulses, noise)

Sets flag entries in noise set to 1 where the corresponding pulse entry has 1.
"""
function correct_preprocessing_flags!(pulses::DLData, noise::DLData)
	pulses = deepcopy(pulses)
            for i in threadpartition(1:length(pulses))
	 	lib = pulses.entries[i]
	 	info("Current library: $(lib.prop[:detector_name])")
   	 	for key in keys(lib)
	     		# if label is missing from noise, copy labels from pulses
	     		if !haskey(noise.entries[i],key)
				info("Appending $(key) label to noise.")
				put_label!(noise.entries[i], key, lib[key])
	    		end
	    		# if label is different in noise, copy labels from pulses
   	    		if (noise.entries[i][key] != lib[key])&&(key !=:preprocessing)
   	   			info("Correct *$(key)* key.")
   	   			noise.entries[i][key] .= lib[key] # works for eventlibs but dldata is immutable
   	   		end
   	 	end
	     	info("Free up allocated RAM.")
	     	dispose(pulses.entries[i])
	     end
end

export post_process
"""
Checks if the labels of *pulses* and *noise* are equal. Correct the failed preprocessing labels for the *noise* bc. they are all 0 but for the *pulses* they are not. Also applies some additional filters for the failed pp., abnormal shape (peak before rise) and energy.
"""
function post_process(pulses::DLData, noise::DLData, detector_names; data_folder="", save_folder="", emin=50, emax=9999)
    for det in detector_names
        info("Processing $(det)...")
        ch_data = pulses[:detector_name=>det]
        ch_noise = noise[:detector_name=>det]
        for key in keys(ch_data)
            # if label is missing from noise, copy labels from pulses
            if !haskey(ch_noise,key)
                info("Appending $(key) label to noise.")
                put_label!(ch_noise, key, ch_data[key])
            end
            # if label is different in noise, copy labels from pulses
            if (ch_noise[key] != ch_data[key])&&(key !=:preprocessing)
                info("Correct *$(key)* key.")
                ch_noise[key] .= ch_data[key] # works for eventlibs but dldata is immutable
            end
        end
	info("Filter failed and abnormal pulses.")
        post_filter!(ch_data,[:FailedPreprocessing, :abnormal_shape],emin,emax, save_folder=dir(save_folder, det))
        post_filter!(ch_noise,[:FailedPreprocessing, :abnormal_shape],emin,emax)
        
	info("Save post filtered files.")
	pulses_dir = joinpath(data_folder*"_pulses")
	noise_dir = joinpath(data_folder*"_noise")
	isdir(pulses_dir)||mkdir(pulses_dir)
	isdir(noise_dir)||mkdir(noise_dir)
        write_lib(ch_data, joinpath(pulses_dir, "$(ch_data[:detector_name]).h5"), true)
        write_lib(ch_noise, joinpath(noise_dir, "$(ch_noise[:detector_name]).h5"), true)
        @assert maximum(ch_data[:FailedPreprocessing]) == 0
        @assert ch_data[:E]==ch_noise[:E]
        @assert ch_data[:FailedPreprocessing]==ch_noise[:FailedPreprocessing]
        
        dispose(pulses[:detector_name=>det])
        dispose(noise[:detector_name=>det])
        gc()
    end
end

export get_run
function get_run(env::DLEnv, pulses::EventCollection, noise::EventCollection; run=92)
    run_data = pulses[find(pulses[:run].==run)]
    run_noise = noise[find(noise[:run].==run)]
    return run_data, run_noise
end

export latent_space_plots
"""
    latent_space_plots(compact,
                       latent_size::Int64,
                       filepath::String,
                       feature_groups::Union{Dict{Any,Any},Void}=nothing;
		       use_cornerplot=false)

Method for plotting all permutation pairs of the latent vector components.
*compact* are the latent representations.
*latent_size* is the length of a latent vector.
*filepath* is the path to save the plots.
*feature_groups* is either 'nothing' or a dict of indices and group names.
"""
function latent_space_plots(compact::EventCollection,
                            latent_size::Int64,
                            filepath::String,
                            feature_groups::Union{Dict{Any,Any},Void}=nothing;
			    use_cornerplot=false)
    pyplot_config()
    plots = []
    #get all combinations of length 2
    pairs = filter(x -> length(x) == 2, [65-findin(bits(i),'1') for i=1:(2^latent_size-1)])
    #make new folder for latent correlation plots
    corr_folder = filepath*"/latent_space_correlations"
    isdir(corr_folder) || mkdir(corr_folder)
    n_events = eventcount(compact)
    #loop over pairs
    if use_cornerplot==false
    	for pair in pairs
        #loop over predefined groups
        	if feature_groups == nothing
        	    fig = latent_space_plot(compact, n_events, pair[2], pair[1])
        	        xlabel!(latexstring("\\Phi_$(pair[2])")*L"(\mathbf{x})")
        	        ylabel!(latexstring("\\Phi_$(pair[1])")*L"(\mathbf{x})")
        	        savefig("$(corr_folder)/$(pair[2])_$(pair[1])")
		    push!(plots, fig)
      	 	else
      	      	  	for (key, value) in feature_groups
      	      	  	    fig = latent_space_plot(compact, n_events, pair[2], pair[1], value[1], value[2])
      	     	  	    xlabel!(latexstring("\\Phi_$(pair[2])")*L"(\mathbf{x})")
        	            ylabel!(latexstring("\\Phi_$(pair[1])")*L"(\mathbf{x})")
      	    	  	    savefig("$(corr_folder)/$(key)_$(pair[2])_$(pair[1])")
			    push!(plots, fig)
      	    	 	 end
      	 	end
	  end
     else
	labels = [latexstring("\\Phi_$(i)")*L"(\mathbf{x})" for i in 1:latent_size]
	savefig(cornerplot(transpose(waveforms(compact)), compact=true, linealpha=0,
	label=labels, xrotation = 45), "$(corr_folder)/cornerplot")
    end
    return plots
end

export latent_space_plot
"""
    latent_space_plot(latent_space::EventLibrary,
                      n_events::Int,
                      latent_x::Int,
                      latent_y::Int,
                      groups::AbstractArray,
                      group_labels::AbstractArray)

Makes 2D plots of latent space vector components. PLots one against the other for *n_events* events.
*latent_space* contains the compact representations.
*n_events* determines how many events to plot.
*latent_x* and *latent_y* are the latent vector components as vecot indices to plot against each other.
*groups* is an array of arrays where each subarray contains the indices of events belonging to that group.
*group_labels* is an array of strings where the strings are the group identifiers.
Returns the plot as a Plots.Plot{Plots.PyPlotBackend} object.
"""
function latent_space_plot(latent_space::EventCollection,
                           n_events::Int,
                           latent_x::Int,
                           latent_y::Int,
                           groups::Union{AbstractArray,Void}=nothing,
                           group_labels::Union{AbstractArray,Void}=nothing)
    
    fig = scatter()
    colors = [:blue, :red, :green, :purple, :orange]
    mini = minimum(waveforms(latent_space)[latent_y,1:min(eventcount(latent_space), n_events)])
    maxi = maximum(waveforms(latent_space)[latent_y,1:min(eventcount(latent_space), n_events)])
    if groups != nothing && group_labels !=nothing
    	for grp in range(1,length(groups))    
        	group = try latent_space[groups[grp]]; catch group = groups[grp] end
        	scatter!(fig, waveforms(group)[latent_x,1:min(eventcount(group), n_events)],
        	      waveforms(group)[latent_y,1:min(eventcount(group), n_events)],
        	      seriestype=:scatter, label="$(group_labels[grp])", marker = (:+,5), color=colors[grp])
		ylims!((-Inf, 1.4*maxi))
    	end
    else
	scatter!(fig, waveforms(latent_space)[latent_x,1:min(eventcount(latent_space), n_events)],
        	      waveforms(latent_space)[latent_y,1:min(eventcount(latent_space), n_events)],
        	      seriestype=:scatter, label="", marker = (:+,5), color=colors[1])
	ylims!((-Inf, 1.4*maxi))
    end
    return fig
end

export get_palette
"""
Returns a color palette as an array of RGBA items.
"""
function get_palette(which::Symbol=:redblue; n=31)
    @assert which in [:redblue, :rainbow] "Parameter *which* mus be either :redblue or :rainbow."
    lgbt = []
    if which == :rainbow
        scale = round(255/n, 2)
        [push!(lgbt, RGBA(scale*i/255,0/255,0/255,1)) for i in 1:5]
        [push!(lgbt, RGBA(255/255,scale*i/255,0/255,1)) for i in 0:5]
        [push!(lgbt, RGBA(scale*(5-i)/255,255/255,0/255,1)) for i in 0:5]
        [push!(lgbt, RGBA(0/255,255/255,scale*i/255,1)) for i in 0:5]
        [push!(lgbt, RGBA(0/255,scale*(5-i)/255,255/255,1)) for i in 0:5]
        [push!(lgbt, RGBA(0/255,0/255,scale*(5-i)/255,1)) for i in 0:2]
    elseif which == :redblue
        scale = round(255/n, 1)
        [push!(lgbt, RGBA(scale*i/255,0/255,scale*(n-i)/255,1)) for i in 1:n]
    end
    colorpalette = reshape(Array(unique(lgbt)), (1,n))
    return colorpalette
end

export pulse_decay_slopes!
"""
    pulse_decay_slopes!(data::EventLibrary,
                          length_pf_slope=150)

Calculates the decay slope of the pulses with least squares and appends a
new label whether the slope is positive or negative.
Source: https://dsp.stackexchange.com/questions/42364/find-smoothed-first-derivative-from-signal-with-noisy-slope
*reconst* is containing the autoencoder reconstructions.
*length_of_slope* determines the last n samples of each reconstruction
waveforms to be considered when calculating the end slope.
Returns the EventLibrary extended with a new label.
"""
function pulse_decay_slopes!(data::EventLibrary,
                               length_of_slope::Int64=150, label_key::Symbol=:decay_slope)
    
    slopes = zeros(Float32, eventcount(data))
    x = collect(1:1:length_of_slope)  # x
    for event in range(1,eventcount(data))
        noisy_end = data.waveforms[:,event][end-length_of_slope+1:end]  # y
        slopes[event] = convert(Float32,sum((x - mean(x)).*(noisy_end - mean(noisy_end)))/sum((x - mean(x)).*(x - mean(x))))
    end
    put_label!(data, label_key, slopes)
end

function pulse_decay_slopes!(data::DLData, length_of_slope::Int64=150, label_key::Symbol=:decay_slope)
    for lib in data
        pulse_decay_slopes!(lib, length_of_slope, label_key)
    end
end

export layer_info
"""
    layer_info(layer::MXNet.mx.SymbolicNode,
               input_shape::NTuple{4,Int64})

Prints the number of trainable weights and biases in the given layer,
those in total up until the given layer and the output shape of the given layer.
*layer*'s info is to be printed.
*input_shape* is a tuple of (height, width, channels, batch size) describing the input shape.
"""
function layer_info(layer::MXNet.mx.SymbolicNode, input_shape::NTuple{4,Int64})
    layer_tuples = mx.infer_shape(layer,input_shape)
    sum = 0
    for tuple in layer_tuples[1][2:end-1] sum += Base.prod(tuple) end
    info("\nLayer weights: ", layer_tuples[1][2:end-1], "\nWeights up to this layer:", sum, "\nOutput shape: ",layer_tuples[2][:])
    return sum
end

export plot_reconstructions
"""
    plot_reconstructions!(reconst::EventLibrary,
                          datatype::String,
                          pulse_type::String,
                          n_events::Int64,                      
                          indices::Union{AbstractArray,Colon}=:,
                          label::String="",
                          color::Union{String,Void}=nothing,
                          save::Union{Bool,Void}=false,
                          plot_title::Union{String,Void}=nothing)

Plots events on top of each other.
*reconst* contains the data to be plotted.
*datatype* is either cal, mix or phy.
*pulse_type* is either charge or current.
*n_events* specifies how many events shall be plotted.
*indices* is an array of integers specifying the indices of the data array, like a group. 
Considers all events by default.
*label* is a string for plot label.
*color* is the color of the plots.
*save* has to be set to true for saving the plot.
*plot_title* well, what could it be?
"""
function plot_reconstructions(reconst::EventLibrary,
                              datatype::String,
                              pulse_type::String,
                              n_events::Int64,                      
                              indices::Union{AbstractArray,Colon}=:,
                              label::String="",
                              color::Union{String,Void}=nothing,
                              save::Union{Bool,Void}=false,
                              plot_title::Union{String,Void}=nothing)
    if pulse_type == "charge"
        legend_pos = :bottomright
    else
        legend_pos = :topright
    end
    plot()
    if color == nothing
        plot!(waveforms(reconst[indices])[:,1],label=label, legend=legend_pos)
        plot!(waveforms(reconst[indices])[:,2:n_events],label="")
    else
        plot!(waveforms(reconst[indices])[:,1],label=label, color=color, legend=legend_pos)
        plot!(waveforms(reconst[indices])[:,2:n_events],label="", color=color)
    end

    xlabel!("time [a.u.]")
    ylabel!("$(pulse_type) [a.u.]")
    
    save&&plot_title!=nothing&&savefig("$(plot_title).png")
end

export plot_reconstructions!
"""
    plot_reconstructions!(reconst::EventLibrary,
                          n_events::Int64,                      
                          indices::Union{AbstractArray,Colon}=:,
                          label::String="",
                          color::Union{String,Void}=nothing,
                          save::Union{Bool,Void}=false,
                          plot_title::Union{String,Void}=nothing)

See at plot_reconstructions.
"""
function plot_reconstructions!(reconst::EventLibrary,
                               n_events::Int64,                      
                               indices::Union{AbstractArray,Colon}=:,
                               label::String="",
                               color::Union{String,Void}=nothing,
                               save::Union{Bool,Void}=false,
                               plot_title::Union{String,Void}=nothing)
    if color == nothing
        plot!(waveforms(reconst[indices])[:,1],label=label)
        plot!(waveforms(reconst[indices])[:,2:n_events],label="")
    else
        plot!(waveforms(reconst[indices])[:,1],label=label, color=color)
        plot!(waveforms(reconst[indices])[:,2:n_events],label="", color=color)
    end
    
    save&&plot_title!=nothing&&savefig("$(plot_title).png")
end

export waveforms_2d_hist
"""
    waveforms_2d_hist(data::Union{DLData,EventLibrary},
                      n_events::Union{String,Void}=nothing,
                      ybins::Int64=100)
Plots waveforms on a 2D histogram.
*data* contains the waveforms to plot.
*n_events* specifies the number of waveforms (first n) to plot.
*ybins* is the number of y axis bins.
Returns the histogram as a Plots.Plot{Plots.PyPlotBackend} object.
"""
function waveforms_2d_hist(data::Union{DLData,EventLibrary},
                           n_events::Union{Int64,Void}=nothing,
                           ybins::Int64=100; folder::Union{String,Void}=nothing)
    pyplot_config()
    # waveform window length
    sample_size = size(waveforms(data))[1]
    if n_events == nothing
        n_events = eventcount(data)
    end
    events = waveforms(data)[:,1:n_events]
    histevents = reshape(events,n_events*sample_size)
    mesh = reshape([ i for i=1:sample_size, j=1:n_events ], (n_events*sample_size))
    x_axis = linspace(0, sample_size, sample_size)
    y_axis = linspace(minimum(events), maximum(events), ybins)
    data_hist = fit(Histogram{Float64}, (mesh, histevents), (x_axis, y_axis), closed=:left)
    figure = plot(data_hist)
    xlabel!("Time [μs]")
    xaxis!(xticks=([0, 100, 200, 300], [0, 1, 2, 3]))
    ylabel!("Charge [a.u.]")
    folder!=nothing&&savefig(joinpath(folder, "reconst_hist"))
    return figure
end

export prepare_for_training
function prepare_for_training(env::DLEnv, ch_data::EventLibrary, ch_noise::EventLibrary, n_events=1e7; augment=0)
    n_events = min(n_events,eventcount(ch_data))
    info("$n_events events will be split.")
    ch_data = DLData([ch_data])
    ch_noise = DLData([ch_noise])
    initialize(ch_data)
    initialize(ch_noise)
    #append risetime
    slow_pulse_threshold(ch_data, split_value=.99)
    risetime_threshold = ch_data.entries[1].prop[:risetime_threshold]
    #augment
    put_label!(ch_data, :augment, convert(Array{Int8,1}, -1.+zeros(eventcount(ch_data))))
    put_label!(ch_noise, :augment, convert(Array{Int8,1}, -1.+zeros(eventcount(ch_noise))))
    if augment > 0
    	ch_data, ch_noise = augment_data(ch_data, ch_noise, copy_n_times=Int64(augment), risetime_threshold=risetime_threshold, add_noise=true)
    end
    #append some labels
    pulse_decay_slopes!(ch_data, 150, :data_end_slope)
    label_surface_bulk!(ch_data, :SE)
    pulse_decay_slopes!(ch_noise, 150, :data_end_slope)
    label_surface_bulk!(ch_noise, :SE)
    #split data
    input_size = size(waveforms(ch_data))[1]
    cnfg = convert(Dict{AbstractString,Array{AbstractFloat,1}},env.config["pulses"]["sets"])
    ch_data = split(ch_data, cnfg)
    ch_noise = split(ch_noise, cnfg)
    return ch_data, ch_noise
end

export plot_spectrum
"""
    plot_spectrum(energies::Array{Float64,1};
                    title::Union{String,Void}=nothing,
                    xmin::Int64=0,
                    xmax::Int64=2500,
                    bins::Int64=250)

Plots energy spectrum on histogram.
*energies* is an 1D array of floats with the event energies.
*title* will be the histogram title.
*xmin* is the lowest energy bin.
*xmax* is the highest energy bin.
*bins* is the number of bins. A rounded number + 1 is suggested for nice bin divisions.
"""
function plot_spectrum(libs::EventCollection...;
                         folder::Union{String,Void}=nothing,
                         xmin::Int64=0,
                         xmax::Int64=2500,
                         bins::Int64=250,
			 key::Symbol=:E,
			 xlabel::String="Energy [keV]",
			 labels=["" ""],
                         kwargs...)
    pyplot_config()
    bin = linspace(xmin, xmax, bins)
    fig = stephist()
    colors = [:blue, :red, :green, :purple]
    for (idx,lib) in enumerate(libs)
	stephist!(fig, lib[key], bin=bin, color=colors[idx], label=labels[idx], w=1.5, size=(1200,600); kwargs...)
    end
    yaxis!("Counts [$(Int64(round((xmax-xmin)/(bins-1)))) keV/bin]")
    xaxis!(xlabel)
    folder!=nothing&&savefig(joinpath(folder,"e_spectrum"))
    return fig
end

export set_optimizer
"""
    set_optimizer(optimizer_name::String, learning_rate::Float64; gamma::Float64=0.9)

*optimizer_name* is e.g. 'ADAM' or 'SGD'.
*gamma* is the learning rate decay, currently for ADAM optimizer.
"""
function set_optimizer(optimizer_name::String, learning_rate::Float64; gamma::Float64=0.9)
  if optimizer_name == "SGD"
    optimizer = mx.SGD(lr=learning_rate, momentum=n["momentum"])
  elseif optimizer_name == "ADAM"
    if gamma < 1
    	optimizer = mx.ADAM(lr=learning_rate, lr_scheduler=mx.LearningRate.Exp(learning_rate, gamma=gamma))
    else
	optimizer = mx.ADAM(lr=learning_rate)
     end
  elseif optimizer_name == "None"
    optimizer = mx.SGD(lr=1e-9)
  end
  return optimizer
end

export risetime_dist
"""
Not used
"""
function risetime_dist(events::EventCollection; folder::Union{String, Void}=nothing)
    pyplot_config()
    xmin = 0
    xmax = maximum(events[:risetime])
    bins = 400
    bin = linspace(xmin, xmax, bins)
    stephist(bin, events[:risetime], yscale=:log10, label="", color=:blue) 

    vline!([parse(formatter(get_value(events, :risetime_threshold)))], label="Upper 1%", color=:red)
    xaxis!("Risetime [ns]")
    yaxis!("Counts [$(convert(Int64,round((xmax-xmin)/(bins-1)))) keV/bin]")
    title!("Ristetime distribution of $(events[:detector_name][1])")
    folder!=nothing&&savefig(joinpath(folder, "risetime_hist"))
    rt_x = collect(0:100:1000)
    rt_y = [length(find(events[:risetime].>t))/eventcount(events) for t in rt_x]
    scatter(rt_x,rt_y, label="", marker=(5,stroke(0)), color=:blue)
    plot!(rt_x,rt_y, label="", color=:blue)
    xaxis!("Risetime [ns]")
    yaxis!("Ratio of events")
    title!("Ristetime distribution of $(events[:detector_name][1])")
    folder!=nothing&&savefig(joinpath(folder, "risetime_ratios"))
#write file
    open(joinpath(folder,"risetimes.txt"), "w") do file
        writedlm(file, ["Risetime [ns], # of events, Ratio of events"])
        writedlm(file, ([rt_x Int64.(rt_y.*eventcount(events)) rt_y]), ", ")
    end
end

export slow_pulse_threshold
"""
Returns the risetime value at which *split_value* percent of all evnets have higher risetime.
"""
function slow_pulse_threshold(data::EventCollection; split_value=.97)
    set_property!(data, :risetime_threshold, Int64(round(sort(data[:risetime])[Int64(round(split_value*eventcount(data)))])))
end

export augment_data
"""
    augment_data(data::DLData, noise::DLData; risetime_threshold::Int64=800, copy_n_times::Int64=1, add_noise=false)

Replicates pulses with risetime larger than *risetime_threshold* and appends the copied pulses to *data*. Does the same with *noise*, using the same baseline traces as for *data*. The baseline traces get normalized with the relative RMS of the *noise* traces.
"""
function augment_data(data::DLData, noise::DLData; risetime_threshold::Int64=800, copy_n_times::Int64=1, add_noise=false)
    @assert copy_n_times>=1 "*copy_n_times* must be at least 1"
#get slow risetime pulses
    slow_pulse_indices = find(x->x>=risetime_threshold, data[:risetime])
    slow_pulses=filter(data, :risetime, x->x>=risetime_threshold)
    slow_noise=filter(noise, :risetime, x->x>=risetime_threshold)
    @assert eventcount(slow_pulses) == eventcount(slow_noise) "Different number of slow pulses found in *data* and *noise*."
#put augment label: -1 for non augmented, 1 for augmented copies, 0 for original augmentation candidates
    !haskey(try data.entries[1].labels catch; data end,:augment)&&put_label!(data, :augment, convert(Array{Int8,1}, -1.+zeros(eventcount(data))))
    !haskey(try noise.entries[1].labels catch; noise end,:augment)&&put_label!(noise, :augment, convert(Array{Int8,1}, -1.+zeros(eventcount(noise))))
    data[:augment][slow_pulse_indices] = 0
    noise[:augment][slow_pulse_indices] = 0
    info("Data samples before augmentation: $(eventcount(data))")
    info("$(eventcount(slow_pulses)) slow pulses will be augmented $(copy_n_times) times.")
# container of original pulses + added slow pulses
    augmented_data = DLData(EventLibrary[])
    augmented_noise = DLData(EventLibrary[])
# check E_eff label and if not present, add (in case of pure physics data)
    !haskey(try data.entries[1].labels catch; data end,:E_eff)&&push_effective_e_label!(data)
    !haskey(try noise.entries[1].labels catch; noise end,:E_eff)&&push_effective_e_label!(noise)
# loop over event libraries
    for (idx, lib) in enumerate(slow_pulses)
        # append the extra slow pulses to the detector eventlibrary
	if add_noise
		data_copies, noise_copies = copypaste_with_noise(lib, slow_noise.entries[idx], copy_n_times-1)
	else
		data_copies = copypaste_events(lib, copy_n_times-1)
		noise_copies = copypaste_events(slow_noise.entries[idx], copy_n_times-1)
		info(eventcount(slow_noise.entries[idx]))
	end
	data_copies[:augment] .= 1
	noise_copies[:augment] .= 1
        push!(augmented_data.entries, flatten(cat([data_copies,try data.entries[idx] catch; data end])))
	push!(augmented_noise.entries, flatten(cat([noise_copies,try noise.entries[idx] catch; noise end])))
    end
# shuffle events and return
    info("Data samples after augmentation: $(eventcount(augmented_data))")
    add_noise&&info("Noise samples after augmentation: $(eventcount(augmented_noise))")
    mixed_indices = randperm(eventcount(augmented_data))
    return DLData([augmented_data[mixed_indices]]), DLData([augmented_noise[mixed_indices]])
end

"""
Takes x clean events and returns n*x noisy events. Also augments the event noise traces with the same baseline trace instance scaled to the normalized relative rms.
"""
function copypaste_with_noise(events::EventLibrary, noise::EventLibrary, n::Int64)
    @assert n>=0
    return add_noise_to_pulses(copypaste_events(events, n), n=eventcount(events)*(n+1), noise_scale=1, noise=copypaste_events(noise, n))
end

"""
Takes x events and returns (n+1)*x events by n times copying the input.
"""
function copypaste_events(events::EventLibrary, n::Int64)
    @assert n>=0
    return if n == 0 events else cat_events(events, copypaste_events(events, n-1)) end
end

export smart_subplots
"""
    smart_subplots(indices::Array, data::EventLibrary...; labelfont=14, tickfont=10, titlefont=10)

Makes subplots on a smart grid layout.
*data* are EventLibrary objects. The events corresponding to *indices* will be plot on top of each other.
"""
function smart_subplots(indices::Array, data::EventCollection...; labels::Any=[L"\mathbf{x}" L"Ψ(Φ(\mathbf{x}))"], series=false, titles="")
    pyplot_config()
    if length(data) == 1 && typeof(data) == EventLibrary
        data = DLData([data])
    end
    # configure x axis
    t_axis = collect(sample_times(data[1]) * 1e6)
    t_axis -= t_axis[1]
    # configure layout
    # prime factorisation of n
    len = length(indices)
    layout = (1, 1)
    if len > 1 && !series
        primes = factor(Vector, len)
        # unique combinations of prime factors
        combinations = unique([primes[c] for c in [65-findin(bits(i),'1') for i=1:(2^length(primes)-1)]])
        # optimal combination ("most square shaped grid layout")
        opt = prod(combinations[findmin(abs.(prod.(combinations).-sqrt(len)))[2]])
        layout = Tuple(sort([opt, Int(floor(len/opt))]))
	pyplot(size=(layout[2]*600,layout[1]*500),legend=true)
    end
    
    # make plots
    plots = []
    #error_info = latexstring("σ_{reconst}/σ_{noise} = $(round(data[1][:error_ratio][i], 2))")
    colors=[:blue :red]
    for (idx, i) in enumerate(indices)
	if :E_eff in keys(data[1])
		info("Effective energy is shown.")
		title = "($(round(data[1][:E_eff][i], 1)) keV)"
	else
		#info("Normal energy is shown.")
		title = "($(round(data[1][:E][i], 1)) keV)"
	end
        push!(plots,plot(t_axis,[waveforms(data_)[:,i] for data_ in data], color=colors, label=labels,
 title=titles*title, legend=:topleft))
    end
    # put plots onto main plot
    if !series
    	fig = plot(plots..., layout=layout, w=2, label=labels, xlabel="Time [μs]", ylabel="Charge [a.u.]")
    else
	fig = []
    	for plt in plots
	    push!(fig, plot(plt, w=2, label=labels, xlabel="Time [μs]", ylabel="Charge [a.u.]"))
	end
    end
    return fig
end

export set_ae_hyperparams
"""
Sets the hyperparameters of the autoencoder.
"""
function set_ae_hyperparams(env, bs, lr, cf, cl, fc)
    env.config["autoencoder"]["batch_size"] = Int64(bs)
    env.config["autoencoder"]["learning_rate"] = lr 
    env.config["autoencoder"]["conv_filters"] = Int64(cf)
    env.config["autoencoder"]["conv_lengths"] = Int64(cl)
    env.config["autoencoder"]["fc"][1] = Int64(fc)
end

export set_dnn_hyperparams
"""
Sets the hyperparameters of the classifier.
"""
function set_dnn_hyperparams(env, bs, lr, ep)
    env.config["latent-dnn-classifier"]["batch_size"] = Int64(bs)
    env.config["latent-dnn-classifier"]["learning_rate"] = lr 
    env.config["latent-dnn-classifier"]["epochs"] = Int64(ep)
end

export get_ar39
function get_ar39(events::EventCollection, e_key::Symbol=:E)
    return find((events[e_key].>50).&
                (events[e_key].<300).&
		(events[:isCal].==0))
end

export get_roi
function get_roi(events::EventCollection, e_key::Symbol=:E)
	return find((events[e_key].>2039-50).&
                (events[e_key].<2039+50).&
		(events[:isCal].==0))
end

export get_augmented
function get_augmented(events::EventCollection, augment_key::Symbol=:augment)
    return find((events[augment_key].>-1))
end


export get_double_beta
function get_double_beta(events::EventCollection, e_key::Symbol=:E)
    return find((events[e_key].>600).&
                (events[e_key].<1300).&
                (events[:isLArVetoed].==0).&
                (events[:isCal].==0))
end

export get_lar
function get_lar(events::EventCollection, lar_key::Symbol=:isLArVetoed)
    return find((events[lar_key].==1).&
                (events[:isCal].==0))
end

export get_high_error
function get_high_error(events::EventCollection, error_key::Symbol=:error_ratio, error_min=-.5, error_max=2.5)
    return find((events[error_key].>error_max).|
		(events[error_key].<error_min))
end

export get_alpha
function get_alpha(events::EventCollection, e_key::Symbol=:E)
    return find((events[e_key].>3500).&
                (events[:isLArVetoed].==0).&
                (events[:isCal].==0))
end

export get_slow
function get_slow(events::EventCollection)
    return find((events[:risetime].>get_value(events, :risetime_threshold)))
end

export label_surface_bulk!
"""
    label_surface_bulk!(events::EventLibrary, label_key::Symbol)

Labels surface and bulk events based on Ar39 and 2vbb events. Does not include LAr and Muon vetoed events.
"""
function label_surface_bulk!(events::EventLibrary, label_key::Symbol)
    labels = zeros(Int8, eventcount(events)) .-1
    ar39_surface_events = get_ar39(events)

    double_beta_events = get_double_beta(events)

    labels[ar39_surface_events] = 1
    labels[double_beta_events] = 0
    
    put_label!(events, label_key, labels)
end

function label_surface_bulk!(data::DLData, label_key::Symbol)
    for lib in data
        label_surface_bulk!(lib, label_key)
    end
end

export dnn_predictions_vs_energy
function dnn_predictions_vs_energy(events::EventLibrary,
                           groups::Union{AbstractArray,Void}=nothing,
                           group_labels::Union{AbstractArray,Void}=nothing;
			   title="scatter",
			   threshold=0.5,
			   show_threshold=true,
			   folder::Union{String,Void}=nothing,
			   log=false)
    pyplot_config()
#get accuracies
    
    if log
    	log_dict = dnn_accuracies(events, truth_key=:SE, pred_key=:psd, threshold=threshold)
    	surface_as_surface = log_dict["correct_surf"]
    	surface_as_bulk = log_dict["fake_bulk"]
    	bulk_as_surface = log_dict["fake_surf"]
   	bulk_as_bulk = log_dict["correct_bulk"]
    	correct = surface_as_surface+bulk_as_bulk
  	wrong = surface_as_bulk+bulk_as_surface

	if sum(values(log_dict))>0
    		m0 = "With λ=$(round(threshold,2)) in test set:"
    		m1 ="$(surface_as_surface) surface events ($(round(100*surface_as_surface/(surface_as_surface+surface_as_bulk),2))%) classified as surface."
    		m2 = "$(surface_as_bulk) surface events ($(round(100*surface_as_bulk/(surface_as_surface+surface_as_bulk),2))%) classified as bulk."
    		m3 = "$(bulk_as_surface) bulk events ($(round(100*bulk_as_surface/(bulk_as_surface+bulk_as_bulk),2))%) classified as surface."
    		m4 = "$(bulk_as_bulk) bulk events ($(round(100*bulk_as_bulk/(bulk_as_surface+bulk_as_bulk),2))%) classified as bulk."
    		m5 = "Correctly classified events: $(correct) ($(round(100*(correct)/eventcount(events),2))%)"
    		m6 = "Misclassified events: $(wrong) ($(round(100*(wrong)/eventcount(events),2))%)"
    		m7 = "All test events: $(eventcount(events))"
#print accuracies
    		info("\n",m0,"\n",m1,"\n",m2,"\n",m3,"\n",m4,"\n",m5,"\n",m6,"\n",m7)
#save textfile of accuracies
    		if folder!=nothing
			m=[]
     			push!(m,m0,m1,m2,m3,m4,m5,m6,m7)
     			info("Save accuracies into file.")
     			open(joinpath(folder,"log.txt"),"w") do f
				for msg in m write(f, msg*"\n") end
    			end
    		end
	   end
    end	
#make scatterplot
    fig = scatter()
    colors = [:red, :blue, :green]
    labels=["Rise time ≥ 1563" "Augmented copy"]
    s=[(:+,5), (5,stroke(0))]
    if groups != nothing && group_labels !=nothing
        for grp in reverse(range(1,length(groups)-1)    )
            indices = groups[grp+1]
		info(grp)
	    scatter!(fig, events[:E][indices],events[:psd][indices], marker=s[grp], c=colors[grp], label=labels[grp]*" ($(length(events[:E][indices])))")
            #scatter!(fig, events[:E],events[:psd], marker_z=log10.(events[:risetime]), marker=(5,stroke(0)), c=:viridis, label="")
        end
    else
        scatter!(fig, data[:E], data[:psd], label="", marker=(5,stroke(0)))
    end
    xaxis!("E [keV]")
    yaxis!("Classifier output", yticks=[0, .2, .4, .6, .8, 1])
    #ylims!((0,1.4))
    #if show_threshold
    #	hline!([threshold],label="Cut threshold at $(threshold)", color=:red)
    #end
    #title!("Class predictions against energy for $(events[:detector_name])")
    folder!=nothing&&savefig(joinpath(folder,title))
    return fig
end

export dnn_class_distribution
function dnn_class_distribution(events::EventLibrary; truth_key::Symbol=:SE, pred_key::Symbol=:psd, cumulate=false, nbins=61, show_threshold=false, threshold::Union{Void,Float64}=nothing, folder::Union{String,Void}=nothing)
    pyplot_config()
    det = get_value(events, :detector_name)
    bins = linspace(0, 1.2, nbins)
    labels = events[truth_key]
    preds = events[pred_key]
    pred_surf = preds[find(x->x==1, labels)]
    pred_bulk = preds[find(x->x==0, labels)]
	xlims = (minimum(preds), maximum(preds))
	push!(pred_surf, 1.1)
	push!(pred_bulk, 1.1)
    surf_hist = fit(Histogram{Float64}, convert(Array{Float64},pred_surf), bins, closed=:right)
    bulk_hist = fit(Histogram{Float64}, convert(Array{Float64},pred_bulk), bins, closed=:right)
    xlims = (minimum(preds)-0.01, maximum(preds)+0.01)
    if cumulate
	surf_hist.weights = cumsum(surf_hist.weights)
	bulk_hist.weights = reverse(cumsum(reverse(bulk_hist.weights)))
    end
    surf_hist.weights ./= sum(surf_hist.weights)
    bulk_hist.weights ./= sum(bulk_hist.weights)
    fig = plot([surf_hist, bulk_hist], color=[:blue :red], linewidth=0,
  fillalpha=0.7,label=[latexstring("^{39}")*"Ar" "2νββ"])
    xlims!(xlims)
    if show_threshold
	if !cumulate
		surf_hist.weights = cumsum(surf_hist.weights)
		bulk_hist.weights = reverse(cumsum(reverse(bulk_hist.weights)))
        end
	edges = collect(bins)
	least_sum = findmin(surf_hist.weights .+ bulk_hist.weights)[2]
	least_diff = findmin(abs.(surf_hist.weights .- bulk_hist.weights))[2]
	threshold_1 = .5*(edges[least_sum] + edges[least_sum+1])
	threshold_2 = .5*(edges[least_diff] + edges[least_diff+1])
	vline!([threshold_1], label="Least sum at $(round(threshold_1,2))")
	vline!([threshold_2], label="Least diff at $(round(threshold_2,2))")
    end

    #if threshold != nothing
	#vline!([threshold], w=2, label="Classifier cut at $threshold", color=:red)
    #end

    xaxis!("Classifier output")
    yaxis!("Event ratio")
    #title!("Class predictions of $(events[:detector_name])")
    folder!=nothing&&savefig(joinpath(folder, if cumulate "cum" else "classdist" end))
    return fig
end

export dnn_total_distribution
function dnn_total_distribution(events::EventLibrary; pred_key::Symbol=:psd, threshold::Union{Void,Float64}=nothing, nbins = 30, folder::Union{String,Void}=nothing)
    pyplot_config()
    bins = linspace(minimum(events.labels[pred_key]), maximum(events.labels[pred_key]), nbins)
    fig = histogram()
    histogram!(events[pred_key], bins=bins, linewidth=0, label="All events ($(eventcount(events)))")
    xaxis!("Classifier output")
    yaxis!("Counts")
    #if threshold != nothing
#	vline!([threshold], w=2, label="Classifier cut at $threshold", color=:red)
#    end
    #title!("Total class distribution of $(events[:detector_name])")
    folder!=nothing&&savefig(joinpath(folder,"total_dist"))
    return fig
end

export dnn_energy_dependence
function dnn_energy_dependence(events::EventCollection; pred_key::Symbol=:psd, threshold::Union{Void,Float64}=nothing, folder::Union{String,Void}=nothing, dist_label="Physics", step=5, emin=50, emax=5500)
    pyplot_config()
    E_axis = collect(emin:step:emax)
    y_axis = linspace(0, 1, 401)
    e_dep_hist = fit(Histogram{Float64}, (convert(Array{Float64},events[:E]),
        convert(Array{Float64},events[pred_key])), (E_axis, y_axis), closed=:right)
    #broadcast!(x -> x <= 0 ? NaN : log10(x), mse_hist.weights, mse_hist.weights)
    fig = plot(e_dep_hist, title="$(dist_label) set", colorbar_title="Counts", clims=(0,99))
    xaxis!("Energy [$(step) keV/bin]")
    yaxis!(yticks=([0,.2,.4,.6, .8, 1], ["", "", "", "", ""]))
    #yaxis!("Classifier output")
    dist = fit(Histogram{Float64}, (convert(Array{Float64},events[pred_key])), y_axis, closed=:left)
    if threshold != nothing
	hline!([threshold], w=1, label="", color=:green)
    end
#make x and y equal length
    dist.weights ./= sum(dist.weights)
    length(dist.weights)!=401&&push!(dist.weights, 0)
    dist_plot = plot(dist.weights, y_axis, label="$(dist_label) events",
 seriestype=:steppre, color=:blue, title="", legendfontsize=12, xflip=true)
    max_ratio = maximum(dist.weights)
    max_xtick = 1e1ceil(1e1*max_ratio*2)/2 # rounded to 5
    info(max_ratio, max_xtick)
    ylabel!("Classifier output")
    yaxis!(yticks=([0,.2,.4,.6, .8, 1]))
    xaxis!("Event ratio "*latexstring("[10^{-2}]"), xticks=(1e-2*collect(0:0.2*max_xtick:max_xtick), Int64.(collect(0:0.2*max_xtick:max_xtick))))
    xlims!(-Inf, 1e-2max_xtick+0.01)
    if threshold != nothing
	hline!([threshold], w=1, label="Classifier cut at $threshold", color=:green)
    end
    fig = plot(dist_plot, fig, plot(grid=false, axis=false), layout=@layout([a{0.3w} a{.7w} x{0.001w}]), size=(1200,500))
    ylims!(-0.02, 1.02)
    return fig, e_dep_hist
end

export dnn_time_dependence
function dnn_time_dependence(events::EventCollection; pred_key::Symbol=:psd, nbins=20,  folder::Union{String,Void}=nothing)
    pyplot_config()
    events = filter(events, :E, x->x>150)
    events_sorted = sort(events, :timestamp)
    sorted_times = (events_sorted[:timestamp]-events_sorted[:timestamp][1])/60/60/24
    bin_width = Int64(round(length(sorted_times)/nbins))
    start_run = minimum(events[:run])
    end_run = maximum(events[:run])
    xticks = vcat(0, cumsum([length(events[:run=>r][:timestamp]) for r in start_run:end_run])[1:end-1])
    fig = histogram2d(convert(Array{Float64}, 1:length(events_sorted[pred_key])),
                convert(Array{Float64},events_sorted[pred_key]), bins=nbins, colorbar_title="Counts")
    xaxis!("Run", xticks=(xticks, collect(start_run:1:end_run)))
    yaxis!("Classifier output\n[$(bin_width) events/bin]", yticks=collect(.2:.2:1))
    ylims!(0,1)
    cfig = plot(placeholder(), fig, placeholder(), layout=@layout([a{.001h}; b; c{.001h}]), size=(600,500))
    #folder!=nothing&&savefig(joinpath(folder,"time_hist"))
    return fig
end

export placeholder
placeholder(;title="") = plot(axis=false, grid=false, title=title)

export optimal_cut_value
function optimal_cut_value(events::EventLibrary; scientific_guess=.9, truth_key::Symbol=:SE, pred_key::Symbol=:psd, folder::Union{String,Void}=nothing, method::String="2nbb+ar39")
    @assert method in ["2nbb+ar39", "misc", "misclassification", "k40", "k42", "40k", "42k", "2nbb"] "Invalid method."
    pyplot_config()
    fig = plot()
    thresholds = collect(0:0.01:1)
    data = []
    ar39_data = []
    if method in ["misc", "misclassification"]
    	corrs = []
    	wros = []
    	for t in thresholds
    	    log_dict = dnn_accuracies(events, truth_key=truth_key, pred_key=pred_key, threshold=t)
    	    push!(corrs,log_dict["correct_surf"]+log_dict["correct_bulk"])
   	    push!(wros,log_dict["fake_surf"]+log_dict["fake_bulk"])
    	end
        opt = thresholds[findmin(wros)[2]]
        data = wros./eventcount(events)
	ylabel = "Misclassification ratio"
	title = "Ratio of misclassified events of $(events[:detector_name])"
    elseif method in ["2nbb+ar39"]
	_2nbb = events[find(events[:SE].==0)]
	tot_2nbb = length(_2nbb)
	_ar39 = events[find(events[:SE].==1)]
	tot_ar39 = length(_ar39)
	active_volume = scientific_guess
	for t in thresholds
	    _2nbb_surf = find(_2nbb[:psd].>=t)
	    _2nbb_bulk = find(_2nbb[:psd].<t)
	    tot_2nbb_surf = length(_2nbb_surf)
	    tot_2nbb_bulk = length(_2nbb_bulk)
 	    _ar39_surf = find(_ar39[:psd].>=t)
	    _ar39_bulk = find(_ar39[:psd].<t)
	    tot_ar39_surf = length(_ar39_surf)
	    tot_ar39_bulk = length(_ar39_bulk)
	    push!(ar39_data, tot_ar39_surf/(tot_ar39_surf+tot_ar39_bulk))
	    push!(data, tot_2nbb_bulk/(tot_2nbb_bulk+tot_2nbb_surf)) # survival ratio
	end
	opt = round(0.01*(findmin(abs.(data .- active_volume))[2]-1),2)
	ylabel = "Survival fractions"
	#title = "Survival ratios of $(events[:detector_name])"
	hline!([active_volume], label="", linestyle=:dash, w=2, color=:grey)
    end
    if method in ["2nbb+ar39"]
    	plot!(fig, thresholds, ar39_data, shape = :circle, label=latexstring("^{39}")*"Ar", color=:blue, marker=(5,stroke(0)), xlimits=(0,1))
    end
    plot!(fig, thresholds, data, shape = :circle, label="2νββ", color=:red, marker=(5,stroke(0)), xlimits=(0,1), legend=:topleft)
    vline!([opt], w=1, label="Classifier cut at $opt", color=:green)
    xaxis!("Classifier cut")
    ylims!(0,1.4)
    yaxis!(ylabel, yticks=[0, .5, scientific_guess, 1])
    folder!=nothing&&savefig(joinpath(folder,"cut_value"))
    return fig, opt
end

export dnn_randomness
function dnn_randomness(events::EventCollection; folder::Union{String,Void}=nothing)
    pyplot_config()
    x=[]
    y=[]
    t=[]
    for th in collect(0:0.01:1)
        log_dict=dnn_accuracies(events, truth_key=:SE, pred_key=:psd, threshold=th)

        all_bulk=log_dict["correct_bulk"]+log_dict["fake_surf"]
        all_surf=log_dict["correct_surf"]+log_dict["fake_bulk"]
        push!(x, log_dict["fake_surf"]/all_bulk)
        push!(y, log_dict["correct_surf"]/all_surf)
        push!(t,th)
    
    end
    fig = plot(x,y, shape = :circle,label="")
    xaxis!("Ratio of misclassified bulk (fake surface) events")
    yaxis!("Ratio of correctly classified surface events")
    #title!("Cut value dependence of surface event separation efficiency")
    folder!=nothing&&savefig(joinpath(folder,"acc"))
    return fig
end

export dnn_accuracies
function dnn_accuracies(events::EventCollection; truth_key::Symbol=:SE, pred_key::Symbol=:psd, threshold=.5)
    @assert threshold <=1 && threshold >= 0
    surface_as_surface = length(events[pred_key][find((events[pred_key].>=threshold).&(events[truth_key].==1))])
    surface_as_bulk = length(events[pred_key][find((events[pred_key].<threshold).&(events[truth_key].==1))])
    bulk_as_surface = length(events[pred_key][find((events[pred_key].>=threshold).&(events[truth_key].==0))])
    bulk_as_bulk = length(events[pred_key][find((events[pred_key].<threshold).&(events[truth_key].==0))])
    correct = surface_as_surface+bulk_as_bulk
    wrong = surface_as_bulk+bulk_as_surface
    log_dict = Dict("correct_surf" => surface_as_surface, "correct_bulk" => bulk_as_bulk, "fake_surf" => bulk_as_surface, "fake_bulk" => surface_as_bulk)
    return log_dict
end

export dnn_survived_events
function dnn_survived_events(events::EventCollection; threshold=.5, pred_key::Symbol=:psd, bins=100, 
    title="Survived events", folder::Union{String,Void}=nothing)
    @assert threshold <=1 && threshold >= 0
    pyplot_config()
    survived = events[find(events[pred_key].<threshold)]
    fig = plot_spectrum(events, survived,
    xmin=Int64(floor(minimum(events[:E]))), xmax=Int64(ceil(maximum(events[:E]))),
    labels=["After LAr veto ($(length(events[:E])) events)" "After classifier cut ($(length(survived[:E])) events)"],
    bins=bins, yscale=:log10)
    #title!(title*" for $(events[:detector_name])")
    folder!=nothing&&savefig(joinpath(folder,title))
    return fig
end

export dnn_aoe_plot
function dnn_aoe_plot(events::EventCollection; pred_key::Symbol=:psd, threshold=.7,
        title="AE distribution", folder::Union{String,Void}=nothing)
    @assert threshold <=1 && threshold >= 0
    pyplot_config()
    fig = plot(events[:E], events[:AoE], seriestype=:scatter, label="After LAr veto", marker=(5,stroke(0)), color=:grey)
    plot!(fig, events[:E][find(events[pred_key].<threshold)], events[:AoE][find(events[pred_key].<threshold)], seriestype=:scatter,
    label="After classifier cut",marker=(5,stroke(0)), color=:red)
    xaxis!("E [keV]")
    yaxis!("A/E")
    ylims!(-Inf, 1.45maximum(events[:AoE]))
    #title!(title*" for $(events[:detector_name])")
    #folder!=nothing&&savefig(joinpath(folder,title))
    return fig
end

export latent_surf_bulk
function latent_surf_bulk(events::EventCollection)
    pyplot_config()
	filter!(events, :risetime, x->x>0)
    fet = group_events(events)
    s=3
    surf=latexstring("^{39}")*"Ar"
    bulk="2νββ"
    f12 = scatter(waveforms(events[fet["subset"][1][1]])[1,:],
            waveforms(events[fet["subset"][1][1]])[2,:], marker=(s,stroke(0)), color=:blue, label=surf)
    
    f12 = scatter!(waveforms(events[fet["subset"][1][2]])[1,:],
            waveforms(events[fet["subset"][1][2]])[2,:], marker=(s,stroke(0)), color=:red, label=bulk)
    xlabel!(latexstring("\\Phi_1")*L"(\mathbf{x})")
    ylabel!(latexstring("\\Phi_2")*L"(\mathbf{x})")
    f13 = scatter(waveforms(events[fet["subset"][1][1]])[1,:],
            waveforms(events[fet["subset"][1][1]])[3,:], marker=(s,stroke(0)), color=:blue, label=surf)
    
    f13 = scatter!(waveforms(events[fet["subset"][1][2]])[1,:],
            waveforms(events[fet["subset"][1][2]])[3,:], marker=(s,stroke(0)), color=:red, label=bulk)
    xlabel!(latexstring("\\Phi_1")*L"(\mathbf{x})")
    ylabel!(latexstring("\\Phi_3")*L"(\mathbf{x})")
    f23 = scatter(waveforms(events[fet["subset"][1][1]])[2,:],
            waveforms(events[fet["subset"][1][1]])[3,:], marker=(s,stroke(0)), color=:blue, label=surf)
    
    f23 = scatter!(waveforms(events[fet["subset"][1][2]])[2,:],
            waveforms(events[fet["subset"][1][2]])[3,:], marker=(s,stroke(0)), color=:red, label=bulk)
    xlabel!(latexstring("\\Phi_2")*L"(\mathbf{x})")
    ylabel!(latexstring("\\Phi_3")*L"(\mathbf{x})")
    return f12, f13, f23
end

export latent_risetime
function latent_risetime(events::EventCollection)
    pyplot_config()
    s=3
	filter!(events, :risetime, x->x>0)
	f12 = scatter(waveforms(events)[1,:],
              waveforms(events)[2,:],
              marker=(s,stroke(0)),
              marker_z=log10.(events[:risetime]), c=:viridis, label="", colorbar=false)
	xlabel!(latexstring("\\Phi_1")*L"(\mathbf{x})")
	ylabel!(latexstring("\\Phi_2")*L"(\mathbf{x})")

	f13 = scatter(waveforms(events)[1,:],
              waveforms(events)[3,:],
              marker=(s,stroke(0)),
              marker_z=log10.(events[:risetime]), c=:viridis, label="", colorbar=false)
	xlabel!(latexstring("\\Phi_1")*L"(\mathbf{x})")
	ylabel!(latexstring("\\Phi_3")*L"(\mathbf{x})")
	
	f23 = scatter(waveforms(events)[2,:],
              waveforms(events)[3,:],
              marker=(s,stroke(0)),
              marker_z=log10.(events[:risetime]), c=:viridis, label="", colorbar=false)
	xlabel!(latexstring("\\Phi_2")*L"(\mathbf{x})")
	ylabel!(latexstring("\\Phi_3")*L"(\mathbf{x})")

	cbar_only = scatter(waveforms(events)[2,:],
              waveforms(events)[3,:],
              marker=(0,stroke(0)),
              marker_z=log10.(events[:risetime]), c=:viridis, label="", axis=false, grid=false, colorbar_title="Log"*latexstring("_{10}")*" rise time")
    return f12, f13, f23, cbar_only
end

export plot_training_curves
function plot_training_curves(n;folder::Union{String,Void}=nothing, xtext="Epochs", ytext="Mean squared error", scale_factor=256, net_type="ae")
    #set figures size  
    pyplot_config()
    scaled_train = n.training_curve[2:end]
    scaled_eval = n.xval_curve[2:end]
    if net_type == "ae"
	scaled_train ./= scale_factor
	scaled_eval ./= scale_factor
    end 
    fig = plot(scaled_train,label="Training set", color=:blue)
    plot!(fig, scaled_eval,label="Validation set", color=:red)
    xlabel!(xtext)
    ylabel!(ytext)
    ylims!(.99*min(minimum(scaled_train), minimum(scaled_eval)), min(1.01*max(maximum(scaled_train), maximum(scaled_eval)), 10))
    title!("")
    folder!=nothing&&savefig(joinpath(folder, "curves"))
    return fig
end

export dnn_dependent_hist
function dnn_dependent_hist(events::EventCollection; pred_key::Symbol=:psd, dependent_key=:risetime, threshold::Union{Void,Float64}=nothing, title="Risetime histogram", ylabel="Rise time [ns]", folder::Union{String,Void}=nothing)
    pyplot_config()
    hist = fit(Histogram{Float64}, (events[pred_key], events[dependent_key]),
        (linspace(0, 1, 400)   , linspace(1.05*minimum(events[dependent_key]), 1.05*maximum(events[dependent_key]), 400)), closed=:left)
    fig = plot(hist, label="", size=(1200, 500), colorbar_title="Counts", clims=(0, 50))
    xaxis!("Classifier output")
    yaxis!("Rise time "*latexstring("[10^{3}")*" ns]")
    #title!(title*" for $(events[:detector_name])")
    if threshold != nothing
        vline!([threshold], w=1, label="Classifier cut at $threshold", color=:green)
    end
    #folder!=nothing&&savefig(joinpath(folder,title))
    fig2 = plot(placeholder(), fig, placeholder(), layout=@layout([x{.0001h}; a; x{.0001h}]), size=(1200, 500))
	yaxis!(yticks=(1e3*[1, 2, 3], [1, 2, 3]))
	ylims!(0, 3400)
    return fig2
end

export dnn_dependent_colorplot
function dnn_dependent_colorplot(events::EventCollection; pred_key::Symbol=:psd,
        dependent_key=:risetime, threshold::Union{Void,Float64}=nothing, title="",
        ylabel="Rise time [ns]", folder::Union{String,Void}=nothing)
    pyplot_config()
#colorplot
    cond1 = get_ar39(events)
    cond2 = get_double_beta(events)
    cond3 = get_alpha(events)
    colors = [:blue, :red, :green]
    colorplot = plot()
    (dependent_key == :risetime)&&scatter!(events[cond1][pred_key], events[cond1][dependent_key], marker=(5,stroke(0)), label=L"$^{39}$Ar", color=colors[1])
    scatter!(events[cond2][pred_key], events[cond2][dependent_key], marker=(5,stroke(0)), label="2νββ", color=colors[if dependent_key == :risetime 2 else 1 end])
    scatter!(events[cond3][pred_key], events[cond3][dependent_key], marker=(5,stroke(0)), label="α [E > 3.5 MeV]", color=colors[if dependent_key == :risetime 3 else 2 end])
    xaxis!("Classifier output")
    yaxis!(ylabel)
    if threshold != nothing
        vline!([threshold], w=1, label="Classifier cut\nat $threshold", color=:green)
    end
    #folder!=nothing&&savefig(joinpath(folder,title))
    return colorplot
end

export dnn_plots
"""
Makes plots for the classifier test set.
"""
function dnn_plots(n, events::EventLibrary; folder::Union{String,Void}=nothing)
    pyplot_config()
    plots = []
    fig, opt = optimal_cut_value(events, folder=folder)
    push!(plots, fig)
    feature_groups = group_events(events)
    #push!(plots, dnn_predictions_vs_energy(events, feature_groups["augment"][1], feature_groups["augment"][2], title="test_scatter",log=false, threshold=opt, folder=folder))
    push!(plots, dnn_class_distribution(events, nbins=101, folder=folder))
     #push!(plots, dnn_total_distribution(events, folder=folder))
    #push!(plots, dnn_time_dependence(events, x="day", nbins=20, folder=folder))
     #push!(plots, dnn_class_distribution(events, cumulate=true, nbins=201, folder=folder))
    push!(plots, plot_training_curves(n, folder=folder, net_type="dnn"))
     #push!(plots, dnn_energy_dependencg(events, threshold=opt, folder=folder))
    push!(plots, latent_surf_bulk(events))
    push!(plots, latent_risetime(events))
    return plots, opt
end

export dnn_result_plots
function dnn_result_plots(compact::EventCollection; threshold::Float64=.5, folder::Union{String,Void}=nothing)
    plots = []
    #plots for the whole physics set
    feature_groups = group_events(compact)
    #push!(plots, dnn_predictions_vs_energy(compact, feature_groups["risetime"][1], feature_groups["risetime"][2], threshold=threshold, folder=folder))
    #push!(plots, dnn_total_distribution(compact, nbins=100, threshold=threshold, folder=folder))
    #push!(plots, dnn_survived_events(compact, threshold=threshold, bins=200, folder=folder))
    aoeplot = dnn_aoe_plot(compact, threshold=threshold, folder=folder)
    aoecolorplot =  dnn_dependent_colorplot(compact, dependent_key=:AoE, threshold=threshold, folder=folder, title="AE_color", ylabel="A/E parameter")
    push!(plots, dnn_time_dependence(compact, nbins=20, folder=folder))
    push!(plots, plot(plot(aoecolorplot, ylims=(1.1minimum(compact[:AoE]), 2maximum(compact[:AoE]))),
		      plot(aoeplot, ytickfontsize=0, ylabel="", ylims=(1.1minimum(compact[:AoE]), 2maximum(compact[:AoE]))),
		      layout=(1,2), size=(1200,500)))
    push!(plots, dnn_dependent_hist(compact, dependent_key=:risetime, threshold=threshold, folder=folder))
    #push!(plots, dnn_dependent_hist(compact, dependent_key=:AoE, threshold=threshold, folder=folder, title="", ylabel="A/E parameter"))
    #push!(plots, dnn_dependent_colorplot(compact, dependent_key=:risetime, threshold=threshold, folder=folder))
    push!(plots, dnn_energy_dependence(compact, threshold=threshold, folder=folder))
    
    return plots
end

export ae_plots
"""
Makes plots for the autoencoder. Everything  is plotted for events from all three sets. This is justified by the similar reconstruction errors and that we are physicists.

*ch_data*: all three datasets of the channel
*ch_noise*: all three datasets of the channel noise
*n*: trained autoencoder model
*folder*: output directory
*scale_factor*: waveform scaling factor
"""
function ae_plots(n, ch_data::DLData, ch_noise::DLData; folder::Union{String,Void}="ae_plots", stats_dir::Union{String,Bool}="plots", scale_factor::Int64=256, make_stats=true)
    pyplot_config()
#get reconstructions for all events
    ch_compact, ch_reconst, feature_groups = get_reconst(ch_data, n)
#to error folder
    info("Plot error distributions.")
    stats_dir_1 = make_stats&&joinpath(stats_dir, "error_stats")
    reconst_error_dict, ch_data, ch_noise = plot_sigmas(n, ch_data, ch_noise, folder=folder, stats_dir=stats_dir_1)
#risetime distribution
    info("Plot risetime distribution.")
    #risetime_dist(ch_data; folder=folder)
#to end_slope folder
    info("Plot end slope distribution")
    stats_dir_2 = make_stats&&joinpath(stats_dir, "endslope_stats")
    end_slope_distribution(ch_data, ch_reconst, folder=folder, stats_dir=stats_dir_2)
#to sample_avg folder
    info("Plot sample average distribution")
    stats_dir_3 = make_stats&&joinpath(stats_dir, "sample_avg_stats")
    save_sample_avg_plots(ch_data, ch_reconst, folder=folder, stats_dir=stats_dir_3)
#to ae folder
    info("Plot training curves.")
    plot_training_curves(n, folder=folder, det=get_value(ch_data, :detector_name))
    info("Plot convolution filters.")
    visualize_1D_convolution(n.model, :conv_1_weight, joinpath(folder, "filters"))
    info("Plot reconstruction waveforms.")
    waveforms_2d_hist(ch_reconst, folder=folder)
#to latent folder
    info("Plot latent space correlations.")
    latent_space_plots(ch_compact, n.config["fc"][1], folder, feature_groups)
    latent_space_plots(ch_compact, n.config["fc"][1], folder, feature_groups, use_cornerplot=true)
#to surf, bulk, slow, high_sigma, alpha folders
    info("Plot examples.")
    plot_reconstruction_examples(ch_data, ch_reconst, n=50, folder=folder)
    return ch_data, ch_noise
end

export get_reconst
function get_reconst(events::EventCollection, n, scale_factor::Int64=256)
    input_size = size(waveforms(events))[1]
    compact = encode(scale_waveforms(events, scale_factor), n)
    reconst = scale_waveforms(decode(compact, n, input_size),1.0/scale_factor)
    pulse_decay_slopes!(reconst, 150, :reconst_end_slope)
    feature_groups = group_events(reconst)
    return compact, reconst, feature_groups
end

export plot_reconstruction_examples
"""
Plots some example pulses together with their reconstruction.

*events*: ideally the test set test set as EventLibrary.
*reconst*: the reconstructions of *events*.
*n*: number of plots to make per group.
*folder*: output directory.
"""
function plot_reconstruction_examples(events::EventCollection, reconst::EventCollection; n::Int64=50, folder::String="plot_reconstruction_examples", exclude=[])
#get events
    slow = ("slow" in exclude)||get_slow(events)
    surf = ("surf" in exclude)||get_ar39(events)
    bulk = ("bulk" in exclude)||get_double_beta(events)
    alpha = ("alpha" in exclude)||get_alpha(events)
    augmented = ("augmented" in exclude)||get_augmented(events)
    high_error_ratio = ("high_error_ratio" in exclude)||get_high_error(events, :error_ratio, mean(events[:error_ratio]) - 6*std(events[:error_ratio]), mean(events[:error_ratio]) + 6*std(events[:error_ratio]))
    high_error_diff = ("high_error_diff" in exclude)||get_high_error(events, :error_diff, mean(events[:error_diff]) - 6*std(events[:error_diff]), mean(events[:error_diff]) + 6*std(events[:error_diff]))
#make plots
    slow_plots = ("slow" in exclude)||smart_subplots(slow[1:min(n, length(slow))], events, reconst, series=true)
    surf_plots = ("surf" in exclude)||smart_subplots(surf[1:min(n, length(surf))], events, reconst, series=true)
    bulk_plots = ("bulk" in exclude)||smart_subplots(bulk[1:min(n, length(bulk))], events, reconst, series=true)
    alpha_plots = ("alpha" in exclude)||smart_subplots(alpha[1:min(n, length(alpha))], events, reconst, series=true)
    augment_plots = ("augmented" in exclude)||smart_subplots(augmented[1:min(n, length(augmented))], events, reconst, series=true)
    high_error_ratio_plots = ("high_error_ratio" in exclude)||smart_subplots(high_error_ratio[1:min(n, length(high_error_ratio))], events, reconst, series=true)
    high_error_diff_plots = ("high_error_diff" in exclude)||smart_subplots(high_error_diff[1:min(n, length(high_error_diff))], events, reconst, series=true)
#save plots to separate directories
    for i in 1:n
    	("slow" in exclude)||(try savefig(slow_plots[i], joinpath(dir(folder, "slow_plots"), "slow_$i.png")); end)
    	("surf" in exclude)||(try savefig(surf_plots[i], joinpath(dir(folder, "surf_plots"), "surface_$i.png")); end)
    	("bulk" in exclude)||(try savefig(bulk_plots[i], joinpath(dir(folder, "bulk_plots"), "bulk_$i.png")); end)
  	("alpha" in exclude)||(try savefig(alpha_plots[i], joinpath(dir(folder, "alpha_plots"), "alpha_$i.png")); end)
	("augmented" in exclude)||(try savefig(augment_plots[i], joinpath(dir(folder, "augment_plots"), "augmented_$i.png")); end)
	("high_error_ratio" in exclude)||(try savefig(high_error_ratio_plots[i], joinpath(dir(folder, "high_error_ratio_plots"), "high_error_ratio_$i.png")); end)
        ("high_error_diff" in exclude)||(try savefig(high_error_diff_plots[i], joinpath(dir(folder, "high_error_diff_plots"), "high_error_diff_$i.png")); end)
    end
end

export save_sample_avg_plots
function save_sample_avg_plots(events::EventCollection, reconst::EventCollection; folder::Union{Void,String}=nothing, stats_dir::Union{String,Bool}=false)
    pyplot_config()
    if folder!=nothing
    	save_dir = dir(folder, "sample_avg")
    else
        save_dir = dir("sample_avg")
    end
    det = get_value(events, :detector_name)
    avg_first_200 = mean(waveforms(events)[1:200,:], 1)
    avg_last_200 = mean(waveforms(events)[201:400,:], 1)
    r_avg_first_200 = mean(waveforms(reconst)[1:200,:], 1)
    r_avg_last_200 = mean(waveforms(reconst)[201:400,:], 1)
#get statistics: mean, median, standard dev
    stats_first = get_statistics(avg_first_200)
    stats_last = get_statistics(avg_last_200)
    stats_r_first = get_statistics(r_avg_first_200)
    stats_r_last = get_statistics(r_avg_last_200)
#make plots
    xlabel = "Pulses"
    ylabel = "Sample average"
    title_first = "Avg. of first 200 samples of $det"
    title_last = "Avg. of last 200 samples of $det"
    savefig(scatter(avg_last_200[:], label="", xlabel=xlabel, ylabel=ylabel, title=title_last, marker=(5, stroke(0))), joinpath(save_dir, "last_200"))
    savefig(scatter(avg_first_200[:], label="", xlabel=xlabel, ylabel=ylabel, title=title_first, marker=(5, stroke(0))), joinpath(save_dir, "first_200"))
    savefig(scatter(r_avg_last_200[:], label="", xlabel=xlabel, ylabel=ylabel, title=title_last, marker=(5, stroke(0))), joinpath(save_dir, "r_last_200"))
    savefig(scatter(r_avg_first_200[:], label="", xlabel=xlabel, ylabel=ylabel, title=title_first, marker=(5, stroke(0))), joinpath(save_dir, "r_first_200"))  
    colors = [:blue :blue :red :red]
    step = 0.001
    x_first = collect(-.25:step:.25)
    x_last = collect(.75:step:1.25)
    hist = stephist([avg_first_200[:], avg_last_200[:], r_avg_first_200[:], r_avg_last_200[:]], layout=(1, 2), bin=[x_first x_last], label=["Data\nσ = $(formatter(stats_first["std"]))" "Data\nσ = $(formatter(stats_last["std"]))" "Reconstructions\nσ = $(formatter(stats_r_first["std"]))" "Reconstructions\nσ = $(formatter(stats_r_last["std"]))"], color=colors, size=(1400, 600), legend=:topleft, xlabel=ylabel, ylabel=["Counts [$(formatter(step))/bin]" ""], title=[title_first title_last])
    savefig(hist, joinpath(save_dir, "histogram"))
#write file
#if no filepath defined
 #   if stats_dir==nothing
#	stats_dir = joinpath(save_dir, "sample_avg_stats")
 #   end
#if filepath does not exist\
    if stats_dir != false
  	  if !ispath(stats_dir)
		log_keys = ["det" "first mean" "first median" "first sigma" "last mean" "last median" "last sigma" "rec first mean" "rec first median" "rec first sigma" "rec last mean" "rec last median" "rec last sigma"]
  	  	open(joinpath(stats_dir), "w") do file
  	  		writedlm(file, (log_keys), " & ")
  	  	end
  	  end
  	  keys = ["mean" "median" "std"]
  	  open(stats_dir, "a") do file
  	      writedlm(file, ([det [formatter(stats_first[k]) for k in keys]... [formatter(stats_last[k]) for k in keys]... [formatter(stats_r_first[k]) for k in keys]... [formatter(stats_r_last[k]) for k in keys]...]), " & ")
  	  end
    end
end

export end_slope_distribution
function end_slope_distribution(events::EventCollection, reconst::EventCollection; title="", folder::Union{String,Void}=nothing, stats_dir::Union{String,Bool}=false)
    pyplot_config()
    if folder!=nothing
    	save_dir = dir(folder, "end_slopes")
    else
        save_dir = dir("end_slopes")
    end
#get stats
    det = get_value(events, :detector_name)
    stats = get_statistics(events[:data_end_slope])
    rec_stats = get_statistics(reconst[:reconst_end_slope])
#plot
    x = linspace(-1e-3, 1e-3, 100)
    fig = stephist(events[:data_end_slope], label="Pulses\nσ = $(formatter(stats["std"]))", bins=x, color=:blue, formatter=:scientific)
    stephist!(fig, reconst[:reconst_end_slope], label="Reconstructions\nσ = $(formatter(rec_stats["std"]))", bins=x, color=:red)
    xaxis!("ω")
    yaxis!("Counts")
    ylims!((0, 1.4*maximum(fig.series_list[2].d[:y])))
    folder!=nothing&&savefig(joinpath(save_dir, "Endslopes_$(get_value(events, :detector_name))"))
#if file does not exist
    if stats_dir != false
   	 if !ispath(stats_dir)
		log_keys = ["det" "endslope mean" "endslope median" "endslope sigma" "rec endslope mean" "rec endslope median" "rec endslope sigma"]
  	  	open(joinpath(stats_dir), "w") do file
  	  		writedlm(file, (log_keys), " & ")
 	   	end
 	   end
 	   keys = ["mean" "median" "std"]
 	   open(stats_dir, "a") do file
 	       writedlm(file, ([det [formatter(stats[k]) for k in keys]... [formatter(rec_stats[k]) for k in keys]...]), " & ")
 	   end
    end
    return fig
end

export gaussian
gaussian(x, p) = p[1] .* exp.(-0.5 .* ((x.-p[2]).^2) ./ (p[3].^2))

export plot_sigmas
"""
Creates reconstruction error statistics and plots.
"""
function plot_sigmas(n, ch_data, ch_noise; folder="plots", stats_dir::Union{String,Bool}=false, scale_factor=256)
    pyplot_config()
    energy_fig = plot()
    joint_dist = plot()
    det = get_value(ch_data, :detector_name)
    save_dir = dir(folder, "error_plots")
    histogram_label = ["Training set", "Test set", "Mixed dataset of $det"]
    grid = Dict("stats"=>["mean" "median" "sigma"],
            "distribution"=>["data" "fit"],
            "dataset"=>["train" "test" "all"], "method"=>["diff" "ratio"])
    log_keys = [ ds*"_"*di*"_"*st for st in grid["stats"], di in grid["distribution"], ds in grid["dataset"]]
    reconst_error_dict = Dict( "diff"=>Dict(lk=>[] for lk in log_keys), "ratio"=>Dict(lk=>[] for lk in log_keys))
    colors=[:blue :red]
    for (j, method) in enumerate(grid["method"])
	joint_dist = plot(label="events")
	if method == "ratio"
		xlabel = L"$\mathrm{\sigma_{reconst}/\sigma_{noise}}$"
		x = linspace(0, 3, 101)
	else
		xlabel = L"$\mathrm{\sigma_{reconst}-\sigma_{noise}}$"
		x = linspace(-0.04, 0.04, 101)
	end		
	for (i, dset) in enumerate(grid["dataset"])
		info(dset)
		if dset != "all"
        		data_subset = ch_data[:set=>dset]
	        	data_noise_subset = ch_noise[:set=>dset]
		else
			data_subset = ch_data
			data_noise_subset = ch_noise
		end
       		compact = encode(scale_waveforms(data_subset, scale_factor), n)
     	 	reconst = scale_waveforms(decode(compact, n, size(waveforms(data_subset))[1]),1.0/scale_factor)
#make joint histograms and return data fit statistics
     		energy_fig, sigmas, stats = make_2d_hist(
                                data_subset, data_noise_subset, reconst, method=method,
                                title="",
                                filename= joinpath(save_dir, "$(dset)_sigma"), xmin=50, xmax=2500, xsteps=246)
#log the data distribution statistics0
		push!(reconst_error_dict[method]["$(dset)_data_mean"], stats["mean"])
		push!(reconst_error_dict[method]["$(dset)_data_median"], stats["median"])
		push!(reconst_error_dict[method]["$(dset)_data_sigma"], stats["std"])
#fit gaussian to distribution
		dist = fit(Histogram{Float64}, sigmas, x, closed=:left)
        	dist.weights ./= sum(dist.weights)
		length(dist.weights)!=length(x)&&push!(dist.weights, 0)
		init_fit_std = if method=="ratio"; .2 else .002 end
    		init_fit_mean = if method=="ratio"; 1.0 else 0.0 end
		fit_ = curve_fit(gaussian, x, dist.weights, [.01, init_fit_mean, init_fit_std])
    		push!(reconst_error_dict[method]["$(dset)_fit_mean"], fit_.param[2])
    		push!(reconst_error_dict[method]["$(dset)_fit_median"], fit_.param[2])
		push!(reconst_error_dict[method]["$(dset)_fit_sigma"], fit_.param[3])
#1d histogram with train and test set errors jointly
		if dset != "all"
			plotted_mean = formatter(stats["mean"])
			plotted_sigma = formatter(stats["std"])
			if (length(string(plotted_mean)) == 3)
				plotted_mean = string(plotted_mean)*"0"
 			end
			if (length(string(plotted_sigma)) == 3)
				plotted_sigma = string(plotted_sigma)*"0"
 			end
        		plot!(joint_dist, x, dist.weights , color=colors[i], seriestype=:steppre, label="$(histogram_label[i])\nμ = $(plotted_mean) σ = $(plotted_sigma)", fillalpha=0.7)
        		xaxis!(xlabel)
        		yaxis!("Event ratio")
			ylims!(-Inf, 0.13)
        		#title!("Normalized error distribution of $det")
		else
#in case of "all", no plot, only push sigmas to dict
			error_key = Symbol("error_$(method)")
			haskey(ch_data.entries[1], error_key)||put_label!(ch_data, error_key, sigmas)
			haskey(ch_noise.entries[1], error_key)||put_label!(ch_noise, error_key, sigmas)
		end
	end    
        savefig(joint_dist, joinpath(save_dir, "error_$(method)_joint_distribution")) 
    end

#write file
#if no filepath defined
#    if stats_dir==nothing#
#	stats_dir = joinpath(save_dir, "error_stats")
#    end
    if stats_dir != false
#if file does not exist
	    if !ispath(stats_dir)
	    	open(stats_dir, "w") do file
			writedlm(file, (["det" ["ratio_"*lk for lk in log_keys]... ["diff_"*lk for lk in log_keys]...]), " & ")
	    	end
	    end
	    open(stats_dir, "a") do file
	       writedlm(file, ([det [formatter(reconst_error_dict["ratio"][lk][1]) for lk in log_keys]... [formatter(reconst_error_dict["diff"][lk][1]) for lk in log_keys]...]), " & ")
 	   end
    end
    return reconst_error_dict, ch_data, ch_noise, energy_fig, joint_dist
end

export plot_time_dependency
function plot_time_dependency(time_consistency_dict::Dict; folder::Union{String,Void}=nothing, data=true)
    pyplot_config()
    fig = scatter(ylabel=L"<$\mathrm{\sigma_{reconst}/\sigma_{noise}}$>")
    #data
    if data
    	scatter!(time_consistency_dict["run"], parse.(time_consistency_dict["mean"]), yerror=parse.(time_consistency_dict["sigma"]), label="", color=:blue)
    	scatter!(time_consistency_dict["run"], parse.(time_consistency_dict["mean"]), label="Data", color=:blue, marker=(5,stroke(0)))
    end
    #fit
    c = if data "red" else "blue" end
    scatter!(time_consistency_dict["run"], parse.(time_consistency_dict["fit_mean"]), yerror=parse.(time_consistency_dict["fit_sigma"]),
            label="", color=c)
    scatter!(time_consistency_dict["run"], parse.(time_consistency_dict["fit_mean"]),
            label="Gaussian fit", color=c, marker=(5,stroke(0)))
    xaxis!("Run")
    temp  = fig.series_list[2].d[:y]
    ymax = maximum(temp[.!isnan.(temp)])
    ylims!(0, 3.5)
    folder!=nothing&&savefig(joinpath(folder,"time_consistency"))
    return fig
end

export exclude_training_samples
"""
*training_data*: EventLibrary of the training set of one detector. Events from this were used for the DNN optimization. Mixed data with 50% cal and 50% phy. E is below 2500 keV.
*all_phy_data*: All physics data of a detector w.o. upper E constraint.
return: *phy_wo_train*: subset of *all_phy_data* w.o. the training samples of *training_data*, the equalized number of surface and bulk events.
"""
function exclude_training_samples(training_data::EventLibrary, all_phy_data::EventLibrary)
    #these need to be thrown away from the final spectrum
    training_samples, _ = equalize_counts_by_label(training_data, label_surface_bulk!, :SE)
    info(typeof(training_samples))
    bl = training_samples[:baseline_level] # baseline level is unique (more than E)
    info("Training data: $(length(bl))")
    useful_sample_indices = find(x->!(x in bl), all_phy_data[:baseline_level])
    @assert length(all_phy_data[:E]) - length(useful_sample_indices) == length(bl) "Something's fishy."
    #filter indices
    return all_phy_data[useful_sample_indices]
end

export reconst_error_summary
"""
Makes a plot for the reconstruction errors for all detectors.
*result_dict*: dictionary of saved parameters during training
*keys*: keys of the dict to use for plotting
*errors*: keys to use for error bars
*plot_title*
*folder*: plot saving folder
return: *fig*: the plot
"""
function reconst_error_summary(result_dict::Dict,
        keys=["ratio_train_data_mean", "ratio_test_data_mean"], 
        errors=["ratio_train_data_sigma", "ratio_test_data_sigma"],
        titles=["Training_set", "Test_set"];
        plot_title="Data",
        folder::Union{Void, String}=nothing)
    pyplot_config()
    fig = scatter(ylabel=L"<$\mathrm{\sigma_{reconst}/\sigma_{noise}}$>", legend=:topleft)
    colors = [:blue, :red, :green, :orange]
    strings=[5, 8, 16, 24, 27]
    for (key, error, t, c) in zip(keys, errors, titles, colors)
        scatter!(convert(Array{String, 1}, result_dict["det"]), result_dict[key], yerror=result_dict[error],
                    label="", color=c, size=(1200,500))
        scatter!(convert(Array{String, 1}, result_dict["det"]), result_dict[key],
                    label=t, color=c, marker=(5,stroke(0)))
    end
    xaxis!("", xrotation=90, xticks=length(result_dict["det"]))
    temp  = fig.series_list[2].d[:y]
    ymax = maximum(temp[.!isnan.(temp)])
    ylims!(-Inf, 1.15ymax)
    vline!(strings, label="", color="grey")
    folder!=nothing&&savefig(joinpath(folder, "$(plot_title)_reconstruction_errors"))
    return fig
end

export wilson
"""
Calculates the binomial proportion confidence interval in case of a ratio histogram.
*total*: array of histogram bin counts. Total count, before cut.
*passed*: array of histogram bin counts. Survived events, after cut.
*level*: confidence level in [0,1]
*return*: lower and upper confidence interval bounds as arrays. (not the range)
"""
function wilson(total::AbstractArray, passed::AbstractArray, level::Float64)
    α = 1 - level
    κ = quantile.(Normal(), [1 - .5*α])[1]
    mode = (passed .+ .5*κ^2)./(total .+ κ^2)
    Δ = κ./(total .+ κ^2).*sqrt.(passed.*(1 .- passed./total) .+ (.5*κ)^2)
    return max.(0,mode .- Δ), min.(1,mode .+ Δ)
end

export typical_shapes
function typical_shapes(events)
    surf = events[get_ar39(events)]
    bulk = events[get_double_beta(events)]
    n_events = eventcount(bulk)
    info(n_events)
    ymin = minimum(waveforms(surf[1:n_events]))
    ymax = maximum(waveforms(surf[1:n_events]))
    fig = plot(placeholder(title=""),
         plot(waveforms_2d_hist(bulk), title=""),
         plot(waveforms_2d_hist(surf, n_events), ylabel="", title=""),
         layout=@layout([x{.0001h}; a b]), size=(1200,500))
    ylims!(ymin, ymax)
    return fig
end

export analyze_runs
function analyze_runs(env, ch_data, ch_noise, n; save_plots::Bool=false)
    error_keys = ["run" "events" "mean" "sigma" "fit_mean" "fit_sigma" "plot"]
    #loop over runs and test
    x = linspace(-1.5, 3.5, 100)
    runs = sort(unique(ch_data[:run]))
    time_consistency_dict = Dict(lk=>[] for lk in error_keys)
     info("Start runs")
	det = get_value(ch_data, :detector_name)
    for run in runs
#make run dir
        run_pulses, run_noise = get_run(env, ch_data, ch_noise, run=run)
        run_compact, run_reconst, _ = get_reconst(run_pulses, n)
#make plots and calculate errors
        std_p, std_n = calculate_deviation(run_pulses, run_reconst, run_noise)
        sigmas = std_p./std_n
        μ = mean(sigmas)
        σ = std(sigmas)
        events = eventcount(run_pulses)
#get distribution of errors
        dist = fit(Histogram{Float64}, sigmas, x, closed=:right)
        dist.weights ./= sum(dist.weights)
        fit_plot = plot(x, dist.weights, seriestype=:steppre, label="", color=:blue)
#make x and y equal length
        length(fit_plot.series_list[1].d[:y])!=length(x)&&push!(fit_plot.series_list[1].d[:y], 0)
#fit Gaussian
        fit_ = curve_fit(gaussian, x, fit_plot.series_list[1].d[:y], [1., 1., .2])
        fit_μ = fit_.param[2]
        fit_σ = fit_.param[3]
        plot!(fit_plot, x, gaussian(x, fit_.param),
            label="Gaussian fit\nμ = $(formatter(fit_μ)) σ = $(formatter(fit_σ))", 
        color=:red)
        ylims!((0, 0.25))
        xaxis!(L"$\mathrm{\sigma_{reconst}/\sigma_{noise}}$")
        yaxis!("Event ratio")
        title!("Run $run of $det")
        save_plots&&savefig(joinpath(dir(det_dir, "run_plots"), "run $run"))
#write stats
        params = [run events formatter(μ) formatter(σ) formatter(fit_μ) formatter(fit_σ) fit_plot] 
        [push!(time_consistency_dict[lk], param) for (lk, param) in zip(error_keys, params)]
    end
    time_dep = plot_time_dependency(time_consistency_dict)
    return time_dep
end

export correlation_coefficient
function correlation_coefficient(events::EventCollection)
    ρ = cor(events[:data_end_slope], events[:reconst_end_slope])
    Δρ = sqrt((1-ρ^2)/(eventcount(events)-2))
    fig = scatter(events[:data_end_slope], events[:reconst_end_slope],
        marker=(3, stroke(0)), color=:blue, label="")
    xaxis!(L"\mathrm{ω_{input}}\ [10^{-3}]", xticks=(1e-3.*[-2, 0, 2, 4], [-2, 0, 2, 4]))
    yaxis!(L"\mathrm{ω_{reconst}}\ [10^{-3}]", yticks=(1e-3.*[-1, 0, 1, 2], [-1, 0, 1, 2]))
    title!(fig, get_value(events, :detector_name))
    return fig, ρ, Δρ
end

export calculate_alpha_class
function calculate_alpha_class(events::EventCollection)
    alpha_classifier = Dict("ANG1"=>176, "ANG2"=>184, "ANG3"=>200, "ANG4"=>196, "ANG5"=>216, "RG1"=>200, "RG2"=>216)
    for det in keys(alpha_classifier)
        ch_data = events[:detector_name=>det]
        labels = zeros(Int8, eventcount(ch_data))
        alpha_indices = find(x->x<alpha_classifier[det], ch_data[:risetime])
        labels[alpha_indices] .= 1
        #savefig(smart_subplots(alpha_indices[1:4], events[:detector_name=>det], labels=["" ""]), joinpath(out_dir, "$det"))
        put_label!(ch_data, :isAlpha, labels)
    end
end

export get_roi_alphas
function get_roi_alphas(events::EventCollection; dettype=:coax)
    if dettype == :coax
        return events[:isAlpha=>1]
    elseif dettype == :bege
        return filter(filter(events, :AoE, x->x>0), :AoE_class, x->x==1)
    end
end

export line_fit
line_fit(x, p) = p[1] .+ p[2].*x

export stderror
function StatsBase.stderror(fit; rtol::Real=NaN, atol::Real=0)
    # computes standard error of estimates from
    #   fit   : a LsqFitResult from a curve_fit()
    #   atol  : absolute tolerance for approximate comparisson to 0.0 in negativity check
    #   rtol  : relative tolerance for approximate comparisson to 0.0 in negativity check
    covar = estimate_covar(fit)
    # then the standard errors are given by the sqrt of the diagonal
    vars = diag(covar)
    vratio = minimum(vars)/maximum(vars)
    if !isapprox(vratio, 0.0, atol=atol, rtol=isnan(rtol) ? Base.rtoldefault(vratio, 0.0, 0) : rtol) && vratio < 0.0
        error("Covariance matrix is negative for atol=$atol and rtol=$rtol")
    end
    return sqrt.(abs.(vars))
end
