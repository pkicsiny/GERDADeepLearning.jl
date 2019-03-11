#save and load pped files
#usage only for small data
using JLD
using Plots, StatsBase, LaTeXStrings
using MXNet
using Primes



export save_working_data
function save_working_data(env::DLEnv, data, name="my_working_data", to_dir="saved")
    dir = resolvepath(env, "data", to_dir)
    isdir(dir) || mkdir(dir)
    save("data/$(to_dir)/$(name).jld", "name", data)
end

export load_working_data
function load_working_data(name="my_working_data", from_dir="data/saved")   
    return load("$(from_dir)/$(name).jld")["name"]
end

export load_jld_data
"""
    load_jld_data(detector::String, datatype::String, pulse_type::String, ds_labels::String)

Loads a saved pulses and noise .jld file.
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
	 shift_midpoints(events::EventLibrary;
 			 center_y=0.5,
 			 target_length=256,
 			 position=0.25)

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
  before = Int64(round(target_length*position)) #64
  after = Int64(round(target_length*(1 - position))) #192
  rwf = zeros(Float32, target_length, eventcount(events)) #256, #events

  @everythread for i in threadpartition(1:eventcount(events))
    index = findmin(abs.(charges.waveforms[:,i] - center_y))[2] #get y=0.5 index
    if (index < before) || (index > s - after) #if not enough space to cut, too left or right
      events[:FailedPreprocessing][i] = 1
    else
      rwf[:,i] = events.waveforms[(index-before+1) : (index+after) , i] #actual data cut of target_length size
    end
  end
  events.waveforms = rwf
  return events
end

export lazy_norm
"""
	lazy_norm(events::EventLibrary)

Normalizes events by dividing them by their energy*2 (E=0.5*Q*U)
"""
function lazy_norm(events::EventLibrary)
  charges = charge_pulses(events; create_new=true)
  energies = events[:E]
  @everythread for i in threadpartition(1:eventcount(events))
    events.waveforms[:,i] *= 1 / (2*energies[i])
  end
  return events
end

export flag_abnormal_pulses
"""
	flag_abnormal_pulses(events::EventLibrary; center_y::Float32=0.5)

Labels pulses which have a peak before main rise.
*center_y* is a float representing the center point of the pulses by which they are to be examined.
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
    @everythread for i in threadpartition(1:eventcount(charges))
        midpoint = findmin(abs.(charges.waveforms[:,i] - center_y))[2]
	midwidth = length(filter(x -> (x<0.6)&&(x>0.4), waveforms(charges)[:,i]))
        abnormal_shape[i] = (sum((events.waveforms[midpoint:end, i].+delta).<events.waveforms[midpoint, i]) > extra_peak_threshold)||(midwidth>width_threshold)
    end
    put_label!(events, :abnormal_shape, abnormal_shape)
    return events
end

export push_effective_e_label!
"""
	push_effective_e_label!(data, noise_scale::Int64)

Appends the effecive energy (:E_eff) label to the data by dividing the energy by the noise scale of the data.
"""
function push_effective_e_label!(data, noise_scale::Int64)
    # rounds effecitve energies to 3 decimals
    put_label!(data, :E_eff, convert(Array{Float32,1}, round.(data[:E] / (noise_scale + 1), 3)))
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
# append missing labels to physics dataset (effective energy and abnormal pulse flag) with default values
    for (idx, key) in enumerate(keys(cal_data))
	if key == :E_eff
		!(key in keys(phy_data))&&put_label!(phy_data, key, phy_data[:E])
    		!(key in keys(phy_data_noise))&&put_label!(phy_data_noise, key, phy_data_noise[:E])
	else
    		!(key in keys(phy_data))&&put_label!(phy_data, key, zeros(Int8, eventcount(phy_data)))
    		!(key in keys(phy_data_noise))&&put_label!(phy_data_noise, key, zeros(Int8, eventcount(phy_data_noise)))
	end		  
    end
    total = Int32(env.config["data_ratio"]["total"][1])
    phy_events = Int32(env.config["data_ratio"]["phy"][1]*total)
    cal_events = Int32(env.config["data_ratio"]["cal"][1]*total)

    @assert(phy_events == total - cal_events, "The ratio of physics and calibration data must add up to 1. Check 						       the json file.")
    @assert(cut_method in ["cut", "fill"], "Cut method must be either 'cut' or 'fill'.")
    @assert(select_method in ["random", "first"], "Select method must be either 'random' or 'fill'.")  

    
    if eventcount(phy_data) < phy_events
        if cut_method == "fill"
            phy_events = eventcount(phy_data)
            cal_events = Int32(total - phy_events)
            info("There are only $(eventcount(phy_data)) physics events. Using all $(eventcount(phy_data)) physics events and filling up the rest from calibration data.")
        elseif cut_method == "cut"
            phy_events = eventcount(phy_data)
            cal_events = Int32(floor(phy_events/env.config["data_ratio"]["phy"][1]*env.config["data_ratio"]["cal"][1]))
            total = Int32(phy_events + cal_events)
            info("There are only $(eventcount(phy_data)) physics events. Reducing total event number to keep ratios of phy: $(phy_events/total), cal: $(cal_events/total).")
        end
    end
    
    if select_method =="random"
        cal_idx = sample(1:eventcount(cal_data),cal_events, replace = false)
        phy_idx = sample(1:eventcount(phy_data),phy_events, replace = false)
    elseif select_method =="first"
        cal_idx = collect(1:cal_events)
        phy_idx = collect(1:phy_events)
    end
    
    result_pulses = DLData(EventLibrary[])
    result_noise = DLData(EventLibrary[])
    #loop over event libraries (detectors)
    for detector in length(cal_data.entries)
        #filter events by indices
        filtered_cal_pulses = cal_data.entries[detector][cal_idx]
        put_label!(filtered_cal_pulses, :isCal, convert(Array{Int8,1}, ones(cal_events)))
        filtered_phy_pulses = phy_data.entries[detector][phy_idx]
        put_label!(filtered_phy_pulses, :isCal, convert(Array{Int8,1}, zeros(phy_events)))
        
        filtered_cal_noise = cal_data_noise.entries[detector][cal_idx]
        put_label!(filtered_cal_noise, :isCal, convert(Array{Int8,1}, ones(cal_events)))
        filtered_phy_noise = phy_data_noise.entries[detector][phy_idx]
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
        #append event library to result dldata
        push!(result_pulses.entries, cal_and_phy_pulses)
        push!(result_noise.entries, cal_and_phy_noise)
    end
    return result_pulses, result_noise
end

export make_2d_hist
"""
	make_2d_hist(env, pulses, noise, reconst;
        	group_indices::AbstractArray=nothing,
        	group_limits::AbstractArray=nothing,
        	title::AbstractString=nothing,
        	xmax=nothing, 
        	xmin=nothing,
        	xsteps=nothing,
        	ymax=nothing,
        	ymin=nothing,
        	ysteps=nothing,
        	filename=nothing,
        	add_dist_plot=false)

Method for calculating standard deviations of the reconstruction errors and the noise.
Plots the difference between the std error and std noise on a histogram.
"""
function make_2d_hist(env, pulses, noise, reconst;
        group_indices::AbstractArray=nothing,
        group_limits::AbstractArray=nothing,
        title::AbstractString=nothing,
        xmax=nothing, 
        xmin=nothing,
        xsteps=nothing,
        ymax=nothing,
        ymin=nothing,
        ysteps=nothing,
        filename=nothing,
        add_dist_plot=false)
    
    std_reconst, std_noise = calculate_deviation(pulses, reconst, noise)
    
    avg_std = std(std_reconst-std_noise)
    info(env, 3, "Std of additional reconstruction deviation: $avg_std")
    
    (ymax == nothing) && (ymax = 10avg_std)
    (ymin == nothing) && (ymin = -10avg_std)
    (ysteps == nothing) && (ysteps = 400)
    err_axis=linspace(ymin, ymax, ysteps) #avg std no known in advance
    
    (xmax == nothing) && (xmax = 1.02maximum(pulses[:E]))
    (xmin == nothing) && (xmin = 0)
    (xsteps == nothing) && (xsteps = 400)
    E_axis = linspace(xmin, xmax, 400)   

    
    sigmas = std_reconst./std_noise # changed from -
    mse_hist = fit(Histogram{Float64}, (pulses[:E], sigmas), (E_axis, err_axis), closed=:left) #data, edges (bins)
    broadcast!(x -> x <= 0 ? NaN : log10(x), mse_hist.weights, mse_hist.weights)
    
    figure = plot(mse_hist, colorbar=:none, xticks = xmin:200:xmax)
    
    xaxis!("Energy (keV)")
    yaxis!(L"$\sigma_{reconst} - \sigma_{noise}$")
    
    (minimum(pulses[:E]) < 50) && vline!([50],label="50 keV")
    (title != nothing) && title!(title) 
    (filename != nothing) && savefig(filename)
    
    #putting them together
    if add_dist_plot    
        sigma_dist_plot = plot_sigma_distributions(env, 
pulses, std_noise, std_reconst, sigmas, err_axis, group_limits, group_indices)
        joint_fig = plot(figure, sigma_dist_plot, size=(1600, 800), layout=@layout([a{0.7w} b]))
        (filename != nothing) && savefig(filename*"_2.png")
        return figure, joint_fig, sigmas
    else
        return figure, sigmas
    end
end

export plot_sigma_distributions
"""
	plot_sigma_distributions(env, pulses, std_noise, std_reconst,
			         sigmas, err_axis, group_limits, group_indices)

Plots the error distribution of the reconstruction errors by energy groups.
"""
function plot_sigma_distributions(env, pulses, std_noise, std_reconst, sigmas, err_axis, group_limits, group_indices)
    info(env,3,"Adding error distribution curves.")
    #second plot
    mse_hist_1d = fit(Histogram{Float64}, sigmas, err_axis, closed=:left)
    #cumulated events
    mse_hist_1d_sub = [fit(Histogram{Float64}, std_reconst[collect(Iterators.flatten(group_indices[1:i]))]-std_noise[collect(Iterators.flatten(group_indices[1:i]))], err_axis, closed=:left) for i in range(1,length(group_indices))]
    sigma_dist_plot = plot(mse_hist_1d.weights, err_axis[1:end-1], line=(:black, :steppre),
        label="All events", fill=(0, :black), legend=:topright, title="Distribution of error")
    for (idx,sub) in enumerate(reverse(mse_hist_1d_sub)) #plotting subsets, reverse bc of overlapping plots

        try plot!(sub.weights, err_axis[1:end-1], line=("red1", :steppre),
            label="$(group_limits[length(group_limits)+1-idx])",
                fill=(0,"gray$(idx*15)")) catch; plot!(sub.weights, err_axis[1:end-1], line=("red1", :steppre),
            label="$(group_limits[length(group_limits)+1-idx])",
                fill=(0,"black")) end
    end
    xaxis!("# of events")
    yaxis!(yticks=nothing)
    
    return sigma_dist_plot
end

export group_events
"""
	group_events(reconst)

Groups the indices of the events according to several criteria based on the event labels.
Returns a dictionary where each keys are grouping criteria and values are a list of two elements.
The first one is an array containing the indices of pulses belonging to each group. The second is the array of group labels.
"""
function group_events(reconst)
    feature_groups = Dict()
    #event energy
    if haskey(reconst,:E)
        energy_groups = [find(E->(E<100)&&(E>=50), reconst[:E]),
                         find(E->(E<200)&&(E>=100), reconst[:E]),
                         find(E->(E<300)&&(E>=200), reconst[:E]),
                         find(E->(E<400)&&(E>=300), reconst[:E]),
		      	 find(E->(E>=400), reconst[:E])]
        energy_group_labels = ["50keV<E<100keV", "100keV<E<200keV", "200keV<E<300keV","300keV<E<400keV", "E>400keV"]
        feature_groups["energy"] = [energy_groups, energy_group_labels]
    end
  
    #decay slope
    if haskey(reconst,:decay_slope)
        slope_groups = [find(x -> x >= 0,reconst[:decay_slope]),
                        find(x -> x < 0,reconst[:decay_slope])]
        slope_group_labels = ["(+)", "(-)"].*" slope"
        feature_groups["slope"] = [slope_groups, slope_group_labels]
    end

        #cal phy
    if haskey(reconst,:isCal)
        datatype_groups = [find(x -> x == 1,reconst[:isCal]),
                        find(x -> x == 0,reconst[:isCal])]
        datatype_group_labels = ["cal.", "phy."].*" pulse"
        feature_groups["datatype"] = [datatype_groups, datatype_group_labels]
    end
        
        #risetime
    if haskey(reconst,:risetime)
        σ = std(reconst[:risetime])
        risetime_groups = [find(x ->(x==0), reconst[:risetime]),
                           find(x ->(x<=300)&&(x>0), reconst[:risetime]),
                           find(x ->(x<=600)&&(x>300), reconst[:risetime]),
                           find(x ->(x>600), reconst[:risetime])]
        risetime_group_labels = " rise time ".*["zero", "below 300", "below 600", "above 600"].*" (ns)"
        feature_groups["risetime"] = [risetime_groups, risetime_group_labels]
    end
    
        #aoe
    if haskey(reconst,:AoE)
        aoe_groups = [find(x -> x>=0,reconst[:AoE]),
                        find(x -> x<0,reconst[:AoE])]
        aoe_group_labels = ["(+)", "(-)"].*" AoE param."
        feature_groups["aoe"] = [aoe_groups, aoe_group_labels]
    end
    return feature_groups
end

export add_noise_to_pulses
"""
    add_noise_to_pulses(data::DLData;
                        baseline_traces::Union{DLData,Void}=nothing,
                        noise_scale::Int32=5,
                        entry::Union{Int32,Void}=22,
                        n::Int32=100000)

*data* is a DLData object of pulses and the function mixes baseline noise to the waveforms of it.
*baseline_traces* is the DLData object of baseline noise that will be loaded from a file if it is not given as input.
*noise_scale* is the scale factor of the baseline noise. 
In general: E_eff=E/(noise_scale+1) (the +1 is bc. of the original baseline noise naturally on the signals)
*entry* specifies an EventLibrary entry to be used in the DLData. 
E.g. mix noise of specific detector to signals of that detector.
If set to nothing, all entries will be used.
*n* specifies the number of data pulses to return with added noise.
Returns a DLData object with 1 entry containing the noise contaminated pulses.
"""
function add_noise_to_pulses(data::DLData;
                             baseline_traces::Union{DLData,Void}=nothing,
                             noise_scale::Int64=5,
                             entry::Union{Int64,Void}=22,
                             n::Int64=100000)
    # load noise data if not specified in input
    if baseline_traces == nothing
        baseline_traces = load_working_data("phy_baselines")
    end
    if entry != nothing
        baseline_traces = baseline_traces.entries[entry].waveforms
        pure_cal_data = data.entries[entry][1:n]
    else
        baseline_traces = waveforms(baseline_traces)
        pure_cal_data = data[1:n]
    end
    # mean of each noise trace
    noise_means = mean(baseline_traces, 1)
    # substract mean
    baseline_traces .-= noise_means
    # filter abnormal noise traces (full 0 or has a slope)
    baseline_traces = baseline_traces[:,find(x->x!=0, mean(baseline_traces,1))]
    baseline_traces = baseline_traces[:,find(x->x<20, std(baseline_traces,1))]

    # sample random noise traces
    noise_idx = sample(1:size(baseline_traces)[2], eventcount(pure_cal_data), replace = false)
    # add baseline noise to data pulses with a scale factor
    pure_cal_data.waveforms .+= noise_scale*baseline_traces[:,noise_idx]
    # create DLData object for contaminated data
    contaminated_cal_data = DLData(EventLibrary[])
    push!(contaminated_cal_data.entries, pure_cal_data)
    # append a flag for the effective energy
    push_effective_e_label!(contaminated_cal_data, noise_scale)
    return contaminated_cal_data
end

export correct_preprocessing_flags!
"""
    correct_preprocessing_flags!(pulses, noise)

Sets flag entries in noise set to 1 where the corresponding pulse entry has 1.
"""
function correct_preprocessing_flags!(pulses, noise)
    for (idx2, key) in enumerate(keys(noise))
        if (noise[key] != pulses[key])&&(key !=:preprocessing)
            info("Correct *$(key)* key.")
            noise[key] .= pulses[key]
        end
    end
end

export latent_space_plots
"""
    latent_space_plots(compact,
                       latent_size::Int64,
                       filepath::String,
                       feature_groups::Union{Dict{Any,Any},Void}=nothing)

Method for plotting all permutation pairs of the latent vector components.
*compact* are the latent representations.
*latent_size* is the length of a latent vector.
*filepath* is the path to save the plots.
*feature_groups* is either 'nothing' or a dict of indices and group names.
"""
function latent_space_plots(compact,
                            latent_size::Int64,
                            filepath::String,
                            feature_groups::Union{Dict{Any,Any},Void}=nothing)
    #get all combinations of length 2
    pairs = filter(x -> length(x) == 2, [65-findin(bits(i),'1') for i=1:(2^latent_size-1)])
    #make new folder for latent correlation plots
    corr_folder = filepath*"/latent_space_correlations"
    isdir(corr_folder) || mkdir(corr_folder)
    n_events = eventcount(compact)
    #loop over pairs
    for pair in pairs
        #loop over predefined groups
        if feature_groups == nothing
            fig = latent_space_plot(compact, n_events, pair[1], pair[2])
                xlabel!("Latent variable no. $(pair[1])")
                ylabel!("Latent variable no. $(pair[2])")
                savefig("$(corr_folder)/$(pair[1])_$(pair[2])")
        else
            for (key, value) in feature_groups
                fig = latent_space_plot(compact, n_events, pair[1], pair[2], value[1], value[2])
                xlabel!("Latent variable no. $(pair[1])")
                ylabel!("Latent variable no. $(pair[2])")
                savefig("$(corr_folder)/$(key)_$(pair[1])_$(pair[2])")
            end
        end
    end
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
function latent_space_plot(latent_space::EventLibrary,
                           n_events::Int,
                           latent_x::Int,
                           latent_y::Int,
                           groups::Union{AbstractArray,Void}=nothing,
                           group_labels::Union{AbstractArray,Void}=nothing)
    
    fig = scatter()
    if groups != nothing && group_labels !=nothing
    	for grp in range(1,length(groups))    
        	group = try latent_space[groups[grp]]; catch group = groups[grp] end
        	scatter!(fig, group.waveforms[latent_x,1:min(eventcount(group), n_events)],
        	      group.waveforms[latent_y,1:min(eventcount(group), n_events)],
        	      seriestype=:scatter, label="$(group_labels[grp])")
    	end
    else
	scatter!(fig, latent_space.waveforms[latent_x,1:min(eventcount(latent_space), n_events)],
        	      latent_space.waveforms[latent_y,1:min(eventcount(latent_space), n_events)],
        	      seriestype=:scatter, label="")
    end
    return fig
end

export reconst_decay_slopes!
"""
    reconst_decay_slopes!(reconst,
                          length_pf_slope=100)

Calculates the decay slope of the pulses with least squares and appends a
new label whether the slope is positive or negative.
Source: https://dsp.stackexchange.com/questions/42364/find-smoothed-first-derivative-from-signal-with-noisy-slope
*reconst* is containing the autoencoder reconstructions.
*length_of_slope* determines the last n samples of each reconstruction
waveforms to be considered when calculating the end slope.
Returns the EventLibrary extended with a new label.
"""
function reconst_decay_slopes!(reconst::EventLibrary,
                               length_of_slope::Int64=100)
    
    slopes = zeros(Float32, eventcount(reconst))
    x = collect(1:1:length_of_slope)  # x
    for event in range(1,eventcount(reconst))
        noisy_end = reconst.waveforms[:,event][end-length_of_slope+1:end]  # y
        slopes[event] = convert(Float32,sum((x - mean(x)).*(noisy_end - mean(noisy_end)))/sum((x - mean(x)).*(x - mean(x))))
    end
    put_label!(reconst, :decay_slope, slopes)
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
    for tuple in layer_tuples[1][2:end] sum += Base.prod(tuple) end
    info("\nLayer weights: ", layer_tuples[1][2:end], "\nWeights up to this layer:", sum, "\nOutput shape: ",layer_tuples[2][:])
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

    xlabel!("time (a.u.)")
    ylabel!("$(pulse_type) (a.u.)")
    
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
                           n_events::Union{String,Void}=nothing,
                           ybins::Int64=100)
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
    xlabel!("time (10 ns)")
    ylabel!("waveforms (10 ns/bin)")
    return figure
end

export plot_e_spectrum
"""
    plot_e_spectrum(energies::Array{Float64,1};
                    title::Union{String,Void}=nothing,
                    xmin::Int64=0,
                    xmax::Int64=2500,
                    bins::Int64=250,
                    logy=false)

Plots energy spectrum on histogram.
*energies* is an 1D array of floats with the event energies.
*title* will be the histogram title.
*xmin* is the lowest energy bin.
*xmax* is the highest energy bin.
*bins* is the number of bins. A rounded number + 1 is suggested for nice bin divisions.
*logy* is a bool to toggle log10 y axis.
"""
function plot_e_spectrum(energies::Array{Float32,1};
                         title::Union{String,Void}=nothing,
                         xmin::Int64=0,
                         xmax::Int64=2500,
                         bins::Int64=250,
                         logy=false)
    bin = linspace(xmin, xmax, bins)
    hist = plot(size=(15 * 39.37, 8 * 39.37))
    e_spectrum = fit(Histogram, energies, bin, closed=:left)
    if logy
        spectrum = plot!(e_spectrum, line=0, label="", yscale = :log10)
    else
        spectrum = plot!(e_spectrum, line=0, label="")
    end
    yaxis!("Counts ($(round((xmax-xmin)/(bins-1), 2)) keV/bin)")
    xaxis!("Energy in keV")
    title != nothing&&title!(title)
    return spectrum
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

export copypaste_pulses
"""
    copypaste_pulses(data::DLData, n::Int64)

Appends *data* to itself (*n*-1) times.
"""
function copypaste_pulses(data::DLData, n::Int64)
    if n==1
        #result = DLData(EventLibrary[])
        #push!(result.entries, flatten(cat([data.entries[i] for i in range(1,length(data))])))
        result = flatten(cat([data.entries[i] for i in range(1,length(data))]))
        deleteat!(data.entries,collect(2:1:length(data.entries)))
        # result is an eventlibrary
        return result
    else
        push!(data.entries,data.entries[1])
        return copypaste_pulses(data, n-1)
    end
end

export replicate_slow_pulses
"""
    replicate_slow_pulses(data::DLData, risetime_threshold::Int64=800)

Replicates pulses with risetime larger than *risetime_threshold* and appends the copied pulses to *data*.
"""
function replicate_slow_pulses(data; risetime_threshold::Int64=800, copy_n_times::Int64=1)
    @assert copy_n_times>=1 "*copy_n_times* must be at least 1"
    slow_pulses=filter(data, :risetime, x->x>risetime_threshold)
    info("$(eventcount(slow_pulses)) slow pulses will be copied $(copy_n_times) times.")
    info("$(eventcount(slow_pulses)*copy_n_times) more pulses in output.")
    # store multiplicated slow pulses
    result = DLData(EventLibrary[])
    # container of original pulses + added slow pulses
    enriched_with_slow_pulses = DLData(EventLibrary[])
    # loop over detectors
    for (idx, lib) in enumerate(slow_pulses)
        # append an eventlibrary to the result
        push!(result.entries,copypaste_pulses(DLData([lib]), copy_n_times))
        # append the extra slow pulses to the detector eventlibrary
        push!(enriched_with_slow_pulses.entries, flatten(cat([result.entries[idx],try data.entries[idx] catch; data end])))
    end
    return enriched_with_slow_pulses
end

export smart_subplots
"""
    smart_subplots(indices::Array, data::EventLibrary...; labelfont=14, tickfont=10, titlefont=10)

Makes subplots on a smart grid layout.
*data* are EventLibrary objects. The events corresponding to *indices* will be plot on top of each other.
"""

function smart_subplots(indices::Array, data::EventLibrary...; labelfont=14, tickfont=10, titlefont=10)
    # configure x axis
    t_axis = collect(sample_times(data[1]) * 1e6)
    t_axis -= t_axis[1]
    # configure layout
    # prime factorisation of n
    len = length(indices)
    primes = factor(Vector, len)
    # unique combinations of prime factors
    combinations = unique([primes[c] for c in [65-findin(bits(i),'1') for i=1:(2^length(primes)-1)]])
    # optimal combination ("most square shaped grid layout")
    opt = prod(combinations[findmin(abs.(prod.(combinations).-sqrt(len)))[2]])
    layout = (Int(floor(len/opt)), opt)
    gr(size=(layout[2]*300,layout[1]*200), legend=false)
    # make plots
    plots = []
    for i in indices
       push!(plots,plot(t_axis,[waveforms(data_)[:,i] for data_ in data], title="E=$(round(data[1][:E][i],4)) keV"))
    end
    # put plots onto main plot
    fig = plot(plots..., label="",layout=layout, titlefont=font(titlefont))
    xaxis!(L"Time\;(\mu s)",font(labelfont),font(tickfont))
    yaxis!(L"Charge\;(a.u.)",font(labelfont),font(tickfont))
    return fig
end
