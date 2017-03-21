# This file is a part of GERDADeepLearning.jl, licensed under the MIT License (MIT).

using MXNet, Plots


export plot_autoencoder
function plot_autoencoder(env::DLEnv, n::NetworkInfo, data::EventLibrary; count=20)
  batch_size = get_properties(env, "autoencoder")["batch_size"]

  if length(data) > count
    data = data[1:count]
  end

  dir = joinpath(env.dir, "plots", "autoencoder")
  isdir(dir) || mkdir(dir)

  println("Generating plots in $dir...")

  model = n.model

  visualize_1D_convolution(model, :conv1_weight, joinpath(dir,"filters1.png"))
  visualize_2D_convolution(model, :conv2_weight, joinpath(dir,"filter2"))

  if length(data) < batch_size
    plot_waveforms_padded = zeros(size(data.waveforms,1), batch_size)
    plot_waveforms_padded[:,1:length(data)] = data.waveforms
  else
    plot_waveforms_padded = data.waveforms
  end
  # plot_waveforms_padded = transform(plot_waveforms_padded)
  provider = provider = mx.ArrayDataProvider(:data => plot_waveforms_padded, batch_size=batch_size)
  plot_reconstructions([model], ["Reconstruction"], data.waveforms, provider, dir, file_prefix=data[:name]*"-", sample_count=length(data))

  loss = model.arch
  open(joinpath(dir,"graphviz.dot"), "w") do file
    println(file, mx.to_graphviz(loss))
  end

  println("Saved plots to $dir")

  return model, dir
end


function plot_reconstructions(models, names, waveforms, provider, plot_dir;
    file_prefix="", sample_count=20, transform=identity,
    bin_width_ns=10, y_label="Current pulse", diagram_font=font(14)
    )
  pred_evals = [transform(mx.predict(model, provider)) for model in models]
  truth = transform(waveforms)

  sample_count = min(sample_count, size(truth, 2))
  x_axis = linspace(0, (size(truth,1)-1)*bin_width_ns, size(truth,1))

  for i in 1:sample_count
    plot(x_axis, truth[:,i], label="Data", linewidth=2, legendfont=diagram_font, legend=:topright)
    for modelI in 1:length(models)
      plot!(x_axis, pred_evals[modelI][:,i], linewidth=3, label=names[modelI])
    end
   xaxis!("Time (ns)", diagram_font)
   yaxis!(y_label, diagram_font)
   savefig("$plot_dir/reconst-$(file_prefix)$i.png")
  end
end

export plot_waveforms
function plot_waveforms(data::Array{Float32, 2}, filepath::AbstractString; bin_width_ns=10, cut=nothing, diagram_font=font(16))
  if cut != nothing
    data = data[cut,:]
  end

  x_axis = linspace(0, (size(data,1)-1)*bin_width_ns, size(data,1))
  plot(size=(600, 400), legend=:none)
  for i in 1:size(data, 2)
    plot!(x_axis, data[:,i], linewidth=3)
  end
  xaxis!("Time (ns)", diagram_font)
  yaxis!("Current pulse", diagram_font)
  savefig(filepath)
end

function plot_waveforms(env::DLEnv, events::EventLibrary; count=4, bin_width_ns=10, cut=nothing)
  filepath = joinpath(env.dir, "plots", "waveforms-$(events[:name]).png")
  println("Plotting waveforms to $filepath")
  plot_waveforms(events.waveforms[:,1:count], filepath)
end


function visualize_1D_convolution(model, name, filename)
  for param in model.arg_params
    if(param[1] == name)
      filters = copy(param[2])[1,:,1,:]
      plot(filters)
      title!("Learned convolutional filters in first layer")
      xaxis!("Delta time")
      yaxis!("Filter value (current waveform)")
      savefig(filename)
    end
  end
end

function visualize_2D_convolution(model, name, filename_base)
  for param in model.arg_params
    if(param[1] == name)
      for i in 1:size(param[2], 4)
        filters = copy(param[2])[1,:,:,i]
        heatmap(filters)
        title!("Learned convolutional filters in second layer")
        xaxis!("Delta time")
        yaxis!("Feature map")
        savefig("$filename_base-$i.png")
      end
    end
  end
end


export plot_classifier_histogram
function plot_classifier_histogram(dir, events::EventLibrary, label_key, psd_key;
    diagram_font=font(14), nbins = 30)

  if !haskey(events, psd_key)
    throw(ArgumentError("events must have label 'psd'."))
  end

  events = filter(events, :E, e -> (e < 3800 && e > 600))

  bins = linspace(minimum(events.labels[psd_key]),maximum(events.labels[psd_key]), nbins)

  if haskey(events.labels, label_key)
    labels = events[label_key]
    psd = events[psd_key]

    psd_SSE = psd[find(l -> l==1, labels)]
    psd_MSE = psd[find(l -> l==0, labels)]

    # Overall PSD distribution
    histogram(psd_SSE, bins=bins, label="Tl DEP", legendfont=diagram_font, linewidth=0)
    histogram!(psd_MSE, bins=bins, label="Bi FEP", fillalpha=0.7, legendfont=diagram_font, linewidth=0)
    xaxis!("Classifier response", diagram_font)
    yaxis!("Event count", diagram_font)
    savefig(joinpath(dir,"Class distribution $(name(events)).png"))
  end

  histogram(events[psd_key], bins=bins, label="All", legendfont=diagram_font, linewidth=0)
  xaxis!("Classifier response", diagram_font)
  yaxis!("Event count", diagram_font)
  savefig(joinpath(dir,"Distribution $(name(events)).png"))

  histogram2d(convert(Array{Float64},events[:E]),
    convert(Array{Float64},events[psd_key]),
    nbins=100)
  savefig(joinpath(dir, "Energy distributions $(name(events)).png"))

  histogram2d(convert(Array{Float64}, 1:length(events.labels[psd_key])), convert(Array{Float64},events.labels[psd_key]))
  savefig(joinpath(dir, "Distribution over time $(name(events)).png"))

  if haskey(events.labels, :multiplicities)
    histogram2d(convert(Array{Float64},events.labels[:multiplicities]), convert(Array{Float64},events.labels[psd_key]))
    savefig(joinpath(dir, "Multiplicity correlation $(name(events)).png"))
  end

  return bins
end

export plot_energy_histogram
function plot_energy_histogram(env::DLEnv, events::EventLibrary, psd_cut, psd_key)
  all = copy(events)
  all.prop[:diagram_label] = "All events"
  cut = filter(all, psd_key, psd -> psd > psd_cut)
  cut.prop[:diagram_label] = "After PSD, cut value $(@sprintf("%.2f",psd_cut))"
  # range = linspace(1500, 2800, 150)
  plot_energy_histogram([all, cut], 150,
      joinpath(env.dir, "plots", "PSD-energies-cut$(@sprintf("%.3f",psd_cut)).png"))
end

function plot_energy_histogram(eventlibs::Array{EventLibrary}, nbins, filename;
  diagram_font=font(14), label=:diagram_label)

  histogram(xticks=[2000, 2500], legend=:bottomleft, legendfont=diagram_font)

  # Peaks before and after
  for events in eventlibs
    histogram!(events[:E], nbins=nbins, label=events[label], linewidth=0)
  end
  xaxis!("Energy (keV)", diagram_font)
  yaxis!("Events", :log10, (1,Inf), diagram_font)
  savefig(filename)
end

function plot_peak_efficiencies_vs_cut(file, events::EventLibrary, cut_key,
      peak_centers, peak_names;
      cut_count=40, diagram_font=font(14))
  cut_values = equal_counts_cut_values(events, cut_count, cut_key)
  curves = cut_efficiency_fluctuations(calc_efficiency_curves(events, cut_key, cut_values, peak_centers))
  amplitude_ratios = curves[:ratio]
  cut_values = transpose(curves[:cut_value])

  # Efficiencies vs Cut
  plot()
  for i in 1:length(peak_centers)
    plot!(cut_values, amplitude_ratios[i,:], label=peak_names[i],
      marker=:circle, markerstrokewidth=0.5, legendfont=diagram_font)
  end
  xaxis!("Cut value", (minimum(cut_values[2:end-1])-0.01, maximum(cut_values[2:end])+0.01), diagram_font)
  yaxis!("Efficiency", (0,1), diagram_font)
  savefig(file)

  return amplitude_ratios
end

function plot_efficiencies(dir, libs::Array{EventLibrary}, cut_key;
    diagram_font=font(10), plot_AoE=false, plot_guess=true)
  # hist_animation = @animate for i in 1:length(cut_values)
  #   cut = cut_values[i]
  #   plot_energy_histogram(dir*"PSD-energies-cut$(@sprintf("%.3f",cut)).png", events[:energies], accept, cut)
  # end
  # gif(hist_animation, dir*"PSD-energies-animation.gif", fps=3)

  peak_centers = [1592.5, 1620.7, 2103.5, 2614.5]
  peak_names = ["Tl DEP", "Bi FEP", "Tl SEP", "Tl FEP"]

  ratios = [plot_peak_efficiencies_vs_cut(joinpath(dir,"PSD Efficiencies vs Cut $(name(lib)).png"),
    lib, cut_key, peak_centers, peak_names) for lib in libs]

  # Background rejection vs Signal efficiency
  plot(legendfont=diagram_font, legend=:bottomleft)
  for (i,lib) in enumerate(libs)
    classifier_label = join(get_classifiers(lib), " + ")
    linetype = :line
    if startswith(name(lib), "train")
      classifier_label = classifier_label * " (training set)"
      linetype = :dash
    end
    ratios[i] = hcat(fill(0, size(ratios[i],1)), ratios[i], fill(1, size(ratios[i],1)) )
    plot!(ratios[i][1,:], 1-ratios[i][2,:],
        line=(2,linetype), label=classifier_label)
  end
  if plot_AoE
    plot!(get_AoE_efficiencies()[:,1], 1-get_AoE_efficiencies()[:,2],
        linewidth=2, label="A/E")
  end
  if plot_guess
    plot!([0,1],[1,0], linewidth=1, label="Random guessing")
  end
  xaxis!("Signal efficiency", (0.2,1), diagram_font)
  yaxis!("Background rejection", (0,1), diagram_font)
  savefig(joinpath(dir,"PSD Efficiency comparison.png"))
end

function get_AoE_efficiencies()
  file = open("/home/iwsatlas1/pholl/workspace/data/AoE_Performance.txt", "r")
  data = readdlm(file)
  return data
end

export plot_classifier
function plot_classifier(env::DLEnv, name, libs::EventLibrary...;
    classifier_key=:psd, label_key=:SSE, plot_AoE=false)
  dir = joinpath(env.dir, "plots", "PSD")
  if !isdir(dir)
    mkdir(dir)
  end

  for lib in libs
    plot_classifier_histogram(dir, lib, label_key, classifier_key)
  end
  plot_efficiencies(dir, collect(libs), classifier_key; plot_AoE=plot_AoE)
end