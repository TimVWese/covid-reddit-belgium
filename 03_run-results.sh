N_THREADS=16

julia -t "${N_THREADS}" --project results-pipeline/01_statistics.jl
julia -t "${N_THREADS}" --project results-pipeline/02_timeseries.jl
julia -t "${N_THREADS}" --project results-pipeline/03_activity.jl
julia -t "${N_THREADS}" --project results-pipeline/04_sentiment.jl
julia -t "${N_THREADS}" --project results-pipeline/05_SIEBC.jl
