N_THREADS=16

julia +1.10.7 -t "${N_THREADS}" --project results-pipeline/01_statistics.jl
julia +1.10.7 -t "${N_THREADS}" --project results-pipeline/02_timeseries.jl
julia +1.10.7 -t "${N_THREADS}" --project results-pipeline/03_activity.jl
julia +1.10.7 -t "${N_THREADS}" --project results-pipeline/04_sentiment.jl
julia +1.10.7 -t "${N_THREADS}" --project results-pipeline/05_SIEBC.jl
