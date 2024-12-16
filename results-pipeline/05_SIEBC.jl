using JLD2
using Turing
using SciMLSensitivity, ReverseDiff
using Interpolations
using Printf
using HypothesisTests
using Optim
using Random
using RollingFunctions

include(joinpath(@__DIR__, "..", "util.jl"))

# Allow to interpret some more dates to improve fitting at boundaries
MCMC_DIR = joinpath(@__DIR__, "..", "mcmc-chains")
SIEBC_DIR = joinpath(RESULT_DIR, "05_siebc")
broad_date_range=Date(2019,11,1):Date(2022,8,31)
date_range = Date(2020, 1, 1):Date(2022, 6, 30)
subreddits = ["belgium"]
topics = [lockdown, mask, vaccin]
nb_comment_threshold = 50

comments, submissions = get_comments_and_submissions(; discard=Dict(:author=>[AUTHOR_AUTO, AUTHOR_DELETE]), date_range=broad_date_range)

"""
   bc_kernel(; α=0.5, ϵ=0.1, type::Symbol=:logistic, truncated=true)

Return a function that performs a bounded confidence update between two values, `xb` and `xsig`.

# Arguments
- `α::Float=0.5`: The weight of `xb` in the update.
- `ϵ::Float=0.1`: The difference resulting in maximal effect
- `γ::Float=50`: Controls the shape of the curve, in case of `type=:logistic`.
- `type::Bool=:bell`: The type of update to perform:
    - `:linear`: Linear update without bounded confidence effects.
    - `:discrete`: Discrete update, the classical model.
    - `:bell`: Bell-shaped update, smooth approximation by derivate of bell curve.
    - `:logistic`: Logistic update, smooth approximation by logistic curve.
"""
function bc_kernel(; α=0.5, ϵ=0.1, γ=50, type::Symbol=:logistic)
    if type == :linear
        return (xb, xsig) -> (1. - α) * xb + α * xsig
    elseif type == :discrete
        return (xb, xsig) -> abs(xb - xsig) < ϵ ? (1. - α) * xb + α * xsig : xb
    elseif type == :bell
        A = α*exp(.5)
        E = -1. / (2*ϵ^2)
        return (xb, xsig) -> begin
            x = xsig - xb
            xb += A*x*exp(E*x^2)
            return xb
        end
    elseif type == :logistic
        ge = γ*ϵ^2
        return (xb, xsig) ->  begin
            x = xsig - xb
            xb += α*x / (1 + exp(γ*(x^2) - ge))
            return xb
        end
    else
        error("Unknown type: $type")
    end
end

@model function fit_bc(comments::Vector{T}, parents, replies; type=:logistic) where T
    α₁ ~ Exponential(.5)
    α₂ ~ Exponential(.5)
    ϵ ~ Uniform(0., 1.)
    bc1 = bc_kernel(; α=α₁, ϵ, type)
    bc2 = bc_kernel(; α=α₂, ϵ, type)

    states = Vector{T}(undef, length(comments))
    states[1] ~ Uniform(0., 1.)
    σ₁ ~ Exponential(.25)
    σ₂ ~ Exponential(.25)

    for t in axes(comments, 1)
        c = bc2(states[t], parents[t])
        comments[t] ~ truncated(Normal(c, σ₂); lower=0., upper=1.)
        if t == size(comments, 1)
            break
        end
        i = bc1(states[t], parents[t])
        for r in replies[t]
            i = bc1(i, r)
        end
        states[t+1] ~ truncated(Normal(i, σ₁); lower=0., upper=1.)
    end
end

"""
    get_author_comments(comments, submissions, topic; threshold=50, extended=false)

Get the comments on `topic` from authors who made at least `threshold` valid comments on said topic.
A valid comment has a valid bert sentiment value and a parent comment with a valid bert sentiment value.
Only the comments that are made and replied to by such authors are returned.
If `extended` is `false`, both the parent and the reply have to be on `topic`; otherwise,
the reply can be on any topic.

# Arguments
- `comments`: DataFrame containing comment data.
- `submissions`: DataFrame containing submission data.
- `topic`: The topic to filter comments by.
- `threshold`: Minimum number of comments an author must have to be included (default is 50).
- `extended`: Boolean flag to determine if all English comments should be considered for parent values (default is false).

# Returns
- `author_comments`: DataFrame of comments from authors who meet the criteria.
- `author_submissions`: DataFrame of submissions associated with the filtered comments.
"""
function get_author_comments(comments, submissions, topic, threshold)
    sort!(comments, :datetime)
    topic_comments = comments[select_lang(comments, "en") .&& comments.topic.==topic, :]
    add_parent_values!(topic_comments, submissions; cols=[:bert, :author, :topic])
    valid_comments = remove_invalid_rows(topic_comments, [:bert, :parent_bert, :parent_author])
    users = get_users(valid_comments, submissions)
    authors = users.author[users.nc .>= threshold]
    author_comments = valid_comments[ein(valid_comments.author, authors) .&&
        ein(valid_comments.parent_author, authors), :]
    author_submissions = submissions[ein(submissions.id, author_comments.submission_id), :]
    return author_comments, author_submissions
end

function retrieve_data(user::AbstractString, selected_comments::DataFrame, reference_comments::DataFrame)
    author_coms = selected_comments[selected_comments.author .== user, :]
    ref_coms = reference_comments[.!(ismissing.(reference_comments.bert)), :]
    Cs = author_coms.bert
    Ps = author_coms.parent_bert
    Rs = [[row.bert for row in eachrow(ref_coms) if !ismissing(row.parent_id) && !ismissing(row.bert) && row.parent_id[4:end] == c.id] for c in eachrow(author_coms)]
    Cs = Float64.(.5(1. .+ Cs))
    Ps = Float64.(.5(1. .+ Ps))
    Rs = [Float64.(.5(1. .+ r)) for r in Rs]
    return Cs, Tuple(Ps), Tuple(Rs)
end

function process(focal_user, comments; type=:logistic, N_samples_per=500, N_parallel_in=6, save_dir=nothing)
    Cs, Ps, Rs = retrieve_data(focal_user, comments, comments)

    model = fit_bc(Cs, Ps, Rs; type)
    chain = sample(model, NUTS(;adtype=AutoReverseDiff()), MCMCThreads(), N_samples_per, N_parallel_in)
    if !isnothing(save_dir)
        try
            JLD2.@save joinpath(save_dir, "$(focal_user).jld2") chain
        catch e
            open(joinpath(save_dir, "$(focal_user).err"), "w") do f
                showerror(f, e)
            end
        end
    end
    return chain
end

function handle(comments, submissions, topic, threshold; type=:logistic, N_samples_per=500,
                N_parallel_in=6, N_parallel_out=5, save_suffix="", mcmc_dir=MCMC_DIR)
    a_coms, _ = get_author_comments(comments, submissions, topic, threshold)
    save_suffix = save_suffix * "_" * string(type)
    save_dir = joinpath(mcmc_dir, "$(topic)$(save_suffix)/")
    !isdir(save_dir) && mkpath(save_dir)

    authors = unique(a_coms.author)
    pb = Progress(length(authors))
    do_step = idx -> begin
    if isfile(joinpath(save_dir, "$(authors[idx]).jld2"))
        next!(pb)
        return
    end
    process(
        authors[idx], a_coms; type, N_samples_per, N_parallel_in, save_dir
    )
    next!(pb)
    end

    sem = Base.Semaphore(N_parallel_out)
    Threads.@threads for idx in 1:length(authors)
        Base.acquire(() -> do_step(idx), sem)
    end
end

for subreddit in subreddits
    output_dir = joinpath(MCMC_DIR, subreddit)
    sub_coms = comments[comments.subreddit .== subreddit, :]
    sub_subs = submissions[submissions.subreddit .== subreddit, :]
    for topic in topics
        handle(sub_coms, sub_subs, topic, nb_comment_threshold; type=:logistic, mcmc_dir=output_dir)
        handle(sub_coms, sub_subs, topic, nb_comment_threshold; type=:linear, mcmc_dir=output_dir)
    end
end

#################
# Interpretation
#################
function order_chain(chain)
    parameter_syms = [:ϵ, :α₁, :α₂, :σ₁, :σ₂]
    state_syms = setdiff(chain.name_map.parameters, parameter_syms)
    all_syms = vcat(parameter_syms, state_syms)
    return hcat([Array(chain[sym])[:] for sym in all_syms]...)
end

function get_comments_and_chains(comments, data_dir)
    comments = comments[:, [:id, :author, :parent_id, :parent_author, :bert, :parent_bert]]
    authors = unique(comments.author)
    available_authors = [s[1:end-5] for s in readdir(data_dir) if occursin(".jld2", s)]
    @assert all([author in available_authors for author in authors]) &&
        all([author in authors for author in available_authors])
    a2idx = Dict([author => idx for (idx, author) in enumerate(authors)])
    chains = [order_chain(JLD2.load(joinpath(data_dir, "$(author).jld2"))["chain"]) for author in authors]

    comments.sent = Vector{Union{Missing, Float64}}(missing, size(comments, 1))
    comments.parent_sent = Vector{Union{Missing, Float64}}(missing, size(comments, 1))
    comments.internal = Vector{Union{Missing, Float64}}(missing, size(comments, 1))
    return comments, chains, a2idx
end

function sample_trajectory!(comments, a2idx, chain_samples; k=25, σ_mult=1, type=:bell)
    ϵ_idx, α₁_idx, α₂_idx, σ₁_idx, σ₂_idx = 1:5
    state_offset = 5

    bcs = [bc_kernel(; α=s[α₂_idx], ϵ=s[ϵ_idx], type) for s in chain_samples]
    σ₂s = [σ_mult*s[σ₂_idx] for s in chain_samples]
    current_index = ones(Int64, length(chain_samples))

    get_parents = ParentLookup(comments, comments[1:0,:])
    random_data = []

    get_internal_state = a_idx -> begin
        cs_idx = state_offset + current_index[a_idx]
        current_index[a_idx] += 1
        return chain_samples[a_idx][cs_idx]
    end

    for comment in eachrow(comments)
        author_idx = a2idx[comment.author]
        parent = get_parents(comment)
        parent_sent = ismissing(parent) ? comment.parent_bert : parent.sent
        ismissing(parent) && push!(random_data, parent_sent)

        author_state = get_internal_state(author_idx)
        comment.internal = author_state
        comment.sent = bcs[author_idx](author_state, parent_sent)
        comment.parent_sent = parent_sent
        comment.sent = rand(truncated(Normal(comment.sent, σ₂s[author_idx]), 0., 1.))
    end

    comments.sent = (2*comments.sent) .- 1
    comments.parent_sent = (2*comments.parent_sent) .- 1

    return vcat(random_data, comments.sent)
end

function infer_type(suffix)
    type = split(suffix, "_")[end]
    type = (type in ("bell", "logistic", "linear", "discrete")) ? Symbol(type) : :bell
    return type
end

"""
    interpret_data(comments, p_val=0.05; N_samples=nothing)

Interpret the data from comments and perform statistical analysis.

# Arguments
- `comments::DataFrame`: A DataFrame containing the comments data with columns `:id`, `:author`, `:parent_id`, `:parent_author`, `:bert`, and `:parent_bert`.
- `p_val::Float64`: The p-value threshold for statistical significance (default is 0.05).
- `N_samples::Union{Int, Nothing}`: The number of samples to draw. If `nothing`, the number of samples is determined by the length of the chains (default is `nothing`).

# Returns
- `hs::Vector{Float64}`: A vector of diagonalness scores for each sample.
- `all_sents::Vector{Float64}`: A vector of all sampled sentences.

# Description
This function processes the comments data, ensuring that all authors have corresponding MCMC chains available.
It then samples trajectories from these chains and performs statistical analysis to compute diagonalness scores and collect all sampled sentences.
"""
function interpret_data(comments, data_dir; σ_mult=1., p_val=0.05, N_samples=nothing)
    comments, chains, a2idx = get_comments_and_chains(comments, data_dir)
    type = infer_type(data_dir)

    selector = (chains, _) -> [rand(eachrow(chain)) for chain in chains]
    if N_samples == nothing
        N_samples = size(chains[1], 1)
        @assert all(chain -> size(chain, 1) == N_samples, chains)
        selector = (chains, idx) -> [chain[idx, :] for chain in chains]
    end

    hs = Vector{Float64}(undef, N_samples)
    all_sents = Vector{Float64}(undef, size(comments, 1)*N_samples)
    mean_internal = zeros(Float64, size(comments, 1))
    mean_diffs = zeros(Float64, (41, 41))

    @showprogress for i in 1:N_samples
        random_data = sample_trajectory!(comments, a2idx, selector(chains, i); σ_mult, type)
        mean_internal += comments.internal
        D, diff = hist_and_diff(comments.sent, comments.parent_sent, random_data; p_val)

        hs[i] = diagonalness(diff)
        all_sents[(i-1)*size(comments, 1)+1:i*size(comments, 1)] = comments.sent
        mean_diffs .+= diff
    end

    return hs, all_sents, mean_internal / N_samples, mean_diffs / N_samples
end

function get_observed_homophily(comments, submissions; p_val=0.05)
    random_data = vcat(comments.bert, submissions.bert)
    rd_f = (n) -> rand(random_data, n)
    Rd_f = () -> get_random_histogram(comments.bert, rd_f(nrow(comments)))
    D = get_structured_histogram(comments.bert, comments.parent_bert)
    diff = get_2d_diff(D, Rd_f, p_val)
    return diagonalness(diff)
end

function earthmoverdistance(a::Vector, b::Vector; width = 0.05)
    return width*sum( abs, cumsum( a ) .- cumsum( b ) )
end

function get_histogram(observed, predicted; filename=missing, width=0.05)
    to_weights = (data) -> begin
        hist = fit(Histogram, data, (-1-width/2):width:(1+width/2))
        return hist.weights ./ (width*sum(hist.weights))
    end
    xs = -1:width:1 # midpoints
    y_obs = to_weights(observed)
    y_pred = to_weights(predicted)
    df = DataFrame(x=xs, y_obs=y_obs, y_pred=y_pred)
    if !ismissing(filename)
        @info "Wassertein distance for $(split(filename, "/")[end]): $(earthmoverdistance(y_obs, y_pred; width=width))"
        CSV.write(filename, df)
    end
    return df
end

function construct_sigma_loss(all_comments, all_submissions, mcmc_dir, suffix; threshold=40, N_samples=25, topics = [lockdown, mask, vaccin], seed=1234)
    comments = Dict()
    chain_samples = Dict()
    a2idx = Dict()
    all_sents = Dict()
    type = infer_type(suffix)

    for topic in topics
        a_coms, _ = get_author_comments(all_comments, all_submissions, topic, threshold)
        comments[topic], chains, a2idx[topic] = get_comments_and_chains(a_coms, joinpath(mcmc_dir, "$(topic)_$(suffix)"))
        chain_samples[topic] = [[rand(eachrow(chain)) for chain in chains] for _ in 1:N_samples]
        all_sents[topic] = Vector{Float64}(undef, size(comments[topic], 1)*N_samples)
    end

    return σ_exps -> begin
        if length(σ_exps) == 1 && length(topics) >1
            σ_exps = fill(σ_exps[1], length(topics))
        end
        W = 0.
        Random.seed!(seed)
        for (t_idx, topic) in enumerate(topics)
            t_coms = comments[topic]
            for (i, chain_sample) in enumerate(chain_samples[topic])
                sample_trajectory!(t_coms, a2idx[topic], chain_sample; σ_mult=σ_exps[t_idx], type)
                all_sents[topic][(i-1)*size(t_coms, 1)+1:i*size(t_coms, 1)] .= t_coms.sent
            end
            hist = get_histogram(t_coms.bert, all_sents[topic])
            W += earthmoverdistance(hist.y_obs, hist.y_pred)^2
        end
        return sqrt(W)
    end
end

function sample_topics!(
        comments, submissions, suffix, subreddit, Ws, hs, full_hs=nothing; N_samples=nothing,
        topics=topics, threshold=nb_comment_threshold, mcmc_dir=MCMC_DIR, result_dir=SIEBC_DIR,
        optim_init=[1.,1.,1.], optim_samples=250, optim_options=Optim.Options(iterations=250, show_trace=true),
    )
    sub_coms = comments[comments.subreddit .== subreddit, :]
    sub_subs = submissions[submissions.subreddit .== subreddit, :]

    subreddit_dir = joinpath(mcmc_dir, subreddit)
    sl = construct_sigma_loss(sub_coms, sub_subs, subreddit_dir, suffix; N_samples=optim_samples, topics, threshold)
    σ_opt = optimize(sl, optim_init, NelderMead(), optim_options)

    Threads.@threads for t_idx in eachindex(topics)
        topic = topics[t_idx]
        a_coms, a_subs = get_author_comments(sub_coms, sub_subs, topic, nb_comment_threshold)
        @info "Processing $(topic): $(length(unique(a_coms.author))) authors that made $(nrow(a_coms)) comments."
        data_dir = joinpath(subreddit_dir, "$(topic)_$(suffix)")
        sample_hs, sents, interns, diff = interpret_data(a_coms, data_dir; N_samples, σ_mult= σ_opt.minimizer[t_idx])
        hist = get_histogram(a_coms.bert, sents; filename=(ismissing(result_dir) ? missing : joinpath(result_dir, "$(subreddit)_$(topic)_histogram.csv")))
        (!ismissing(result_dir)) && get_state_evolution(a_coms, interns; write_path=joinpath(result_dir, "$(subreddit)_$(topic)_internal_state.csv"))

        Ws[topic][suffix] = earthmoverdistance(hist.y_obs, hist.y_pred)
        hs[topic][suffix] = median(sample_hs)
        hs[topic]["observed"] = get_observed_homophily(a_coms, a_subs)
        (!isnothing(full_hs)) && (full_hs[topic] = sample_hs)
    end
end

function create_alpha_table(topics, suffix, mcmc_dir, result_dir; p_val=0.05)
    output = open(joinpath(result_dir, "alpha_table.tex"), "w")
    @printf output "\\begin{tabular}{r|ccc}\n"
    @printf output "& \\( \\alpha_u \\)    & \\( \\alpha_e \\)    & \\( \\kappa \\)  \\\\\\hline\n"
    test_per_user = Dict()
    full_tests = Dict()
    for topic in topics
        total_true = 0
        total_count = 0
        α₁s = []
        α₂s = []
        topic_dir = joinpath(mcmc_dir, "$(topic)_$(suffix)")
        test_per_user[topic] = []
        for f in readdir(topic_dir)
            chain = JLD2.load(joinpath(topic_dir, f))["chain"]
            chain = Array(chain[[:α₁, :α₂]])
            total_true += count(chain[:,1] .< chain[:,2])
            total_count += size(chain, 1)
            push!(test_per_user[topic], MannWhitneyUTest(Float64.(chain[:,1]), Float64.(chain[:,2])))
            push!(α₁s, mean(chain[:,1]))
            push!(α₂s, mean(chain[:,2]))
        end
        prop_samp = total_true / total_count
        prop_users = count((α₁s .< α₂s) .&& (pvalue.(test_per_user[topic]) .< p_val)) / length(α₁s)
        full_tests[topic] = MannWhitneyUTest(Float64.(α₁s), Float64.(α₂s))

        @printf output "%s && \\( %0.5f \\pm %0.5f \\) & \\( %0.5f \\pm %0.5f \\) & %0.5f \\\\\n" string(topic) mean(α₁s) std(α₁s) mean(α₂s) std(α₂s) prop_users
    end
    close(output)
    return full_tests, test_per_user
end

function export_boxplot(predicted, filename)
    df = DataFrame([Symbol(key) => value for (key, value) in predicted])
    CSV.write(filename, df)
end

function get_state_interpolator(comments)
    xs = datetime2unix.(comments.datetime)
    ys = comments.state
    itp = nrow(comments) > 1 ? interpolate((xs,), ys, Gridded(Linear())) : x -> ys[1]
    return x -> begin
        xu = datetime2unix(x)
        if !(minimum(xs) <= xu <= maximum(xs))
            return missing
        end
        return itp(xu)
    end
end

function get_state_evolution(comments, internals; q=.25, min_data=16, window=14, write_path=nothing, date_range=date_range)
    comments.state = internals
    ts = minimum(comments.date):Day(1):maximum(comments.date)
    itps = [get_state_interpolator(comments[comments.author .== a, :]) for a in unique(comments.author)]
    A = hcat([[itps[i](DateTime(t)) for i in eachindex(itps)] for t in ts]...) # Internal data
    rough_valid = (count(.!ismissing.(A), dims=1) .> 0)[:]
    A = A[:, rough_valid]
    ts = ts[rough_valid]
    fine_valid = (count(.!ismissing.(A), dims=1) .> min_data)[:]

    rlm = d -> rollmean(d, window)
    q1 = rlm([quantile(skipmissing(a), q) for a in eachcol(A)])
    ms = rlm([median(skipmissing(a)) for a in eachcol(A)])
    q2 = rlm([quantile(skipmissing(a), 1-q) for a in eachcol(A)])

    fine_valid = fine_valid .&& ein(ts, date_range)
    ts = ts[fine_valid]
    fine_valid = fine_valid[7:end-7]
    q1 = q1[fine_valid]
    ms = ms[fine_valid]
    q2 = q2[fine_valid]

    q1 = 2*q1 .- 1.
    ms = 2*ms .- 1.
    q2 = 2*q2 .- 1.

    pd = per_day(comments, :bert=>median)
    obs = rlm(pd.bert[rough_valid])[fine_valid]

    if !isnothing(write_path)
        df = DataFrame(date=ts, q1=q1, median=ms, q2=q2, obs=obs)
        CSV.write(write_path, df)
    end

    return ts, q1, ms, q2, obs
end

isdir(SIEBC_DIR) || mkpath(SIEBC_DIR)

for subreddit in subreddits
    suffix = "logistic"

    types = ["observed", "logistic", "linear"]
    result_df = DataFrame()
    result_df.topic = vcat([fill(topic, length(types)) for topic in topics]...)
    result_df.type = vcat([types for _ in topics]...)

    Ws = Dict()
    hs = Dict()
    full_hs = Dict()

    for topic in topics
        Ws[topic] = Dict()
        hs[topic] = Dict()
        Ws[topic]["observed"] = 0.
    end

    sample_topics!(comments, submissions, suffix, subreddit, Ws, hs, full_hs; optim_init=[.6,.6,.6])
    export_boxplot(full_hs, joinpath(SIEBC_DIR, "$(subreddit)_homophily.csv"))
    create_alpha_table(topics, suffix, joinpath(MCMC_DIR, subreddit), SIEBC_DIR)

    suffix="linear"
    sample_topics!(comments, submissions, suffix, subreddit, Ws, hs; result_dir=missing)

    result_df.W = [Ws[row.topic][row.type] for row in eachrow(result_df)]
    result_df.h = [hs[row.topic][row.type] for row in eachrow(result_df)]
    CSV.write(joinpath(SIEBC_DIR, "$(subreddit)_measures.csv"), result_df)
end

