using JLD2
using Turing
using SciMLSensitivity, ReverseDiff
using Interpolations
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
nb_comment_threshold = 40

comments, submissions = get_comments_and_submissions(; discard=Dict(:author=>[AUTHOR_AUTO, AUTHOR_DELETE]), date_range=broad_date_range)

"""
    bc_kernel(; α=0.5, ϵ=0.1, γ=50, type::Symbol=:logistic)

Return a function `(xb, xsig)` that performs a bounded-confidence update of `xb` towards `xsig`.

# Arguments
- `α=0.5`: The weight given to the signal `xsig` in the update.
- `ϵ=0.1`: The difference at which the bounded-confidence effect kicks in.
- `γ=50`: Controls the steepness of the curve when `type=:logistic`.
- `type::Symbol=:logistic`: The type of update:
    - `:linear`: Linear update without bounded-confidence effects.
    - `:discrete`: Discrete update, the classical model.
    - `:bell`: Bell-shaped smooth approximation (derivative of a bell curve).
    - `:logistic`: Logistic smooth approximation.
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

@model function fit_bc_stateless(comments::Vector{T}, parents, replies; type=:logistic) where T
    α ~ Exponential(.5)
    ϵ ~ Uniform(0., 1.)
    bc = bc_kernel(; α, ϵ, type)

    initial_state ~ Uniform(0., 1.)
    σ ~ Exponential(.25)

    current_state = initial_state
    for t in axes(comments, 1)
        c = bc(current_state, parents[t])
        comments[t] ~ truncated(Normal(c, σ); lower=0., upper=1.)
        if t == size(comments, 1)
            break
        end
        for r in replies[t]
            current_state = bc(current_state, r)
        end
    end
end

"""
    get_author_comments(comments, submissions, topic, threshold)

Get the English comments on `topic` from authors who made at least `threshold` valid
comments on it. A valid comment has a bert sentiment value and a parent (comment or
submission) with a bert sentiment value. Only comments whose author and parent author
both clear the threshold are returned.

# Returns
- `author_comments`: comments from authors who meet the criteria.
- `author_submissions`: submissions associated with those comments.
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

function process(focal_user, comments; model_type=:internal, type=:logistic, N_samples_per=500, N_parallel_in=6, save_dir=nothing)
    Cs, Ps, Rs = retrieve_data(focal_user, comments, comments)

    model = model_type == :stateless ? fit_bc_stateless(Cs, Ps, Rs; type) : fit_bc(Cs, Ps, Rs; type)
    chain = sample(model, NUTS(;adtype=AutoReverseDiff()), MCMCThreads(), N_samples_per, N_parallel_in)
    plls = pointwise_loglikelihoods(model, chain)
    if !isnothing(save_dir)
        try
            JLD2.@save joinpath(save_dir, "$(focal_user).jld2") chain plls
        catch e
            open(joinpath(save_dir, "$(focal_user).err"), "w") do f
                showerror(f, e)
            end
        end
    end
    return chain
end

function handle(comments, submissions, topic, threshold; model_type=:internal, type=:logistic, N_samples_per=500,
                N_parallel_in=6, N_parallel_out=5, save_suffix="", mcmc_dir=MCMC_DIR)
    a_coms, _ = get_author_comments(comments, submissions, topic, threshold)
    save_suffix = save_suffix * "_" * string(type)
    (model_type == :stateless) && (save_suffix *= "_stateless")
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
        authors[idx], a_coms; model_type, type, N_samples_per, N_parallel_in, save_dir
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
        handle(sub_coms, sub_subs, topic, nb_comment_threshold; model_type=:stateless, type=:logistic, N_samples_per=100, mcmc_dir=output_dir)
    end
end

#################
# Interpretation
#################
function order_chain(chain; model_type=:internal)
    if model_type == :stateless
        parameter_syms = [:ϵ, :α, :σ, :initial_state]
    else
        parameter_syms = [:ϵ, :α₁, :α₂, :σ₁, :σ₂]
    end
    state_syms = setdiff(chain.name_map.parameters, parameter_syms)
    all_syms = vcat(parameter_syms, state_syms)
    return hcat([Array(chain[sym])[:] for sym in all_syms]...)
end

function get_comments_chains_plls(comments, data_dir)
    comments = comments[:, [:id, :author, :parent_id, :parent_author, :bert, :parent_bert]]
    authors = unique(comments.author)
    available_authors = [s[1:end-5] for s in readdir(data_dir) if occursin(".jld2", s)]
    @assert all([author in available_authors for author in authors]) &&
        all([author in authors for author in available_authors])
    a2idx = Dict([author => idx for (idx, author) in enumerate(authors)])
    model_type = occursin("stateless", data_dir) ? :stateless : :internal
    chains = [order_chain(JLD2.load(joinpath(data_dir, "$(author).jld2"))["chain"]; model_type) for author in authors]
    plls = [JLD2.load(joinpath(data_dir, "$(author).jld2"))["plls"] for author in authors]

    comments.sent = Vector{Union{Missing, Float64}}(missing, size(comments, 1))
    comments.parent_sent = Vector{Union{Missing, Float64}}(missing, size(comments, 1))
    comments.internal = Vector{Union{Missing, Float64}}(missing, size(comments, 1))
    return comments, chains, plls, a2idx
end

function sample_trajectory!(comments, a2idx, chain_samples; σ_mult=1, type=:bell, model_type=:internal)
    if model_type == :stateless
        ϵ_idx, α_idx, σ_idx, initial_state_idx = 1:4
        bcs = [bc_kernel(; α=s[α_idx], ϵ=s[ϵ_idx], type) for s in chain_samples]
        σs = [σ_mult*s[σ_idx] for s in chain_samples]
        current_states = [s[initial_state_idx] for s in chain_samples]
    else
        ϵ_idx, α₁_idx, α₂_idx, σ₁_idx, σ₂_idx = 1:5
        state_offset = 5
        bcs = [bc_kernel(; α=s[α₂_idx], ϵ=s[ϵ_idx], type) for s in chain_samples]
        σ₂s = [σ_mult*s[σ₂_idx] for s in chain_samples]
        current_index = ones(Int64, length(chain_samples))
    end

    get_parents = ParentLookup(comments, comments[1:0,:])
    random_data = []

    get_internal_state = a_idx -> begin
        if model_type == :stateless
            return current_states[a_idx]
        else
            cs_idx = state_offset + current_index[a_idx]
            current_index[a_idx] += 1
            return chain_samples[a_idx][cs_idx]
        end
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
        
        σ_val = model_type == :stateless ? σs[author_idx] : σ₂s[author_idx]
        comment.sent = rand(truncated(Normal(comment.sent, σ_val), 0., 1.))
        
        # For stateless model, current_state is updated deterministically during MCMC fitting
        # No need to update here during simulation
    end

    comments.sent = (2*comments.sent) .- 1
    comments.parent_sent = (2*comments.parent_sent) .- 1

    return vcat(random_data, comments.sent)
end

"""
    pll_dict_to_matrix(pll_dict) -> Matrix{Float64}

Convert the Dict returned by `pointwise_loglikelihoods` into a
(n_observations × n_samples) matrix expected by `WAIC`.

Keys are sorted by their index so rows are in observation order.
"""
function pll_dict_to_matrix(pll_dict::AbstractDict)
    sorted_keys = sort(collect(keys(pll_dict)),
                       by = k -> parse(Int, match(r"\[(\d+)\]", k).captures[1]))
    return reduce(vcat, [vec(pll_dict[k])' for k in sorted_keys])
end

"""
    WAIC(pll_matrix::AbstractMatrix)

Compute WAIC from a pointwise log-likelihood matrix of shape
(n_observations × n_samples).
"""
function WAIC(ppls::AbstractMatrix)
    lppd = sum(log.(mean(exp.(ppls), dims=2)))
    p_eff = sum(var(ppls, dims=2))
    return -2 * (lppd - p_eff)
end
WAIC(ppls::AbstractDict) = WAIC(pll_dict_to_matrix(ppls))

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
function interpret_data(comments, data_dir; σ_mult=1., p_val=0.05, N_samples=nothing, width=0.05)
    comments, chains, plls, a2idx = get_comments_chains_plls(comments, data_dir)
    type = infer_type(data_dir)
    model_type = occursin("stateless", data_dir) ? :stateless : :internal

    selector = (chains, _) -> [rand(eachrow(chain)) for chain in chains]
    if isnothing(N_samples)
        N_samples = size(chains[1], 1)
        @assert all(chain -> size(chain, 1) == N_samples, chains)
        selector = (chains, idx) -> [chain[idx, :] for chain in chains]
    end

    hs = Vector{Float64}(undef, N_samples)
    all_sents = Vector{Float64}(undef, size(comments, 1)*N_samples)
    mean_internal = zeros(Float64, size(comments, 1))
    n_bins = Int(2 / width)
    mean_diffs = zeros(Float64, (n_bins, n_bins))

    @showprogress for i in 1:N_samples
        random_data = sample_trajectory!(comments, a2idx, selector(chains, i); σ_mult, type, model_type)
        mean_internal += comments.internal
        _, diff = hist_and_diff(comments.sent, comments.parent_sent, random_data; p_val, width)

        hs[i] = diagonalness(diff)
        all_sents[(i-1)*size(comments, 1)+1:i*size(comments, 1)] = comments.sent
        mean_diffs .+= diff
    end

    return hs, all_sents, mean_internal / N_samples, mean_diffs / N_samples, WAIC.(plls)
end

function get_observed_homophily(comments, submissions; p_val=0.05)
    random_data = vcat(comments.bert, submissions.bert)
    rd_f = (n) -> rand(random_data, n)
    Rd_f = () -> get_unconditional_histogram(comments.bert, rd_f(nrow(comments)))
    D = get_structured_histogram(comments.bert, comments.parent_bert)
    diff = get_2d_diff(D, Rd_f, p_val)
    return diagonalness(diff)
end

function earthmoverdistance(a::Vector, b::Vector; width = 0.05)
    return width*sum( abs, cumsum( a ) .- cumsum( b ) )
end

function wasserstein_distance(a::Vector, b::Vector)
    ecdf_a = ecdf(Float64.(skipmissing(a)))
    ecdf_b = ecdf(Float64.(skipmissing(b)))
    all_points = sort(union(a, b))
    
    # Calculate the W1 distance as integral of |F_a(x) - F_b(x)|
    distance = 0.0
    
    for i in 1:(length(all_points)-1)
        x1, x2 = all_points[i], all_points[i+1]
        y1 = abs(ecdf_a(x1) - ecdf_b(x1))
        distance += y1 * (x2 - x1)
    end
    
    return distance
end

function KS_distance(a::Vector, b::Vector)
    ecdf_a = ecdf(Float64.(skipmissing(a)))
    ecdf_b = ecdf(Float64.(skipmissing(b)))
    eval_points = sort(union(a, b))
    @assert ecdf_a(1.) == 1. && ecdf_b(1.) == 1.
    return maximum(abs.(ecdf_a.(eval_points) .- ecdf_b.(eval_points)))
end

function get_histogram(observed, predicted; filename=missing, width=0.05)
    xs = (-1+width/2):width:(1-width/2) # midpoints
    y_obs = histogram_weights(observed; width)
    y_pred = histogram_weights(predicted; width)
    df = DataFrame(x=xs, y_obs=y_obs, y_pred=y_pred)
    if !ismissing(filename)
        @info "EM distance for $(split(filename, "/")[end]): $(earthmoverdistance(y_obs, y_pred; width=width))"
        @info "Wasserstein distance for $(split(filename, "/")[end]): $(wasserstein_distance(observed, predicted))"
        @info "KS distance for $(split(filename, "/")[end]): $(KS_distance(observed, predicted))"
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
    model_type = occursin("stateless", suffix) ? :stateless : :internal

    for topic in topics
        a_coms, _ = get_author_comments(all_comments, all_submissions, topic, threshold)
        comments[topic], chains, _, a2idx[topic] = get_comments_chains_plls(a_coms, joinpath(mcmc_dir, "$(topic)_$(suffix)"))
        chain_samples[topic] = [[rand(eachrow(chain)) for chain in chains] for _ in 1:N_samples]
        all_sents[topic] = Vector{Float64}(undef, size(comments[topic], 1)*N_samples)
    end

    return σ_exps -> begin
        if length(σ_exps) == 1 && length(topics) >1
            σ_exps = fill(σ_exps[1], length(topics))
        end
        if any(σ_exps .<= 0.)
            return Inf
        end
        KS = 0.
        Random.seed!(seed)
        for (t_idx, topic) in enumerate(topics)
            t_coms = comments[topic]
            for (i, chain_sample) in enumerate(chain_samples[topic])
                sample_trajectory!(t_coms, a2idx[topic], chain_sample; σ_mult=σ_exps[t_idx], type, model_type)
                all_sents[topic][(i-1)*size(t_coms, 1)+1:i*size(t_coms, 1)] .= t_coms.sent
            end
            KS += KS_distance(t_coms.bert, all_sents[topic])^2
        end
        return sqrt(KS)
    end
end

function sample_topics!(
        comments, submissions, suffix, subreddit, EMs, Ws, KSs, hs, WAICs, Ns, full_hs=nothing; N_samples=nothing,
        topics=topics, threshold=nb_comment_threshold, mcmc_dir=MCMC_DIR, result_dir=SIEBC_DIR,
        optim_init=[1.,1.,1.], optim_samples=100, optim_options=Optim.Options(iterations=250, show_trace=true),
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
        sample_hs, sents, interns, diff, waics = interpret_data(a_coms, data_dir; N_samples, σ_mult= σ_opt.minimizer[t_idx])
        hist = get_histogram(a_coms.bert, sents; filename=(ismissing(result_dir) ? missing : joinpath(result_dir, "$(subreddit)_$(topic)_histogram.csv")))

        if !ismissing(result_dir)
            write_path = joinpath(result_dir, "$(subreddit)_$(topic)_internal_state.csv")
            periods = KEYDATES[topic].date
            get_state_evolution(a_coms, interns; write_path, periods)
        end

        EMs[topic][suffix] = earthmoverdistance(hist.y_obs, hist.y_pred; width=0.05)
        Ws[topic][suffix] = wasserstein_distance(a_coms.bert, sents)
        KSs[topic][suffix] = KS_distance(a_coms.bert, sents)
        hs[topic][suffix] = sample_hs
        hs[topic]["observed"] = get_observed_homophily(a_coms, a_subs)
        WAICs[topic][suffix] = waics
        Ns[topic] = (length(unique(a_coms.author)), nrow(a_coms))
        (!isnothing(full_hs)) && (full_hs[topic] = sample_hs)
    end
end

function create_alpha_table(topics, suffix, mcmc_dir, result_dir; p_val=0.05)
    output = open(joinpath(result_dir, "alpha_table.tex"), "w")
    @printf output "\\begin{tabular}{r|cccc}\n"
    @printf output "& \\( \\epsilon \\) [9?\\%% CI]    & \\( \\alpha_u \\) [9?\\%% CI]    & \\( \\alpha_e \\) [9?\\%% CI]    & \\( \\kappa \\)  \\\\\\hline\n"
    test_per_user = Dict()
    full_tests = Dict()
    ql = p_val / 2
    qh = 1 - ql
    for topic in topics
        total_true = 0
        total_count = 0
        ϵs = Float64[]
        α₁s = Float64[]
        α₂s = Float64[]
        topic_dir = joinpath(mcmc_dir, "$(topic)_$(suffix)")
        test_per_user[topic] = []
        for f in readdir(topic_dir)
            chain = JLD2.load(joinpath(topic_dir, f))["chain"]
            chain = Array(chain[[:α₁, :α₂, :ϵ]])
            total_true += count(chain[:,1] .< chain[:,2])
            total_count += size(chain, 1)
            push!(test_per_user[topic], MannWhitneyUTest(Float64.(chain[:,1]), Float64.(chain[:,2])))
            push!(α₁s, mean(chain[:,1]))
            push!(α₂s, mean(chain[:,2]))
            push!(ϵs, 2*mean(chain[:,3])) # transfer to original scale, not necessary for α since the difference is already doubled
        end
        prop_samp = total_true / total_count
        prop_users = count((α₁s .< α₂s) .&& (pvalue.(test_per_user[topic]) .< p_val)) / length(α₁s)
        full_tests[topic] = MannWhitneyUTest(Float64.(α₁s), Float64.(α₂s))

        e_mean = fmt_digit(mean(ϵs), 5)
        e_std = fmt_digit(std(ϵs), 5)
        e_ql = fmt_digit(quantile(ϵs, ql), 5)
        e_qh = fmt_digit(quantile(ϵs, qh), 5)
        a1_mean = fmt_digit(mean(α₁s), 5)
        a1_std = fmt_digit(std(α₁s), 5)
        a1_ql = fmt_digit(quantile(α₁s, ql), 5)
        a1_qh = fmt_digit(quantile(α₁s, qh), 5)
        a2_mean = fmt_digit(mean(α₂s), 5)
        a2_std = fmt_digit(std(α₂s), 5)
        a2_ql = fmt_digit(quantile(α₂s, ql), 5)
        a2_qh = fmt_digit(quantile(α₂s, qh), 5)
        prop_s = fmt_digit(prop_users, 5)
        @printf output "\\topic{%s} & \\( %s \\pm %s \\, [%s, %s] \\) & \\( %s \\pm %s \\, [%s, %s] \\) & \\( %s \\pm %s \\, [%s, %s] \\) & %s \\\\\n" uppercasefirst(string(topic)) e_mean e_std e_ql e_qh a1_mean a1_std a1_ql a1_qh a2_mean a2_std a2_ql a2_qh prop_s
    end
    @printf output "\\end{tabular}\n"
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

_ci_quantile(v::AbstractVector, p) = length(v) > 1 ? quantile(v, p) : missing
_ci_quantile(v, p) = missing

function result_df_to_latex(result_df; filename=nothing, measures = Dict("dh" => "\\( \\Delta h \\)", "WAIC_mean" => "\\( WAIC \\)"))
    escape_tex(s) = replace(string(s), "_" => " ")

    topics = collect(unique(result_df.topic))
    types = collect(unique(result_df.type))
    nmeas = length(measures)
    Ncols = 1 + nmeas * length(topics)   # first col for type
    colspec = "l" * repeat("c", Ncols-1)

    has_ci(m) = Symbol(m * "_q025") in propertynames(result_df) && Symbol(m * "_q975") in propertynames(result_df)

    buf = IOBuffer()
    println(buf, "\\begin{tabular}{" * colspec * "}")
    # First header row: empty first cell and topic names spanning nmeas columns
    header1 = " & " * join([ "\\multicolumn{$(nmeas)}{c}{\\emph{" * escape_tex(t) * "}}" for t in topics ], " & ")
    println(buf, header1 * " \\\\")
    # Subheader row: 'Model' label and measure names
    header2 = "\\textbf{Model} & " * join([ms for _ in topics for (_, ms) in measures ], " & ")
    println(buf, header2 * " \\\\ \\hline")
    # Data rows: one per model type
    for ty in types
        vals = [ty]
        for t in topics
            row = result_df[(result_df.topic .== t) .&& (result_df.type .== ty), :]
            for (m, _) in measures
                if nrow(row) == 0
                    push!(vals, "--")
                else
                    v = row[1, Symbol(m)]
                    lo = has_ci(m) ? row[1, Symbol(m * "_q025")] : missing
                    hi = has_ci(m) ? row[1, Symbol(m * "_q975")] : missing
                    if !ismissing(lo) && !ismissing(hi)
                        push!(vals, "$(fmt_digit(Float64(v), 6)) [$(fmt_digit(lo, 6)), $(fmt_digit(hi, 6))]")
                    else
                        push!(vals, fmt_digit(Float64(v), 6))
                    end
                end
            end
        end
        println(buf, join(vals, " & ") * " \\\\")
    end
    println(buf, "\\end{tabular}")

    latex = String(take!(buf))
    if !isnothing(filename)
        open(filename, "w") do io
            write(io, latex)
        end
    end

    return latex
end

isdir(SIEBC_DIR) || mkpath(SIEBC_DIR)

for subreddit in subreddits
    suffix = "logistic"

    types = ["observed", "logistic", "linear", "logistic_stateless"]
    result_df = DataFrame()
    result_df.topic = vcat([fill(topic, length(types)) for topic in topics]...)
    result_df.type = vcat([types for _ in topics]...)

    EMs = Dict()
    Ws = Dict()
    KSs = Dict()
    WAICs = Dict()
    hs = Dict()
    full_hs = Dict()
    Ns = Dict()

    for topic in topics
        EMs[topic] = Dict()
        Ws[topic] = Dict()
        KSs[topic] = Dict()
        hs[topic] = Dict()
        WAICs[topic] = Dict()
        EMs[topic]["observed"] = 0.
        Ws[topic]["observed"] = 0.
        KSs[topic]["observed"] = 0.
        WAICs[topic]["observed"] = 0.
    end

    sample_topics!(comments, submissions, suffix, subreddit, EMs, Ws, KSs, hs, WAICs, Ns, full_hs; optim_init=[.6,.6,.6])
    export_boxplot(full_hs, joinpath(SIEBC_DIR, "$(subreddit)_homophily.csv"))
    create_alpha_table(topics, suffix, joinpath(MCMC_DIR, subreddit), SIEBC_DIR)

    suffix="linear"
    sample_topics!(comments, submissions, suffix, subreddit, EMs, Ws, KSs, hs, WAICs, Ns; result_dir=missing)

    suffix="logistic_stateless"
    sample_topics!(comments, submissions, suffix, subreddit, EMs, Ws, KSs, hs, WAICs, Ns; result_dir=missing)

    result_df.nb_authors = [Ns[row.topic][1] for row in eachrow(result_df)]
    result_df.nb_comments = [Ns[row.topic][2] for row in eachrow(result_df)]
    result_df.W = [Ws[row.topic][row.type] for row in eachrow(result_df)]
    result_df.EM = [EMs[row.topic][row.type] for row in eachrow(result_df)]
    result_df.KS = [KSs[row.topic][row.type] for row in eachrow(result_df)]
    result_df.WAIC_mean = [mean(WAICs[row.topic][row.type]) for row in eachrow(result_df)]
    result_df.WAIC_std = [std(WAICs[row.topic][row.type]) for row in eachrow(result_df)]
    result_df.WAIC_q025 = [_ci_quantile(WAICs[row.topic][row.type], 0.025) for row in eachrow(result_df)]
    result_df.WAIC_q975 = [_ci_quantile(WAICs[row.topic][row.type], 0.975) for row in eachrow(result_df)]
    result_df.h_mean = [mean(hs[row.topic][row.type]) for row in eachrow(result_df)]
    result_df.h_std = [std(hs[row.topic][row.type]) for row in eachrow(result_df)]
    result_df.h_median = [median(hs[row.topic][row.type]) for row in eachrow(result_df)]
    result_df.h_q025 = [_ci_quantile(hs[row.topic][row.type], 0.025) for row in eachrow(result_df)]
    result_df.h_q975 = [_ci_quantile(hs[row.topic][row.type], 0.975) for row in eachrow(result_df)]
    result_df.dh = [row.h_mean - hs[row.topic]["observed"] for row in eachrow(result_df)]
    result_df.dh_q025 = [ismissing(row.h_q025) ? missing : row.h_q025  - hs[row.topic]["observed"] for row in eachrow(result_df)]
    result_df.dh_q975 = [ismissing(row.h_q975) ? missing : row.h_q975 - hs[row.topic]["observed"] for row in eachrow(result_df)]
    result_df.hq = [mean(hs[row.topic][row.type] .>= hs[row.topic]["observed"]) for row in eachrow(result_df)]
    CSV.write(joinpath(SIEBC_DIR, "$(subreddit)_measures.csv"), result_df)

    result_df_to_latex(result_df; filename=joinpath(SIEBC_DIR, "$(subreddit)_measures.tex"))
end
