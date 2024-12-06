using JLD2
using Turing
using SciMLSensitivity, ReverseDiff
using Interpolations
using Printf
using HypothesisTests

include(joinpath(@__DIR__, "..", "util.jl"))
# Allow to interpret some more dates to improve fitting at boundaries
broad_date_range=Date(2019,11,1):Date(2022,8,31)
date_range = Date(2020, 1, 1):Date(2022, 6, 30)
comments, submissions = get_comments_and_submissions(; discard=Dict(:author=>[AUTHOR_AUTO, AUTHOR_DELETE]), date_range=broad_date_range)
topics = [vaccin, mask, lockdown]
MCMC_PATH = joinpath(@__DIR__, "..", "mcmc-chains")

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
function get_author_comments(comments, submissions, topic; threshold=50)
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

function handle(comments, submissions, topic; type=:logistic, threshold=50, N_samples_per=500,
    N_parallel_in=6, N_parallel_out=5, save_suffix="", mcmc_path=MCMC_PATH)
    a_coms, _ = get_author_comments(comments, submissions, topic; threshold=threshold)
    save_suffix = save_suffix * "_" * string(type)
    save_dir = joinpath(mcmc_path, "$(topic)$(save_suffix)/")
    !isdir(save_dir) && mkdir(save_dir)

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

for topic in topics
    handle(comments, submissions, topic; type=:logistic)
    handle(comments, submissions, topic; type=:linear)
end
