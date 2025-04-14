include(joinpath(dirname(@__FILE__), "..", "util.jl"))
comments, submissions = get_comments_and_submissions(; discard=Dict(:author=>[AUTHOR_AUTO,]))

subreddits = ["belgium"]

path = joinpath(RESULT_DIR, "04_sentiment")
contexts =["gen", "prev_parent"]
topics = ["lockdown"=>x->x.topic==lockdown, "mask"=>x->x.topic==mask, "vaccin"=>x->x.topic==vaccin, ]
signals = ["bert"=>x->select_lang(x, "en")]
n = 5

"""
    aggregate_mean_values(comments::AbstractDataFrame, cols)

Aggregate the mean of values in `cols` for each column in `comments`.
"""
function aggregate_mean_values(comments::AbstractDataFrame, cols)
    result = DataFrame([col => [0.,] for col in cols]...)
    counts = Dict([col => 0 for col in cols])
    for comment in eachrow(comments)
        for col in cols
            if !ismissing(comment[col])
                result[1, col] += comment[col]
                counts[col] += 1
            end
        end
    end
    foreach(col -> result[1, col] /= counts[col], cols)
    return result[1, :]
end

"""
    get_ancestors(comment::DataFrameRow, parent_lookup::ParentLookup, generations::Int)

Get the ancestors (parents of parents) of `comment` up to `generations` back.
"""
function get_ancestors(comment::DataFrameRow, parent_lookup::ParentLookup, generations::Int, cols)
    if !ismissing(comment.depth) && comment.depth < generations
        return missing
    end
    ancestors = Matrix{Union{Float64,Missing}}(undef, generations, length(cols))
    current = comment
    for i in 1:generations
        parent = parent_lookup(current)
        if ismissing(parent) || (is_submission(parent) && i < generations)
            return missing
        end
        foreach(j -> ancestors[i,j] = parent[cols[j]], eachindex(cols))
        current = parent
    end
    return DataFrame(ancestors, cols)
end

"""
    add_ancestor_means!(to_process, submissions, generations; all_comments=nothing, cols=BASE_SENTIMENT_COLUMNS)

Add the mean of the values in `cols` of the `genrations` ancestors of `to_process`.
"""
function add_ancestor_means!(to_process, submissions, generations; all_comments=nothing, cols=BASE_SENTIMENT_COLUMNS)
    all_comments = isnothing(all_comments) ? to_process : all_comments
    get_parent = ParentLookup(all_comments, submissions)
    if !("depth" in names(to_process))
        add_depth!(to_process)
    end
    for col in cols
        to_process[!, "gen_$(generations)_$col"] = Array{Union{Float64,Missing}}(missing, size(to_process, 1))
    end
    pb = Progress(size(to_process, 1), 1)

    Threads.@threads for row in eachrow(to_process)
        ancestors = get_ancestors(row, get_parent, generations, cols)
        ancestors_mean = !ismissing(ancestors) ? aggregate_mean_values(ancestors, cols) : continue
        foreach(col -> row["gen_$(generations)_$col"] = ancestors_mean[col], cols)
        next!(pb)
    end
end

"""
    get_previous_comment(row::DataFrameRow, comments::DataFrame)

Get the previous comment of the author of `row` in `comments`.
"""
function get_previous_comments_parents(row::DataFrameRow, comments::DataFrame, parent_lookup::ParentLookup, cols, nb=1)
    author = row.author
    datetime = row.datetime
    previous = comments[comments.author .== author, :] |>
        x -> sort(x[x.datetime .<= datetime, :], :datetime) |>
        x -> size(x, 1) < nb ? missing : x[end-nb+1:end, :]
    if ismissing(previous)
        return missing, missing
    else
        parents = Matrix{Union{Float64,Missing}}(undef, size(previous, 1), length(cols))
        for (i, comment) in enumerate(eachrow(previous))
            parent = parent_lookup(comment)
            if ismissing(parent)
                return missing, missing
            end
            foreach(j -> parents[i,j] = parent[cols[j]], eachindex(cols))
        end
        return previous, DataFrame(parents, cols)
    end
end

"""
    add_previous_means!(to_process, submissions, number; all_comments=nothing, cols=BASE_SENTIMENT_COLUMNS)

Add the mean of the values in `cols` of the previous `number` comments of the author of `to_process`.
"""
function add_previous_means!(to_process, submissions, number; all_comments=nothing, cols=BASE_SENTIMENT_COLUMNS)
    all_comments = isnothing(all_comments) ? to_process : all_comments
    get_parent = ParentLookup(all_comments, submissions)
    for col in cols
        to_process[!, "prev_$(number)_$col"] = Array{Union{Float64,Missing}}(missing, size(to_process, 1))
        to_process[!, "prev_parent_$(number)_$col"] = Array{Union{Float64,Missing}}(missing, size(to_process, 1))
    end
    pb = Progress(size(to_process, 1), 1)

    Threads.@threads for row in eachrow(to_process)
        previous_comments, previous_parents = get_previous_comments_parents(row, to_process, get_parent, cols, number)
        previous_means = !ismissing(previous_comments) ? aggregate_mean_values(previous_comments, cols) : missing
        if !ismissing(previous_means)
            foreach(col -> row["prev_$(number)_$col"] = previous_means[col], cols)
        end
        prev_parent_means = !ismissing(previous_parents) ? aggregate_mean_values(previous_parents, cols) : missing
        if !ismissing(prev_parent_means)
            foreach(col -> row["prev_parent_$(number)_$col"] = prev_parent_means[col], cols)
        end
        next!(pb)
    end
end

function add_both_contexts!(comments, submissions, context_size; all_comments=comments, cols=["bert",])
    for i in 1:context_size
        add_ancestor_means!(comments, submissions, i; all_comments=all_comments, cols=cols)
        add_previous_means!(comments, submissions, i; all_comments=all_comments, cols=cols)
    end
end

function get_differences(comments, submissions, context_size; p_val=0.05, os=50, signal="bert", contexts=contexts)
    submissions = submissions[ein(submissions.id, comments.submission_id), :]
    random_data = vcat(comments[!,signal], submissions[!,signal])

    sents = []
    diffs = []
    for context in contexts
        for i in 1:context_size
            col2 = "$(context)_$(i)_" * signal
            D, diff = hist_and_diff(comments[!,signal], comments[!,col2], random_data; i, p_val, os)
            push!(sents, D)
            push!(diffs, diff)
        end
    end

    sz = (context_size, length(contexts))
    return reshape(sents, sz), reshape(diffs, sz)
end

function generate_heatmaps(comments, submissions, topics, signals)
    result = Dict()
    for (n_topic, s_topic) in topics
        result[n_topic] = Dict()
        for (n_model, s_model) in signals
            lang_comments = comments[s_model.(eachrow(comments)), :]
            cb = lang_comments[s_topic.(eachrow(lang_comments)), :]
            sb = submissions[ein(submissions.id, cb.submission_id), :]
            add_parent_values!(cb, sb; all_comments=lang_comments, cols=[n_model,])
            cb = remove_invalid_rows(cb, [n_model, "parent_$(n_model)"])
            sb = remove_invalid_rows(sb, [n_model])
            random_data = vcat(cb[!,n_model], sb[!,n_model])
            sent, diff = hist_and_diff(cb[!,n_model], cb[!, "parent_$(n_model)"], random_data; p_val=0.05)
            result[n_topic][n_model] = Dict()
            result[n_topic][n_model][:sent] = sent
            result[n_topic][n_model][:diff] = diff
            result[n_topic][n_model][:h] = diagonalness(diff)
            result[n_topic][n_model][:N_users] = size(unique(cb.author), 1)
            result[n_topic][n_model][:N_comments] = size(cb, 1)
        end
    end
    return result
end

function handle_contexts(comments, submissions, topics, signals, n)
    result = Dict()
    for (n_topic, s_topic) in topics
        result[n_topic] = Dict()
        for (n_model, s_model) in signals
            lang_comments = comments[s_model.(eachrow(comments)), :]
            cb = comments[[s_topic(row) && s_model(row) for row in eachrow(comments)], :]
            sb = submissions[ein(submissions.id, cb.submission_id), :]
            add_both_contexts!(cb, sb, n; all_comments=lang_comments, cols=[n_model,])
            col1 = "gen_$(n)_$(n_model)"
            col2 = "prev_parent_$(n)_$(n_model)"
            cb_s = remove_invalid_rows(cb, [col1, col2])
            N = [size(cb_s, 1), size(unique(cb_s.author),1)]
            @info "Evaluating $(N[1]) comments of $(N[2]) authors for $(n_topic)"
            _, diffs = get_differences(cb_s, sb, n; signal=n_model, contexts=["gen", "prev_parent"])
            diags = DataFrame("x"=>1:n)
            diags.gen = diagonalness.(diffs[:, 1])
            diags.prev_parent = diagonalness.(diffs[:, 2])
            result[n_topic][n_model] = Dict()
            result[n_topic][n_model][:diags] = diags
            result[n_topic][n_model][:N_comments] = N[1]
            result[n_topic][n_model][:N_users] = N[2]
        end
    end
    return result
end

isdir(path) || mkdir(path)

for subreddit in subreddits
    comments_sub = comments[comments.subreddit .== subreddit, :]
    submissions_sub = submissions[ein(submissions.id, comments_sub.submission_id), :]

    other_info = DataFrame()
    other_info.topic = vcat([fill(String(topic[1]), length(signals)) for topic in topics]...)
    other_info.model = vcat(fill([signal[1] for signal in signals], length(topics))...)
    pair_to_row = Dict((other_info.topic[r] => other_info.model[r]) => r for r in 1:nrow(other_info))

    heatmaps = generate_heatmaps(comments_sub, submissions_sub, topics, signals)
    Ns_users_full = Array{Int64, 1}(undef, nrow(other_info))
    Ns_comments_full = Array{Int64, 1}(undef, nrow(other_info))
    hs_full = Array{Float64, 1}(undef, nrow(other_info))
    for (topic, models) in heatmaps
        for (model, data) in models
            export_as_index_list(data[:sent], joinpath(path, "$(subreddit)_$(topic)_$(model)_sent.csv"))
            export_as_index_list(data[:diff], joinpath(path, "$(subreddit)_$(topic)_$(model)_diff.csv"))
            Ns_users_full[pair_to_row[topic=>model]] = data[:N_users]
            Ns_comments_full[pair_to_row[topic=>model]] = data[:N_comments]
            hs_full[pair_to_row[topic=>model]] = data[:h]
        end
    end
    other_info.N_users_full = Ns_users_full
    other_info.N_comments_full = Ns_comments_full
    other_info.h_full = hs_full

    result = handle_contexts(comments_sub, submissions_sub, topics, signals, n)
    N_users_context = Array{Int64, 1}(undef, nrow(other_info))
    N_comments_context = Array{Int64, 1}(undef, nrow(other_info))
    hs_context = Array{Float64, 1}(undef, nrow(other_info))
    for (topic, models) in result
        for (model, data) in models
            CSV.write(joinpath(path, "$(subreddit)_$(topic)_$(model)_context.csv"), data[:diags])
            N_users_context[pair_to_row[topic=>model]] = data[:N_users]
            N_comments_context[pair_to_row[topic=>model]] = data[:N_comments]
            hs_context[pair_to_row[topic=>model]] = data[:diags][1,2]
        end
    end
    other_info.N_users_context = N_users_context
    other_info.N_comments_context = N_comments_context
    other_info.h_context = hs_context

    CSV.write(joinpath(path, "$(subreddit)_other_info.csv"), other_info)
end

