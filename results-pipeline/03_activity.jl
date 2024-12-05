include(joinpath(dirname(@__FILE__), "..", "util.jl"))
comments, submissions = get_comments_and_submissions(; discard=Dict(:author=>[AUTHOR_DELETE, AUTHOR_AUTO]))

subreddits = ["belgium"]
topics = [lockdown, mask, vaccin]
path = joinpath(RESULT_DIR, "03_ecdf")
"""
    add_matching_ancestors!(comments::DataFrame, submissions::DataFrame; cols=[:topic,])

For each comment in the `comments` DataFrame, this function counts the number of
ancestors that have the same value for specified columns (`cols`).
The counts are stored in new columns in the `comments` DataFrame, named `nb_same_<col>_bf` for
each column in `cols`.

# Arguments
- `comments::DataFrame`: The DataFrame containing comment data.
- `submissions::DataFrame`: The DataFrame containing submission data.
- `cols::Vector{Symbol}`: A vector of column names to check for matching values. Default is `[:topic]`.

"""
function add_matching_ancestors!(comments::DataFrame, submissions::DataFrame; cols=[:topic,])
    sort!(comments, :datetime)
    get_parent = ParentLookup(comments, submissions)
    value_cols = ["nb_same_$(col)_bf" for col in cols]
    for col in value_cols
        comments[!, col] = zeros(Int64, nrow(comments))
    end

    @showprogress for outer_row in eachrow(comments)
        found_cols = fill(false, length(cols))
        row = outer_row
        while !all(found_cols)
            parent = get_parent(row)
            if ismissing(parent)
                if haskey(get_parent.submission_id_to_idx, row.submission_id)
                    for col in cols[.!found_cols]
                        value = submissions[get_parent.submission_id_to_idx[row.submission_id], col]
                        row[col] == value ?  row["nb_same_$(col)_bf"] += 1 : nothing
                    end
                end
                break
            end
            for (i, col) in enumerate(cols)
                if !found_cols[i] && row[col] == parent[col]
                    row["nb_same_$(col)_bf"] = is_submission(parent) ? 1 : parent["nb_same_$(col)_bf"] + 1
                    found_cols[i] = true
                end
            end
            is_submission(parent) ? break : row = parent
        end
    end
end

function add_root_column(comments, submissions)
    comments, submissions = deepcopy(comments), deepcopy(submissions)
    add_matching_ancestors!(comments, submissions; cols=[:topic, :author])

    comments.is_root = comments.nb_same_topic_bf .== 0
    is_first_comment_on_topic = row -> row.nb_same_topic_bf == 0 || row.nb_same_author_bf == 0
    comments = comments[is_first_comment_on_topic.(eachrow(comments)), :]
    submissions.is_root .= true
    return comments, submissions
end

function root_position(author, root_df)
    return findfirst(x -> x.is_root, eachrow(root_df[root_df.author .== author, :]))
end

"""
    Pmn(m, n)

Pmn(m,n)[k] is the probability distribution that when m A's and n B's are randomly arranged, the first B appears at position k
"""
function Pmn(m, n)
    mpn = BigInt(m+n)
    nm1 = BigInt(n-1)
    N = binomial(mpn, BigInt(n))
    P = k -> binomial(mpn-BigInt(k), nm1) / N
    return [P(k) for k in 1:m+1]
end

function get_random_adoption_data(root_df)
    random_data = [Pmn(row.nb_replies, row.nb_roots) for row in eachrow(root_df) if row.nb_roots > 0]
    max_domain = maximum(length.(random_data))
    random_data = [vcat(r, fill(0.0, max_domain-length(r))) for r in random_data]
    random_data = sum(random_data) ./ nrow(root_df)
    return 0:max_domain-1, cumsum(random_data)
end

function get_activity_ecdf(root_df)
    root_pos = root_df.root_position[.!ismissing.(root_df.root_position)]
    nb_befores = sort(unique(root_pos)) .- 1
    ecdf = [count(root_pos .<= i+1) for i in nb_befores] ./ nrow(root_df)
    return nb_befores, ecdf
end

function roots_per_user(comments, submissions; cols::Dict=Dict([:topic=>vaccin]), discard=[AUTHOR_AUTO, AUTHOR_DELETE])
    selector = df -> reduce(.&, (df[!, col] .== val for (col, val) in cols))
    root_df = sort(vcat(
        comments[selector(comments), [:author, :datetime, :is_root]],
        submissions[selector(submissions), [:author, :datetime, :is_root]]
    ), :datetime)
    root_df = root_df[.!ismissing.(root_df.is_root), :]

    df = get_users(comments[selector(comments),:], submissions[selector(submissions),:], Dict(:is_root=>[true, false]))
    df = df[.!ein(df.author, discard), :]
    df.nb_roots = df.nc_true .+ df.ns
    df.nb_replies = df.nc_false
    df.root_position = map(row -> row.nb_roots > 0 ? root_position(row.author, root_df) : missing, eachrow(df))
    return df[!, [:author, :nc, :ns, :nb_roots, :nb_replies, :root_position]]
end

isdir(path) || mkdir(path)
comments, submissions = add_root_column(comments, submissions)

for subreddit in subreddits
    for topic in topics
        data = roots_per_user(comments, submissions; cols=Dict([:topic=>topic]), discard=[AUTHOR_DELETE, AUTHOR_AUTO])
        x_ecdf, y_ecdf = get_activity_ecdf(data)
        x_ecdf .+= 1 # transfrorm from nb_before to position of first
        x_random, y_random = get_random_adoption_data(data)
        ml = min(length(x_ecdf), length(x_random))
        data = DataFrame(
            x = x_ecdf[1:ml],
            ecdf = y_ecdf[1:ml],
            random = y_random[1:ml]
        )
        CSV.write(joinpath(path, "$(subreddit)_$(topic).csv"), data)
    end
end

