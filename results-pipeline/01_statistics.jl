using PowerLaws

include(joinpath(dirname(@__FILE__), "..", "util.jl"))
comments, submissions = get_comments_and_submissions(; discard=Dict(:author=>[AUTHOR_AUTO]))

topics = [lockdown, mask, vaccin]
langs = ["en", "nl", "fr"]
categories = Dict([:topic=>topics,:language=>langs])

function get_all_cols(categories; prefixes=["ns", "nc"])
    cols = []
    for (_, options) in categories
        for option in options
            for prefix in prefixes
                push!(cols, Symbol("$(prefix)_$option"))
            end
        end
    end
    return cols
end

"""
    activity_per_category!(users, comments, submissions, column::Symbol; options=[], user_dict=Dict())

Calculate and update activity metrics per category for users based on comments and submissions.

# Arguments
- `users`: A DataFrame containing user data.
- `comments`: A DataFrame containing comment data.
- `submissions`: A DataFrame containing submission data.
- `column`: A Symbol representing the column name to categorize by.
- `options`: An optional array of categories to consider. Defaults to the union of categories found in `comments` and `submissions`.
- `user_dict`: An optional dictionary mapping authors to user indices. Defaults to a dictionary created from `users.author`.

# Updates
- Adds/updates columns in `users` DataFrame for:
  - Number of comments per category (`nc_<column>_<option>`)
  - Number of submissions per category (`ns_<column>_<option>`)
  - Total number of activities per category (`nt_<column>_<option>`)
  - Weighted activity per category (`w_<column>_<option>`), normalized by total activities.
"""
function activity_per_category!(users, comments, submissions, column::Symbol; options=[], user_dict=Dict())
    options = length(options) > 0 ? options : Set(comments[!, column]) ∪ Set(submissions[!, column])
    user_dict = length(user_dict) > 0 ? user_dict : Dict([author => i for (i, author) in enumerate(users.author)])
    missing_allowed = any(ismissing.(options))

    for option in options
        users[!, "nc_$option"] .= 0
        users[!, "ns_$option"] .= 0
    end

    for (df, type) in [(comments, "nc"), (submissions, "ns")]
        for row in eachrow(df)
            row_option = row[column]
            if (ismissing(row_option) && !missing_allowed) || (!ismissing(row_option) && !(row_option in options))
                continue
            end
            users[user_dict[row.author], "$(type)_$row_option"] += 1
        end
    end

    for option in options
        users[!, "nt_$option"] = users[!, "nc_$option"] .+ users[!, "ns_$option"]
        users[!, "w_$option"] = users[!, "nt_$option"] ./ users.total
    end
end

"""
    get_users(comments::DataFrame, submissions::DataFrame) -> DataFrame

Aggregate user activity from two DataFrames: `comments` and `submissions`.

# Arguments
- `comments::DataFrame`: A DataFrame containing user comments. Must include columns `author` and `id`.
- `submissions::DataFrame`: A DataFrame containing user submissions. Must include columns `author` and `id`.

# Returns
- `DataFrame`: A DataFrame with the aggregated number of comments and submissions per author. 
  Includes columns `author`, `num_comments`, and `num_submissions`, and a calculated `total` column 
  representing the sum of comments and submissions for each author.
"""
function get_users(comments, submissions)
    users = [comments, submissions] |>
        x -> (groupby(df, :author) for df in x) |>
        x -> (combine(g, :id => length, renamecols=false) for g in x) |>
        x -> (DataFrames.rename(df, "id" => "n") for df in x) |>
        x -> outerjoin(x..., on=:author, renamecols="c" => "s") |>
        x -> coalesce.(x, 0)
    users.total = users.nc .+ users.ns
    return users
end

function get_users(comments, submissions, categories::Dict)
    users = get_users(comments, submissions)
    user_dict = Dict([author => i for (i, author) in enumerate(users.author)])
    for (category, options) in categories
        activity_per_category!(users, comments, submissions, category; options=options, user_dict=user_dict)
    end
    return users
end

users = get_users(comments, submissions, categories)

open(joinpath(RESULT_DIR, "01_degree_data", "number_table.tex"), "w") do io
    println(io, "\\begin{tabular}{r|ccc}")
    println(io, "  & Users & Comments & Submissions \\\\\\hline")
    println(io, "  \\textbf{Total} & $(length(users.author)-1) & $(sum(users.nc)) & $(sum(users.ns)) \\\\\\hline")
    println(io, "  \\textbf{Per language} & & & \\\\")
    for lang in langs
        lang_users = users[users[!,"nt_$(lang)"] .> 0, :]
        lang_coms = sum(lang_users[!, "nc_$lang"])
        lang_subs = sum(lang_users[!, "ns_$lang"])
        println(io, "  $lang & $(nrow(lang_users)-1) & $(lang_coms) & $(lang_subs) \\\\")
    end
    println(io, "  \\textbf{Per topic} & & & \\\\")
    for topic in topics
        topic_users = users[users[!,"nt_$(topic)"] .> 0, :]
        topic_coms = sum(topic_users[!, "nc_$topic"])
        topic_subs = sum(topic_users[!, "ns_$topic"])
        println(io, "  $topic & $(nrow(topic_users)-1) & $(topic_coms) & $(topic_subs) \\\\")
    end
    println(io, "\\end{tabular}")
end

# Without deleted
users = users[users.author .!= AUTHOR_DELETE, :]

fit = DataFrame(topic=String[], alpha=Float64[], minx=Float64[], KS=Float64[])

for topic in topics
    col = "nt_$(topic)"
    ds = sort(setdiff(unique(users[!, col]), [0]))
    counts = [count(x->x==d, users[!, col]) for d in ds]
    df = DataFrame(:ds=>ds, :counts=>counts)
    params, KS = estimate_parameters(users[users[!, col] .> 0, col], DiscretePowerLaw, xmins=1:10)
    α = params.α
    A = df.counts[10] / (df.ds[10]^(-1. *α))
    df.fit = A*(df.ds.^(-1. * α))
    CSV.write(joinpath(RESULT_DIR, "01_degree_data", "$(topic).csv"), df)
    push!(fit, (string(topic), α, params.θ, KS))
end

CSV.write(joinpath(RESULT_DIR, "01_degree_data", "fit_info.csv"), fit)

