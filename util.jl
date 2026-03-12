using CSV
using DataFrames
using Dates
using JLD2
using ProgressMeter
using Statistics
using StatsBase
using Distributions
using SQLite
using Printf

global DATA_DIR, RESULT_DIR
const DATA_DB = joinpath(@__DIR__, "data", "06_belgium.db")
const RESULT_DIR = joinpath(@__DIR__, "results")

global AUTHOR_AUTO, AUTHOR_DELETE
const AUTHOR_AUTO = "AutoModerator"
const AUTHOR_DELETE = "[deleted]"

global DATE_RANGE
DATE_RANGE = Date(2020, 1, 1):Day(1):Date(2022, 6, 30)

@enum Topic begin
    vaccin = 1
    mask = 2
    lockdown = 3
    school = 4
    quarantine = 5
    horeca = 6
    test = 7
    curfew = 8
    other = 9
    notapplicable = 10
end

const TOPIC_TITLES = Dict(
    lockdown => "Lockdowns",
    mask => "Masks",
    vaccin => "Vaccination",
)

"""
    ein(data, set)

Return a boolean array of the same length as `data`, where `true` indicates that the element is in `set`.
"""
function ein(data, set)
    return [x in set for x in data]
end

function select_lang(df, lang)
    return .!ismissing.(df.language) .&& df.language .== lang
end

"""
    handle_reddit_db(path, table)

read the data of `table` in sqlite db `db` to a `DataFrame`
"""
function handle_reddit_db(db, table; discard::Dict=Dict())
    df = DataFrame(DBInterface.execute(db, "SELECT * FROM $table"))
    for (col, vals) in discard
        df = df[.!ein(df[!, col], vals), :]
    end
    return df
end

"""
    get_comments_and_submissions(path)

Get the comments and submissions from the reddit database at `path`.
"""
function get_comments_and_submissions(path=DATA_DB; discard::Dict=Dict(), date_range=DATE_RANGE)
    db = SQLite.DB(path)
    comments = handle_reddit_db(db, "comment"; discard=discard)
    submissions = handle_reddit_db(db, "submission"; discard=discard)

    if !isnothing(date_range)
        submissions = submissions[ein(submissions.date, date_range), :]
        comments = comments[ein(comments.submission_id, submissions.id), :]
    end
    return comments, submissions
end

"""
    remove_invalid_rows(df::DataFrame, cols)

Remove rows from `df` where any of the columns in `cols` is `missing`, `nothing`, or `NaN`.
"""
function remove_invalid_rows(df::DataFrame, cols)
    return filter(row -> all(col ->
        row[col] !== nothing && row[col] !== missing &&
        !(row[col] isa AbstractFloat && isnan(row[col])),
    cols), df)
end

struct ParentLookup
    comments::DataFrame
    submissions::DataFrame
    comment_id_to_idx::Dict{String, Int}
    submission_id_to_idx::Dict{String, Int}
end

function ParentLookup(comments::DataFrame, submissions::DataFrame)
    comment_id_to_idx = Dict(id => i for (i, id) in enumerate(comments.id))
    submission_id_to_idx = Dict(id => i for (i, id) in enumerate(submissions.id))
    return ParentLookup(comments, submissions, comment_id_to_idx, submission_id_to_idx)
end

function (lookup::ParentLookup)(row::DataFrameRow; full=true)
    parent_id = row.parent_id
    if ismissing(parent_id) || length(parent_id) <= 3
        return missing
    end
    is_comment = parent_id[1:3] == "t1_"
    id = parent_id[4:end]
    df = is_comment ? lookup.comments : lookup.submissions
    id_to_idx = is_comment ? lookup.comment_id_to_idx : lookup.submission_id_to_idx
    if !haskey(id_to_idx, id)
        return missing
    end
    return full ? df[id_to_idx[id], :] : id_to_idx[id]
end

is_comment = row::DataFrameRow -> "body" in names(row)
is_submission = row::DataFrameRow -> "title" in names(row)

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

"""
    per_day(df::DataFrame, ops::Pair...)

Aggregate per day, aggregating the desired columns by the `ops`. 
"""
function per_day(df::DataFrame, ops::Pair...)
    pd = groupby(df, :date) |>
          group -> combine(group, :id => length, ops..., renamecols=false) |>
                   df -> sort(df, :date)
    DataFrames.rename!(pd, :id => :num)
    result = DataFrame(:date=>minimum(pd.date):Day(1):maximum(pd.date))
    leftjoin!(result, pd, on=:date)
    return coalesce.(result, 0)
end

"""
    add_parent_value(comments, submissions; all_comments=nothing, cols=BASE_SENTIMENT_COLUMNS)

Add the value in `cols` of the parent comment or submission to the comment row.
"""
function add_parent_values!(to_process, submissions; all_comments=nothing, cols=BASE_SENTIMENT_COLUMNS)
    all_comments = isnothing(all_comments) ? to_process : all_comments
    get_parent = ParentLookup(all_comments, submissions)
    for col in cols
        to_process[!, "parent_$col"] = Array{Union{eltype(to_process[!, col]),Missing}}(missing, size(to_process, 1))
    end

    Threads.@threads for row in eachrow(to_process)
        parent = get_parent(row)
        if ismissing(parent)
            continue
        end
        foreach(col -> row["parent_$col"] = parent[col], cols)
    end
end

"""
    mean_sampling(data, n; multiplier=1)

Sample `n` elements from `data` and calculate the mean. Repeat `multiplier`*length(data) times.
"""
function mean_sampling(data, n; nb_samples=length(data))
    data = data[.!ismissing.(data) .&& .! isnan.(data)]
    return [mean(sample(data, n; replace=true)) for _ in 1:nb_samples]
end

function histogram_weights(data; width=0.05)
    hist = fit(Histogram, data, (-1-width):width:1, closed=:right)
    ws = hist.weights[2:end]
    ws[1] += hist.weights[1] # merge last two bins
    return ws ./ (width*sum(ws))
end

function get_unconditional_histogram(d1, d2; width=0.05)
    valid = .!ismissing.(d1 + d2) .&& .!isnan.(d1 + d2)
    d1,d2 = Float64.(d1[valid]), Float64.(d2[valid])

    r1 = histogram_weights(d1; width)
    r2 = histogram_weights(d2; width)
    R = r1*r2'

    return R
end

function get_structured_histogram(d1, d2; width=0.05)
    valid = .!ismissing.(d1 + d2) .&& .!isnan.(d1 + d2)
    d1,d2 = Float64.(d1[valid]), Float64.(d2[valid])
    bins = (-1. - width):width:1
    HD = fit(Histogram, (d1,d2), (bins, bins), closed=:right)
    D = HD.weights[2:end, 2:end]  # remove the extra bin on the left and bottom
    D[1,:] .+= HD.weights[1, 1:end-1]  # add left extra bin to first column
    D[:,1] .+= HD.weights[1:end-1, 1]  # add bottom extra bin to first row
    D[1,1] += HD.weights[1,1]  # add bottom-left extra bin to (1,1)
    return D / (width^2 * sum(D))
end

function get_2d_diff(D, R_f; N=50)
    Rs = [R_f() for _ in 1:N]
    R = mean(Rs)

    P = Matrix{Float64}(undef, size(D))
    for idx in eachindex(D)
        op = D[idx] > R[idx] ? (<) : (>)
        c = count(Rc -> op(D[idx], Rc[idx]), Rs)
        P[idx] = c/N
    end

    return D - R, P
end

function get_2d_diff(D, R_f, p_val; os=5)
    N = os*ceil(Int64, 1. / p_val)
    diff, P = get_2d_diff(D, R_f; N)
    diff[P .>= p_val] .= 0.
    return diff
end

"""
    hist_and_diff(base_data, observed_data, random_data; p_val=nothing, i=1, os=5)

Calculate the structured histogram and the difference between the observed and random data.
"""
function hist_and_diff(base_data, observed_data, random_data; width=0.05, p_val=nothing, i=1, os=5)
    rd_func = (n) -> mean_sampling(random_data, i; nb_samples=n)
    Rd_func = () -> get_unconditional_histogram(base_data, rd_func(size(base_data, 1)); width)
    D = get_structured_histogram(base_data, observed_data; width)
    diff =  isnothing(p_val) ? get_2d_diff(D, Rd_func; N=20*os)[1] : get_2d_diff(D, Rd_func, p_val; os)
    return D, diff
end

function diagonalness(D::Matrix{Float64})
    step = 2. / size(D, 1)
    xs = (-1 + step/2):step:(1 - step/2)
    ys = (-1 + step/2):step:(1 - step/2)
    @assert length(xs) == size(D, 1)
    @assert length(ys) == size(D, 2)
    f = (x, y) -> 1 - 2 * abs(x - y)
    return sum(D[i, j] * (step^2) * f(xs[i], ys[j]) for i in axes(D, 1) for j in axes(D, 2))
end

"""
    export_as_index_list(D, filename)

Export a 2D matrix as a list of indices and values.
"""
function export_as_index_list(D, filename)
    step = 2. / size(D, 1)
    xs = (-1 + step/2):step:(1 - step/2)
    ys = (-1 + step/2):step:(1 - step/2)
    @assert length(xs) == size(D, 1)
    @assert length(ys) == size(D, 2)
    df = DataFrame(x = [xs[i] for i in axes(D, 1) for j in axes(D, 2)],
                   y = [ys[j] for i in axes(D, 1) for j in axes(D, 2)],
                   z = [D[i,j] for i in axes(D, 1) for j in axes(D, 2)])
    CSV.write(filename, df)
end

"""
    fmt(x, ndigits) -> String

Format a number with `ndigits` decimals, then strip trailing zeros and
a trailing decimal point if needed, to get clean output like:
0.00125, 0.24472, -0.00631, 1.002, 1
"""
function fmt_digit(x, ndigits::Int)
    s = @sprintf("%.*f", ndigits, x)
    s = replace(s, r"0+$" => "")         # drop trailing zeros
    s = replace(s, r"\.$"  => "")        # drop dangling decimal point
    return s
end
