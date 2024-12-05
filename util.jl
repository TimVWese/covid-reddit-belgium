using CSV
using DataFrames
using Dates
using JLD2
using ProgressMeter
using Statistics
using StatsBase
using Distributions
using SQLite

global DATA_DIR, RESULT_DIR
const DATA_DB = joinpath(@__DIR__, "data", "06_belgium.db")
const RESULT_DIR = joinpath(@__DIR__, "results")

global AUTHOR_AUTO, AUTHOR_DELETE
const AUTHOR_AUTO = "AutoModerator"
const AUTHOR_DELETE = "[deleted]"

global DATE_RANGE
DATE_RANGE = Date(2020, 1, 1):Day(1):Date(2022, 12, 31)

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
