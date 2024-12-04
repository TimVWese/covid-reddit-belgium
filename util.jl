using CSV
using DataFrames
using Dates
using ProgressMeter
using Statistics
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
