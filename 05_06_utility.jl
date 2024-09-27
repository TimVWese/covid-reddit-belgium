using SQLite
using DataFrames
using Dates
using ProgressMeter

input_db = SQLite.DB("05_belgium_top_sent.db")
output_db = SQLite.DB("06_belgium_covid.db")

comments = DataFrame(DBInterface.execute(input_db, "SELECT * FROM comment"))
submissions = DataFrame(DBInterface.execute(input_db, "SELECT * FROM submission"))

global topic_cols, sentiment_models
topic_cols = ["vaccine", "masks", "lockdown", "schools", "quarantine", "closinghoreca", "testing", "curfew", "othermeasure", "notapplicable"]
sentiment_models = ["bert", "vader"]

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

function str2topic(str::String)::Topic
    t = findfirst(topic -> occursin(string(topic), str), instances(Topic))
    return isnothing(t) ? notapplicable : Topic(t)
end

function str2topic(sym::Symbol)::Topic
    return str2topic(string(sym))
end

str2float(x::String) = parse(Float64, x)
str2float(x::Missing) = missing

function discard_short_results!(df; threshold=3)
    sentiment_cols = [n for n in names(df) if any(m->occursin(m, n), sentiment_models)]
    allowmissing!(df, topic_cols)
    allowmissing!(df, sentiment_cols)
    for row in eachrow(df)
        if length(split(row.processed, " ")) <= threshold || row.processed in ("[deleted]", "[removed]")
            row.topic = notapplicable
            map(x -> row[x] = missing, topic_cols)
            map(x -> row[x] = missing, sentiment_cols)
        end
    end
end

function handle_df!(df::DataFrame)
    if "created_utc" in names(df)
        df.datetime = unix2datetime.(df.created_utc)
        df.date = Date.(df.datetime)
        select!(df, Not(:created_utc))
    end

    df.subreddit = lowercase.(df.subreddit)
    df.topic = str2topic.(df.topic)
    map(col -> df[!,col] .= str2float.(df[!,col]), topic_cols)

    discard_short_results!(df)

    df.bert = -1. * df.bert_negative .+ df.bert_positive
    df.bert_multi = -1. * df.bert_multi_1 .-.5*df.bert_multi_2 .+ .5*df.bert_multi_4 .+ df.bert_multi_5
    DataFrames.rename!(df, :vader_compound => :vader)
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

"""
    add_depth!(comments::DataFrame)

Append the depth of the comment to the `comments` DataFrame.
"""
function add_depth!(comments::DataFrame)
    submissions = DataFrame(:id => []) # placeholder for get parent
    comments.depth = Array{Union{Int64, Missing}}(missing, size(comments, 1))
    get_parent = ParentLookup(comments, submissions)
    @showprogress for row in eachrow(comments)
        if row.parent_id[1:3] == "t3_"
            row.depth = 1
        else
            parent = get_parent(row)
            row.depth = ismissing(parent) ? missing : parent.depth + 1
        end
    end
end

"""
    cascade_topics!(comments::DataFrame, submissions::DataFrame, model="mbert", threshold=0.5)

Update the topics in `comments`, such that if the topic is not applicable, or the score is below `threshold`,
the topic is inferred from the parent comment or submission. returns the number of updated topics.
"""
function cascade_topics!(comments::DataFrame, submissions::DataFrame)
    to_be_replaced = row -> row.topic in (notapplicable, other)
    updates = 0
    get_parent = ParentLookup(comments, submissions)

    @showprogress for row in eachrow(comments)
        if to_be_replaced(row)
            parent = get_parent(row)
            if !ismissing(parent) && row.topic != parent.topic
                row.topic = parent.topic
                updates += 1
            end
        end
    end

    return updates
end

df = submissions
if "created_utc" in names(df)
    df.datetime = unix2datetime.(df.created_utc)
    df.date = Date.(df.datetime)
    select!(df, Not(:created_utc))
end

handle_df!(comments)
handle_df!(submissions)

add_depth!(comments)

cascade_topics!(comments, submissions)

SQLite.load!(comments, output_db, "comment")
SQLite.load!(submissions, output_db, "submission")
