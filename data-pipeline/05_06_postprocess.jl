include(joinpath(@__DIR__, "..", "util.jl"))

input_db = SQLite.DB("data/05_belgium.db")
output_db = SQLite.DB("data/06_belgium.db")

comments = DataFrame(DBInterface.execute(input_db, "SELECT * FROM comment"))
submissions = DataFrame(DBInterface.execute(input_db, "SELECT * FROM submission"))

global topic_cols, sentiment_models
topic_cols = ["vaccine", "masks", "lockdown", "schools", "quarantine", "closinghoreca", "testing", "curfew", "othermeasure", "notapplicable"]
sentiment_models = ["bert"]

function to_topic(str::String)::Topic
    t = findfirst(topic -> occursin(string(topic), str), instances(Topic))
    return isnothing(t) ? notapplicable : Topic(t)
end

to_topic(sym::Symbol)::Topic = to_topic(string(sym))
to_topic(t::Topic) = t

Base.Float64(s::AbstractString) = parse(Float64, s)
Base.Float64(::Missing) = missing

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
    df.topic = to_topic.(df.topic)
    map(col -> df[!,col] .= Float64.(df[!,col]), topic_cols)

    discard_short_results!(df)

    ("bert_negative" in names(df)) && (df.bert = -1. * df.bert_negative .+ df.bert_positive)
    ("bert_multi_1" in names(df)) && (df.bert_multi = -1. * df.bert_multi_1 .-.5*df.bert_multi_2 .+ .5*df.bert_multi_4 .+ df.bert_multi_5)
    ("vader_compound" in names(df)) && rename!(df, :vader_compound => :vader)
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
    to_be_replaced = row -> ismissing(row.topic) || row.topic in (notapplicable, other)
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

handle_df!(comments)
handle_df!(submissions)

add_depth!(comments)
cascade_topics!(comments, submissions)

SQLite.load!(comments, output_db, "comment")
SQLite.load!(submissions, output_db, "submission")

function get_number(db_loc, table)
    db = SQLite.DB(joinpath("data", db_loc))
    return DataFrame(DBInterface.execute(db, "SELECT COUNT(*) FROM $table"))[1,1]
end

db_df = DataFrame(:db => [f for f in readdir("data") if endswith(f, ".db")])
db_df.n_subs = get_number.(db_df.db, "submission")
db_df.n_coms = get_number.(db_df.db, "comment")
db_df.total = db_df.n_subs .+ db_df.n_coms