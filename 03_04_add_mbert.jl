using SQLite
using DataFrames
using NPZ

DTAI_TOPICS = ["vaccine", "masks", "lockdown", "schools", "quarantine", "closinghoreca", "testing", "curfew", "othermeasure", "notapplicable"]

topic_path = "03_topics/"
input_db = SQLite.DB("02_reddit_belgium.db")
output_db = SQLite.DB("04_belgium_topics.db")

comments = DataFrame(DBInterface.execute(input_db, "SELECT * FROM comment"))
submissions = DataFrame(DBInterface.execute(input_db, "SELECT * FROM submission"))

function unpack_npz(file, ids, columns)
    data = npzread(file)
    df = DataFrame(data, columns)
    df.id = ids
    return df
end

function get_topic(row, columns)
    scores = row[columns][:]
    idx = argmax(scores)
    return columns[idx]
end

function handle(df, columns, file)
    topics = unpack_npz(file, df.id, columns)
    innerjoin!(df, topics, on=:id)
    df.add_topic = get_topic.(eachrow(df))
    return df
end

function store(output_path, comments, submissions)
    target = SQLite.DB(output_path)
    DBInterface.execute(target, "DROP TABLE IF EXISTS comment")
    DBInterface.execute(target, "DROP TABLE IF EXISTS submission")

    SQLite.load!(comments, target, "comment")
    SQLite.load!(submissions, target, "submission")
end

function get_random_topic(df, models=("mbert", "deberta", "bart"))
    result = Dict()
    row = rand(1:size(df, 1))
    result["processed"] = df.processed[row]
    for model in models
        labels = []
        scores = []
        for col in names(df)
            if occursin(model, col)
                push!(labels, col)
                push!(scores, df[row, col])
            end
        end
        idx = argmax(scores)
        result[model] = (labels[idx], scores[idx])
    end
    return result
end


################################################
# Combine the results of the hugging face models
################################################
file = joinpath(topic_path, "submission_final_mbert.npy")
submissions = handle(submissions, DTAI_TOPICS, file)

file = joinpath(topic_path, "comment_final_mbert.npy")
comments = handle(comments, DTAI_TOPICS, file)


##########################
# Filter for covid topics
##########################
using Statistics

submission_threshold = 0.1
comment_threshold = 0.2


allowed_sumbmissions = Set([
    row.id for row in eachrow(submissions) if row.mbert_notapplicable < submission_threshold
])

comments = comments[[length(split(row.processed, " ")) > 3 for row in eachrow(comments)], :]
using Statistics
comment_mean = combine(groupby(comments, :submission_id), :mbert_notapplicable => mean)
comment_mean = comment_mean[ismissing.(comment_mean.submission_id).==false, :]
union!(allowed_sumbmissions, Set([
    row.submission_id for row in eachrow(comment_mean) if row.mbert_notapplicable_mean < comment_threshold
]))

submission_selection = submissions[[row.id in allowed_sumbmissions for row in eachrow(submissions)], :]
comment_selection = comments[[row.submission_id in allowed_sumbmissions for row in eachrow(comments)], :]

output_path = topic_path * "belgium_covid_very_strict.db"
store(output_path, comment_selection, submission_selection)


#####################
# Check some topics
#####################

get_random_topic(submissions)
get_random_topic(comments)

get_random_topic(comment_selection)
get_random_topic(submission_selection)
