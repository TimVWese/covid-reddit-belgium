using SQLite
using DataFrames
using NPZ

DTAI_TOPICS = ["vaccine", "masks", "lockdown", "schools", "quarantine", "closinghoreca", "testing", "curfew", "othermeasure", "notapplicable"]

topic_path = "data/03_topics/"
input_db = SQLite.DB("data/02_belgium.db")
output_db = SQLite.DB("data/04_belgium.db")

comments = DataFrame(DBInterface.execute(input_db, "SELECT * FROM comment"))
submissions = DataFrame(DBInterface.execute(input_db, "SELECT * FROM submission"))

function unpack_npz(file, ids, columns)
    data = Float64.(npzread(file))
    df = DataFrame(data, columns)
    df.id = ids
    return df
end

function get_topic(row, columns)
    scores = row[columns][:]
    return argmax(scores)
end

function handle(df, columns, file)
    topics = unpack_npz(file, df.id, columns)
    df = innerjoin(df, topics, on=:id)
    df.topic = [get_topic(row, columns) for row in eachrow(df)]
    return df
end

function store(output_db, comments, submissions)
    DBInterface.execute(output_db, "DROP TABLE IF EXISTS comment")
    DBInterface.execute(output_db, "DROP TABLE IF EXISTS submission")

    SQLite.load!(comments, output_db, "comment")
    SQLite.load!(submissions, output_db, "submission")
end

################################################
# Combine the results of the hugging face models
################################################
file = joinpath(topic_path, "submission_final_mbert.npy")
sub_wt = handle(submissions, DTAI_TOPICS, file)

file = joinpath(topic_path, "comment_final_mbert.npy")
com_wt = handle(comments, DTAI_TOPICS, file)


##########################
# Filter for covid topics
##########################
using Statistics

disallowed = (:notapplicable,)
comment_threshold = 0.9

allowed_sumbmissions = Set([
    row.id for row in eachrow(sub_wt) if !(row.topic in disallowed)
])

prop_na = x -> count(in(disallowed), x) / length(x)
com_prop = combine(groupby(com_wt, :submission_id), :topic=>prop_na)
com_prop = com_prop[.!ismissing.(com_prop.submission_id), :]
union!(allowed_sumbmissions, Set([
    row.submission_id for row in eachrow(com_prop) if row.topic_function <= comment_threshold
]))

submission_selection = sub_wt[[row.id in allowed_sumbmissions for row in eachrow(sub_wt)], :]
comment_selection = com_wt[[row.submission_id in allowed_sumbmissions for row in eachrow(com_wt)], :]
@info "Nb of posts kept: $(nrow(submission_selection) + nrow(comment_selection))"

store(output_db, comment_selection, submission_selection)
