using SQLite
using DataFrames
using DelimitedFiles

#! First step would be to add submission id!!!

input_db = SQLite.DB("01_b1b2.db")
output_db = SQLite.DB("02_reddit_belgium.db")
scratch_dir = "03_topics/"

covid_terms = ["corona", "virus", "covid", "mask", "masque", "lockdown", "confinement", "quarantin", "curfew", "couvre-feu", "avondklok", "vaccin", "vax", "pcr", "plf", "locator form", "cst", "safe ticket"]

submissions = DataFrame(DBInterface.execute(input_db, "SELECT * FROM submission"))
covid_submissions_id = Set()

for term in covid_terms
    covid_submissions_id = covid_submissions_id ∪ Set(submissions[findall(x -> occursin(term, lowercase(x)), submissions.title), :id])
end
length(covid_submissions_id)

for term in covid_terms
    covid_submissions_id = covid_submissions_id ∪ Set(submissions[findall(x -> occursin(term, lowercase(x)), submissions.selftext), :id])
end
length(covid_submissions_id)

for term in covid_terms
    query = "SELECT DISTINCT submission_id FROM comment WHERE submission_id IS NOT NULL AND submission_id != '' AND LOWER(body) LIKE '%' || LOWER('$term') || '%';"
    covid_submissions_id = covid_submissions_id ∪ Set(DataFrame(DBInterface.execute(input_db, query)).submission_id)
end
length(covid_submissions_id)

covid_submission_id_str = "'" * join(covid_submissions_id, "','") * "'"
covid_submissions = DataFrame(DBInterface.execute(input_db, "SELECT * FROM submission WHERE id IN ($covid_submission_id_str)"))
covid_comments = DataFrame(DBInterface.execute(input_db, "SELECT * FROM comment WHERE submission_id IN ($covid_submission_id_str)"))

function clean_markdown(input_string::String)
    # Remove Markdown links
    cleaned_string = replace(input_string, r"\[(.*?)\]\(.*?\)" => s"\1")
    # Remove other Markdown syntax (e.g., bold, italics)
    cleaned_string = replace(cleaned_string, r"\*\*(.*?)\*\*" => s"\1")  # Bold
    cleaned_string = replace(cleaned_string, r"\*(.*?)\*" => s"\1")      # Italics
    cleaned_string = replace(cleaned_string, r"#+\s*(.*?)" => s"\1")     # Headers
    cleaned_string = replace(cleaned_string, r"https?://[^\s]+|www\.[^\s]+|ftp://[^\s]+" => "") # other urls
    cleaned_string = replace(cleaned_string, r"&gt.*\n"=>"") # quotes
    return cleaned_string
end

covid_submissions.processed = [clean_markdown(covid_submissions[i,:title]*"\n\n"*covid_submissions[i,:selftext]) for i in 1:nrow(covid_submissions)]
covid_comments.processed = clean_markdown.(covid_comments.body)

DBInterface.execute(output_db, "DROP TABLE IF EXISTS submission")
SQLite.load!(covid_submissions, output_db, "submission")
DBInterface.execute(output_db, "DROP TABLE IF EXISTS comment")
SQLite.load!(covid_comments, output_db, "comment")

repl_nl = x -> replace(x, "\n"=>"\\n")

writedlm(joinpath(scratch_dir, "submission.txt"), repl_nl.(covid_submissions.processed), '\n')
writedlm(joinpath(scratch_dir, "comment.txt"), repl_nl.(covid_comments.processed), '\n')
