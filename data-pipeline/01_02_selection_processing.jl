using SQLite
using DataFrames
using DelimitedFiles
using Dates

input_db = SQLite.DB("data/01_belgium.db")
output_db = SQLite.DB("data/02_belgium.db")

function get_submission_ids(database, terms)
    submissions = DataFrame(DBInterface.execute(database, "SELECT * FROM submission"))
    covid_submissions_id = Set()
    for term in terms
        covid_submissions_id = covid_submissions_id ∪ Set(submissions[findall(x -> occursin(term, lowercase(x)), submissions.title), :id])
    end

    for term in terms
        covid_submissions_id = covid_submissions_id ∪ Set(submissions[findall(x -> occursin(term, lowercase(x)), submissions.selftext), :id])
    end

    for term in terms
        query = "SELECT DISTINCT submission_id
                 FROM comment
                 WHERE submission_id IS NOT NULL AND
                       submission_id != '' AND
                       LOWER(body) LIKE '%' || LOWER('$term') || '%';"
        covid_submissions_id = covid_submissions_id ∪ Set(DataFrame(DBInterface.execute(database, query)).submission_id)
    end

    return covid_submissions_id
end

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

covid_terms = ["corona", "virus", "covid", "mask", "masque", "lockdown", "confin", "quarant", "curfew", "avondklok", "couvre-feu", "couvre feu", "vaccin", "vax", "jab", "booster", "prik", "piqûre", "piqure", "pcr", "plf", "locator form", "cst", "safe ticket"]
# Month extra padding to deal with rolling averages later
start_date = Int64(datetime2unix(DateTime(2019, 12, 1)))
end_date = Int64(datetime2unix(DateTime(2023, 1, 31)))

covid_submissions_id = get_submission_ids(input_db, covid_terms)
covid_submission_id_str = "'" * join(covid_submissions_id, "','") * "'"
covid_submissions = DataFrame(DBInterface.execute(input_db, "
    SELECT *
    FROM submission
    WHERE id IN ($covid_submission_id_str)
        AND created_utc >= $start_date
        AND created_utc < $end_date
    "
))

covid_comments = DataFrame(DBInterface.execute(input_db, "
    SELECT *
    FROM comment
    WHERE submission_id IN ($covid_submission_id_str)"
))
println("Number of posts: ", nrow(covid_submissions) + nrow(covid_comments))

covid_submissions.processed = [clean_markdown(covid_submissions[i,:title]*"\n\n"*covid_submissions[i,:selftext]) for i in 1:nrow(covid_submissions)]
covid_comments.processed = clean_markdown.(covid_comments.body)

DBInterface.execute(output_db, "DROP TABLE IF EXISTS submission")
SQLite.load!(covid_submissions, output_db, "submission")
DBInterface.execute(output_db, "DROP TABLE IF EXISTS comment")
SQLite.load!(covid_comments, output_db, "comment")
