import sqlite3
import sys

from tqdm import tqdm

paths = {
    "input":  "data/04_belgium.db",
    "target": "data/05_belgium.db",
}

# Columns added by the sentiment step. `language` is the detected language;
# the bert_* scores are filled only for languages with a sentiment model.
SENTIMENT_COLUMNS = ["language", "bert_negative", "bert_neutral", "bert_positive"]

def define_transformer_models(models):
    from transformers import pipeline
    # language detection
    lang_pipe = pipeline(
        "text-classification", model="papluca/xlm-roberta-base-language-detection",
    )
    models["lang"] = lambda s: lang_pipe(s)[0]["label"]

    # Per-language BERT sentiment. en yields negative/neutral/positive;
    # fr/nl yield positive/negative only (neutral left as None).
    models["bert"] = {}
    bert_en_pipe = pipeline(
        "text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=0
    )
    def bert_en(string):
        full_result = bert_en_pipe(string, top_k=None)
        return {item["label"]: item["score"] for item in full_result}
    models["bert"]["en"] = bert_en

    bert_fr_pipe = pipeline("text-classification", model="philschmid/pt-tblard-tf-allocine")
    def bert_fr(string):
        full_result = bert_fr_pipe(string, top_k=None)
        result_dict = {item["label"].lower(): item["score"] for item in full_result}
        result_dict["neutral"] = None
        return result_dict
    models["bert"]["fr"] = bert_fr

    bert_nl_pipe = pipeline(
        "text-classification", model="DTAI-KULeuven/robbert-v2-dutch-sentiment", device=0,
    )
    def bert_nl(string):
        full_result = bert_nl_pipe(string, top_k=None)
        result_dict = {item["label"].lower(): item["score"] for item in full_result}
        result_dict["neutral"] = None
        return result_dict
    models["bert"]["nl"] = bert_nl

def init_and_wrap_models():
    """
    Return a dictionary of models with the structure:
    {
        "lang": <language detection model>,
        "bert": {  # HuggingFace per-language sentiment models
            "en": <english sentiment model>,
            "fr": <french sentiment model>,
            "nl": <dutch sentiment model>,
        },
    }
    """
    models = {}
    define_transformer_models(models)
    return models

def get_bert_scores(string, model, results, prefix=""):
    scores = model(string)
    results[prefix+"bert_negative"] = scores["negative"]
    results[prefix+"bert_neutral"] = scores["neutral"]
    results[prefix+"bert_positive"] = scores["positive"]

def get_sentiments(item, string, models, prefix=""):
    """
    Detect the language of `string` and, when a matching BERT model exists,
    fill the sentiment scores on `item` in place. Skips very short texts.
    """
    if string is None or len(string.split()) < 4:
        return
    if len(string) >= 512:
        string = string[:512]

    lang = models["lang"](string)
    item[prefix+"language"] = lang

    if lang in models["bert"]:
        get_bert_scores(string, models["bert"][lang], item, prefix)

def get_type(column):
    if column in ("id",):
        return "TEXT PRIMARY KEY"
    if column in ("idx",):
        return "INTEGER PRIMARY KEY"
    elif column in ("created_utc", "score", "num_comments", "controversiality", "topic_idx"):
        return "INTEGER"
    elif "bert" in column or "upvote_ratio" in column or "prob" in column:
        return "REAL"
    return "TEXT"

def add_table(cursor, table, base_columns):
    columns = list(base_columns) + SENTIMENT_COLUMNS

    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            {", ".join([f"{column} {get_type(column)}" for column in columns])}
        )
    """
    )
    return

def prepare_database(database_location, submission_cols, comment_cols):
    """
    Create the target database and the `submission`/`comment` tables, mirroring
    the input columns and adding the columns in `SENTIMENT_COLUMNS`.
    """

    conn = sqlite3.connect(database_location)
    c = conn.cursor()
    add_table(c, "submission", submission_cols)
    add_table(c, "comment", comment_cols)
    conn.commit()
    conn.close()

def prepare_item(item, data, models, columns):
    for i in range(len(data)):
        item[columns[i]] = data[i]
    for i in range(len(data), len(columns)):
        item[columns[i]] = None

    get_sentiments(item, item["processed"], models)

def get_connections_cursors(input_path, target_path=None):
    input_conn = sqlite3.connect(input_path)
    input_cursor = input_conn.cursor()
    target_conn = input_conn if target_path is None else sqlite3.connect(target_path)
    target_cursor = target_conn.cursor()
    return input_conn, input_cursor, target_conn, target_cursor

def get_subreddits(input_cursor):
    input_cursor.execute("SELECT DISTINCT subreddit FROM submission")
    subreddits = input_cursor.fetchall()
    input_cursor.execute("SELECT DISTINCT subreddit FROM comment")
    subreddits.extend(input_cursor.fetchall())
    return set([subreddit[0] for subreddit in subreddits])

def get_total_count(input_cursor):
    input_cursor.execute('SELECT COUNT(*) FROM submission')
    total_count = input_cursor.fetchone()[0]
    input_cursor.execute('SELECT COUNT(*) FROM comment')
    total_count += input_cursor.fetchone()[0]
    return total_count

def get_filtered_items(input_cursor, target_cursor, subreddit, table, columns):
    target_cursor.execute(f"SELECT id FROM {table}")
    handled_ids = set([item[0] for item in target_cursor.fetchall()])

    query = f"""
        SELECT {", ".join(columns)}
        FROM {table} WHERE subreddit = ?
    """
    input_cursor.execute(query, (subreddit, ))
    return [item for item in input_cursor.fetchall() if item[0] not in handled_ids], len(handled_ids)

def get_insert_string(table, base_columns):
    """
    Generate an insert string for inserting into a table, based on the `columns`.
    """
    columns = list(base_columns) + SENTIMENT_COLUMNS

    insert_string = "INSERT INTO " + table + " ("
    values_string = "VALUES ("
    for key in columns:
        insert_string += key + ", "
        values_string += ":" + key + ", "

    insert_string = insert_string[:-2] + ")"
    values_string = values_string[:-2] + ")"
    return insert_string + " " + values_string, columns

def get_column_names(cursor, table_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns_info = cursor.fetchall()
    return [col[1] for col in columns_info]

def from_database(input_path, target_path, models, commit_every=1000):
    # Init connections, get subreddit list, and get the counts for the progress bar
    _, input_cursor, target_conn, target_cursor = get_connections_cursors(input_path, target_path)
    subreddits = get_subreddits(input_cursor)
    outer_progress_bar = tqdm(total=get_total_count(input_cursor))

    for sub_idx, subreddit in enumerate(subreddits):
        outer_progress_bar.set_description(f"{sub_idx}/{len(subreddits)}")

        # Process submissions
        existing_submission_cols = get_column_names(input_cursor, "submission")
        execution_string, submission_columns = get_insert_string("submission", existing_submission_cols)
        submission = {}
        count = 0

        # Get the submission ids that are already in the target database
        filtered_submissions, num_handled = get_filtered_items(input_cursor, target_cursor, subreddit, "submission", existing_submission_cols)
        outer_progress_bar.update(num_handled)

        for submission_tuple in tqdm(filtered_submissions, desc=f"Submissions {subreddit}"):
            # Commit and update status if necessary
            count += 1
            if count % commit_every == 0:
                target_conn.commit()
                outer_progress_bar.update(commit_every)

            # Prepare and execute
            try:
                prepare_item(submission, submission_tuple, models, submission_columns)
                target_cursor.execute(execution_string, submission)
            except Exception as e:
                sys.stderr.write(str(e) + "\n\n while processing: " + str(submission_tuple[0]) + "\nin submission (db)")

        # Finalise submissions
        target_conn.commit()
        outer_progress_bar.update(count%commit_every)
        outer_progress_bar.set_description(f"{sub_idx}.5/{len(subreddits)}")

        # Process comments
        existing_comment_cols = get_column_names(input_cursor, "comment")
        execution_string, comment_columns = get_insert_string("comment", existing_comment_cols)
        comment = {}
        count = 0
        # Get the comment ids that are already in the target database
        filtered_comments, num_handled = get_filtered_items(input_cursor, target_cursor, subreddit, "comment", existing_comment_cols)
        outer_progress_bar.update(num_handled)

        for comment_tuple in tqdm(filtered_comments, desc=f"Comments {subreddit}"):
            # Commit and update status if necessary
            count += 1
            if count % commit_every == 0:
                target_conn.commit()
                outer_progress_bar.update(commit_every)

            # Prepare and execute
            try:
                prepare_item(comment, comment_tuple, models, comment_columns)
                target_cursor.execute(execution_string, comment)
            except Exception as e:
                sys.stderr.write(str(e) + "\n\n while processing: " + str(comment_tuple[0]) + "\nin comment (db)")

        # Finalise comments
        target_conn.commit()
        outer_progress_bar.update(count%commit_every)

    return

def main_from_database():
    models = init_and_wrap_models()
    conn = sqlite3.connect(paths["input"])
    c = conn.cursor()
    prepare_database(paths["target"], get_column_names(c, "submission"), get_column_names(c, "comment"))
    conn.close()
    from_database(paths["input"], paths["target"], models)
    return

if __name__ == "__main__":
    main_from_database()
