import sqlite3
import os
import sys
import time
from urllib.error import HTTPError

# from utils.db_utils import *
from tqdm import tqdm
from time import sleep
from concurrent.futures import ProcessPoolExecutor

global THREADED, enabled_models, paths, SENTIMENT_COLUMNS, perspective_langs, bert_multi_langs

THREADED = False # Recommended if perspective is used
enabled_models = {
    "transformers": True,
    "pattern": False, # If True, transformers must be true as well to infer the language
    "vader": False,
    "perspective": False,
}

paths = {
    "input":  "data/04_belgium.db",
    "target": "data/05_belgium.db",
}

SENTIMENT_COLUMNS = []
if enabled_models["transformers"]:
    SENTIMENT_COLUMNS.extend(["language", "bert_negative", "bert_neutral", "bert_positive"])
    SENTIMENT_COLUMNS.extend(["bert_multi_1", "bert_multi_2", "bert_multi_3", "bert_multi_4", "bert_multi_5"])
if enabled_models["pattern"]:
    SENTIMENT_COLUMNS.extend(["polarity", "subjectivity"])
if enabled_models["vader"]:
    SENTIMENT_COLUMNS.extend(["vader_negative", "vader_neutral", "vader_positive", "vader_compound"])
if enabled_models["perspective"]:
    SENTIMENT_COLUMNS.extend(["toxicity"])

perspective_langs = {'de', 'es', 'fr', 'hi', 'nl', 'ja', 'id', 'sv', 'ru', 'it', 'en', 'pt', 'ko', 'cs', 'zh', 'pl', 'ar'}
bert_multi_langs = {"en", "fr", "nl", "de", "es", "it"}

def timed_request(request, max_wait_time=10, verbose=False):
    """
    Make a request to reddit. If the request fails because of too many requests,
    wait for a certain amount of time and try again.
    """
    def wait(i):
        t = 2**i
        if verbose:
            print("Too many requests, waiting {} seconds".format(t))
        time.sleep(t)

    for i in range(max_wait_time):
        try:
            return request()
        except HTTPError as e:
            if e.code == 429 or e.code // 100 == 5:
                wait(i)
            else:
                raise e

    return request()

def define_transformer_models(models):
    from transformers import pipeline
    # language detection
    lang_pipe = pipeline(
        "text-classification", model="papluca/xlm-roberta-base-language-detection",
    )
    models["lang"] = lambda s: lang_pipe(s)[0]["label"]

    # BERT
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

    bert_multi_pipe = pipeline(
        "text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment", device=0,
    )
    def bert_multi(string):
        full_result = bert_multi_pipe(string, top_k=None)
        results = [None, None, None, None, None]
        for item in full_result:
            try:
                results[int(item["label"][0]) - 1] = item["score"]
            except:
                pass
        return results
    models["bert_multi"] = bert_multi

def define_pattern_models(models):
    from pattern.en import sentiment as p_en_sentiment
    from pattern.fr import sentiment as p_fr_sentiment
    from pattern.nl import sentiment as p_nl_sentiment

    def pattern_uwrapper(func, string):
        full_result = func(string)
        return {
            "polarity": full_result[0],
            "subjectivity": full_result[1],
        }

    models["pattern"] = {
        "en": lambda string: pattern_uwrapper(p_en_sentiment, string),
        "fr": lambda string: pattern_uwrapper(p_fr_sentiment, string),
        "nl": lambda string: pattern_uwrapper(p_nl_sentiment, string),
    }

def define_vader_models(models):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_en_analyzer = SentimentIntensityAnalyzer()
    def vader_en(string):
        full_result = vader_en_analyzer.polarity_scores(string)
        return {
            "positive": full_result["pos"],
            "negative": full_result["neg"],
            "neutral": full_result["neu"],
            "compound": full_result["compound"],
        }
    models["vader"] = {"en": vader_en}

def define_perspectiveAPI_models(models):
    from perspective import PerspectiveAPI
    perspective_pipe = PerspectiveAPI(os.getenv("PERSPECTIVE_API_KEY"))
    def toxicity(string):
        sleep(0.75)
        try:
            full_result = timed_request(lambda: perspective_pipe.score(string))
        except HTTPError as e:
            if e.code == 400:
                return float('nan')
            else:
                raise e
        return full_result["TOXICITY"]
    models["toxicity"] = toxicity

def init_and_wrap_models():
    """
    Returns a dictionary of models, with the following structure:
    {
        "lang": <language detection model>,
        "bert": { # Provided by HuggingFace BERT models
            "en": <english sentiment model>,
            "fr": <french sentiment model>,
            "nl": <dutch sentiment model>,
        },
        "vader": { # Provided by VADER
            "en": <english sentiment model>,
        },
        "pattern": { # Provided by pattern
            "en": <english sentiment model>,
            "fr": <french sentiment model>,
            "nl": <dutch sentiment model>,
        },
        "toxicity": <toxicity model>, # Provided by Perspective API
        "bert_multi": <multilingual bert sentiment model>,
    """
    models = {}

    if enabled_models["transformers"]:
        define_transformer_models(models)
    else:
        models["lang"] = lambda s: None
        models["bert"] = dict()
        models["bert_multi"] = lambda s: None

    if enabled_models["pattern"]:
        define_pattern_models(models)
    else:
        models["pattern"] = dict()

    if enabled_models["vader"]:
        define_vader_models(models)
    else:
        models["vader"] = dict()

    if enabled_models["perspective"]:
        define_perspectiveAPI_models(models)
    else:
        models["toxicity"] = lambda s: None

    return models

def get_bert_scores(string, model, results, prefix=""):
    scores = model(string)
    results[prefix+"bert_negative"] = scores["negative"]
    results[prefix+"bert_neutral"] = scores["neutral"]
    results[prefix+"bert_positive"] = scores["positive"]

def get_vader_scores(string, model, results, prefix=""):
    scores = model(string)
    results[prefix+"vader_negative"] = scores["negative"]
    results[prefix+"vader_neutral"] = scores["neutral"]
    results[prefix+"vader_positive"] = scores["positive"]
    results[prefix+"vader_compound"] = scores["compound"]

def get_pattern_scores(string, model, results, prefix=""):
    scores = model(string)
    results[prefix+"polarity"] = scores["polarity"]
    results[prefix+"subjectivity"] = scores["subjectivity"]

def get_toxicity_score(string, model, results, prefix=""):
    results[prefix+"toxicity"] = model(string)

def get_bert_multi_scores(string, model, results, prefix=""):
    scores = model(string)
    results[prefix+"bert_multi_1"] = scores[0]
    results[prefix+"bert_multi_2"] = scores[1]
    results[prefix+"bert_multi_3"] = scores[2]
    results[prefix+"bert_multi_4"] = scores[3]
    results[prefix+"bert_multi_5"] = scores[4]

def get_sentiments(item, string, models, prefix="", threaded = False):
    """
    Returns a dictionary of sentiments, if which the structure can be seen directly below.
    The `models` parameter is the dictionary returned by `init_and_wrap_models()`.
    """
    if string is None or len(string.split()) < 4:
        return
    if len(string) >= 512:
        string = string[:512]

    lang = models["lang"](string)
    item[prefix+"language"] = lang

    # Create a ProcessPoolExecutor
    if threaded:
        with ProcessPoolExecutor() as executor:
            # Run each model in a separate thread
            if lang in models["bert"]:
                executor.submit(get_bert_scores, string, models["bert"][lang], item, prefix)
            if lang in models["vader"]:
                executor.submit(get_vader_scores, string, models["vader"][lang], item, prefix)
            if lang in models["pattern"]:
                executor.submit(get_pattern_scores, string, models["pattern"][lang], item, prefix)
            if lang in perspective_langs:
                executor.submit(get_toxicity_score, string, models["toxicity"], item, prefix)
            if lang in ("en", "fr", "nl", "de", "es", "it"):
                executor.submit(get_bert_multi_scores, string, models["bert_multi"], item, prefix)

    else:
        if lang in models["bert"]:
            get_bert_scores(string, models["bert"][lang], item, prefix)
        if lang in models["vader"]:
            get_vader_scores(string, models["vader"][lang], item, prefix)
        if lang in models["pattern"]:
            get_pattern_scores(string, models["pattern"][lang], item, prefix)
        if lang in perspective_langs:
            get_toxicity_score(string, models["toxicity"], item, prefix)
        if lang in bert_multi_langs:
            get_bert_multi_scores(string, models["bert_multi"], item, prefix)

    return

def get_type(column):
    if column in ("id",):
        return "TEXT PRIMARY KEY"
    if column in ("idx",):
        return "INTEGER PRIMARY KEY"
    elif column in ("created_utc", "score", "num_comments", "controversiality", "topic_idx"):
        return "INTEGER"
    elif (
        "bert" in column
        or "vader" in column
        or "polarity" in column
        or "subjectivity" in column
        or "toxicity" in column
        or "upvote_ratio" in column
        or "prob" in column
    ):
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
    Prepares the database for storing selected data; i.e. check and if necessary create tables.
      * a sqlite database at `database_location`
      * the tables `submission` with columuns
        * `id` (TEXT, PRIMARY KEY)
        * `subreddit` (TEXT)
        * `author` (TEXT)
        * `author_flair_text` (TEXT)
        * `author_flair_richtext` (TEXT)
        * `title` (TEXT)
        * `selftext` (TEXT)
        * `created_utc` (INTEGER)
        * `score` (INTEGER)
        * `url` (TEXT)
        * `num_comments` (INTEGER)
        * `upvote_ratio` (REAL)
        where % is both `title` and `selftext`
      * the table `comment` with columns
        * `id` (TEXT, PRIMARY KEY)
        * `parent_id` (TEXT)
        * `submission_id` (TEXT)
        * `subreddit` (TEXT)
        * `author` (TEXT)
        * `author_flair_text` (TEXT)
        * `author_flair_richtext` (TEXT)
        * `created_utc` (INTEGER)
        * `score` (INTEGER)
        * `body` (TEXT)
        * `controversiality` (INTEGER)
    As well as the columns for the sentiment analysis, based on th input column (see `SENTIMENT_COLUMNS`)
    """

    conn = sqlite3.connect(database_location)
    c = conn.cursor()
    add_table(c, "submission", submission_cols)
    add_table(c, "comment", comment_cols)
    conn.commit()
    conn.close()

def prepare_item(item, data, models, columns, threaded=False):
    for i in range(len(data)):
        item[columns[i]] = data[i]
    for i in range(len(data), len(columns)):
        item[columns[i]] = None

    get_sentiments(item, item["processed"], models, threaded=threaded)

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

def from_database(input_path, target_path, models, commit_every=1000, threaded=False):
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
                prepare_item(submission, submission_tuple, models, submission_columns, threaded=threaded)
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
                prepare_item(comment, comment_tuple, models, comment_columns, threaded)
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
    from_database(paths["input"], paths["target"], models, threaded=THREADED)
    return

if __name__ == "__main__":
    main_from_database()
