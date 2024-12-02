# The script in this file is meant to read reddit submissions and comments from
# a .zst file in json format and insert them into a sqlite database.

import sqlite3
from datetime import datetime
import json
from tqdm import tqdm
import os
import zstandard

global input_dir, database_location, SUBMISSION_COLUMNS, COMMENT_COLUMNS

input_dir = "00_zsts"
database_location = "01_belgium.db"

SUBMISSION_COLUMNS = (
    "id",
    "subreddit",
    "author",
    "author_flair_text",
    "author_flair_richtext",
    "title",
    "selftext",
    "created_utc",
    "score",
    "url",
    "num_comments",
    "upvote_ratio",
)

COMMENT_COLUMNS = (
    "id",
    "parent_id",
    "subreddit",
    "author",
    "author_flair_text",
    "author_flair_richtext",
    "created_utc",
    "score",
    "body",
    "controversiality",
)

def read_and_decode(
    reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0
):
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size
    if previous_chunk is not None:
        chunk = previous_chunk + chunk
    try:
        return chunk.decode()
    except UnicodeDecodeError:
        if bytes_read > max_window_size:
            raise UnicodeError(
                f"Unable to decode frame after reading {bytes_read:,} bytes"
            )
        # print(f"Decoding error with {bytes_read:,} bytes, reading another chunk")
        return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)


def read_lines_zst(file_name):
    with open(file_name, "rb") as file_handle:
        buffer = ""
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(
            file_handle
        )
        while True:
            chunk = read_and_decode(reader, 2**27, (2**29) * 2)

            if not chunk:
                break
            lines = (buffer + chunk).split("\n")

            for line in lines[:-1]:
                yield line, file_handle.tell()

            buffer = lines[-1]

        reader.close()

def zst_linewise(line_op, file_path):
    """
    Read a .zst file line by line and apply line_op to each line.
    """
    bad_lines = 0
    file_size = os.stat(file_path).st_size
    pb = tqdm(total=file_size, unit="B", unit_scale=True, desc=file_path.split("/")[-1], leave=False)

    try:
        for line, b in read_lines_zst(file_path):
            pb.update(b-pb.n)
            try:
                line_op(line)

            except (KeyError, json.JSONDecodeError) as err:
                bad_lines += 1

        pb.close()
    except Exception as err:
        pb.close()
        raise err

    return bad_lines

def get_execution_string(table, columns):
    """
    Generate an execution string for inserting into a table, based on the `columns`.
    """
    insert_string = "INSERT INTO " + table + " ("
    values_string = "VALUES ("
    for key in columns:
        insert_string += key + ", "
        values_string += ":" + key + ", "

    insert_string = insert_string[:-2] + ")"
    values_string = values_string[:-2] + ")"
    return insert_string + " " + values_string

def prepare_item(item, columns):
    for key in columns:
        if key not in item:
            item[key] = None
        if type(item[key]) in (list, dict):
            item[key] = json.dumps(item[key])

def prepare_database(database_location):
    """
    Prepare the database for the reddit data; i.e. check and if necessary create
      * a sqlite database at database_location
      * the table `submissions` in the database, with columns see below
      * the table `comments` in the database, with columns see below

    """
    conn = sqlite3.connect(database_location)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS submission (
            id text PRIMARY KEY,
            subreddit text,
            title text,
            created_utc integer,
            author text,
            score integer,
            num_comments integer,
            selftext text,
            url text,
            author_flair_text text,
            author_flair_richtext text,
            upvote_ratio real
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS comment (
            id text PRIMARY KEY,
            parent_id text,
            subreddit text,
            subreddit_id text,
            score integer,
            body text,
            created_utc integer,
            author text,
            controversiality integer,
            author_flair_text text,
            author_flair_richtext text
        )
    """)

    conn.commit()
    conn.close()

    return

def handle_submissions(file_path, cursor):
    execute_string = get_execution_string("submission", SUBMISSION_COLUMNS)
    def handle_submission(line):
        submission = json.loads(line)
        try:
            prepare_item(submission, SUBMISSION_COLUMNS)
            cursor.execute(execute_string, submission)
        except Exception as err:
            cursor.connection.rollback()
            raise err

    zst_linewise(handle_submission, file_path)
    cursor.connection.commit()
    return True

def handle_comments(file_path, cursor):
    execute_string = get_execution_string("comment", COMMENT_COLUMNS)
    duplicates = 0
    def handle_comment(line):
        nonlocal duplicates
        comment = json.loads(line)
        try:
            prepare_item(comment, COMMENT_COLUMNS)
            cursor.execute(execute_string, comment)
        except Exception as err:
            if "UNIQUE constraint failed" in str(err):
                duplicates += 1
            else:
                cursor.connection.rollback()
                raise err

    zst_linewise(handle_comment, file_path)
    cursor.connection.commit()
    print(f"Found {duplicates} duplicates in {file_path}")
    return

def append_submission_id(database_location):
    """
    Iterate over all rows in the `comment` table of the database at `database_location` and
    find and add the `submission_id` column to it, refereing to the post in the `submission` table.
    """
    conn = sqlite3.connect(database_location)
    cursor = conn.cursor()

    # Check if a submission id column exists
    cursor.execute("SELECT COUNT(*) FROM pragma_table_info('comment') WHERE name='submission_id';")
    if cursor.fetchone()[0] == 0:
        cursor.execute("ALTER TABLE comment ADD COLUMN submission_id TEXT;")

    cursor.execute("SELECT id FROM comment WHERE submission_id IS NOT NULL AND submission_id != '';")
    found_ids = set([id[0] for id in cursor.fetchall()])

    cursor.execute("SELECT id FROM comment WHERE submission_id IS NULL OR submission_id = '' ORDER BY created_utc DESC")
    comment_ids = cursor.fetchall()

    count = 0
    for current_id in tqdm(comment_ids):
        current_id = current_id[0]
        if current_id in found_ids:
            continue
        path = [current_id, ]
        submission_id=""
        while True:
            cursor.execute("SELECT parent_id FROM comment WHERE id = ?", (path[-1],))
            parent_id = cursor.fetchone()
            if parent_id is None or parent_id[0] is None: # cul-de-sac; we'll never know
                submission_id = None
                break
            prefix, parent_id = parent_id[0].split("_")
            if prefix == "t3": # parent is a submission
                submission_id = parent_id
                break
            if parent_id in found_ids: # we know the submission id of the parent
                cursor.execute("SELECT submission_id FROM comment WHERE id = ?", (parent_id,))
                try:
                    submission_id = cursor.fetchone()[0]
                except:
                    submission_id = None
                break
            path.append(parent_id)

        for id in path:
            cursor.execute("UPDATE comment SET submission_id = ? WHERE id = ?", (submission_id, id))
            found_ids.add(id)
            count += 1

        if count > 1000:
            conn.commit()
            count = 0

    conn.commit()
    return

def main_subreddit():
    """
    Build database from the subreddit.zst files in the input directory. I.e. check
    if the file name contains "submission" or "comment".
    """
    prepare_database(database_location)
    conn = sqlite3.connect(database_location)
    c = conn.cursor()

    handled_file = os.path.join(input_dir, "db_handled.dat")
    handled = (
        {line.strip() for line in open(handled_file, "r")}
        if os.path.exists(handled_file)
        else set()
    )

    for file in tqdm(os.listdir(input_dir)):
        if file in handled or file[-4:] != ".zst":
            continue

        file_path = os.path.join(input_dir, file)
        if "submission" in file:
            handle_submissions(file_path, c)
        elif "comment" in file:
            handle_comments(file_path, c)

        with open(handled_file, "a") as f:
            f.write(f"{file}\n")

    conn.close()
    return

main_subreddit()
append_submission_id(database_location)