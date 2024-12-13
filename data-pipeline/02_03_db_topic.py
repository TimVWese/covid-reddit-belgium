import os
import sqlite3
from tqdm import tqdm
import numpy as np
from transformers import pipeline

db_path = "data/02_belgium.db"
output_path = "data/03_topics/"

global DTAI_TOPICS
DTAI_TOPICS = ["vaccine", "masks", "lockdown", "schools", "quarantine", "closing-horeca", "testing", "curfew", "other-measure", "not-applicable"]

class Classifier:
    classifiers = {}

    raw_results = {}
    results_mbert = []

    dtai_label_to_idx = {topic: idx for idx, topic in enumerate(DTAI_TOPICS)}

    def __init__(self, n_docs, device=-1):

        print("Loading mbert on device ", device)
        mbert_pipe = pipeline("text-classification", model="DTAI-KULeuven/mbert-corona-tweets-belgium-topics", device=device)
        self.classifiers["mbert"] = lambda docs : mbert_pipe(docs, top_k=None)
        self.raw_results["mbert"] = []
        self.results_mbert = np.zeros((n_docs, len(DTAI_TOPICS)))

    def unpack_dtai(self, doc_results, rows):
        for row_idx, row in enumerate(rows):
            doc_result = doc_results[row_idx]
            for label in doc_result:
                self.results_mbert[row, self.dtai_label_to_idx[label["label"]]] = label["score"]

    def __call__(self, rows, documents):
        for i in range(len(documents)):
            if len(documents[i]) > 512:
                documents[i] = documents[i][:512]
            if len(documents[i]) <= 3:
                documents[i] = "other"

        self.unpack_dtai(self.classifiers["mbert"](documents), rows)

        return

    def store(self, path):
        np.save(path+"_mbert.npy", self.results_mbert)
        return

    def load(self, path):
        """
        Load previously stored results
        """
        if os.path.isfile(path+"_mbert.npy"):
            self.results_mbert = np.load(path+"_mbert.npy")

    def find_next_row(self, start_point=0):
        """
        Find the minimim row number over the enabled models that has not been handled yet,
        i.e. its row in the result matrix sums to zero.
        """
        def find_index(results, threshold=1e-6):
            sums = np.sum(np.abs(results), axis=1)
            indices = np.where(sums < threshold)
            return indices[0][0] if indices[0].size > 0 else len(sums)

        minv = find_index(self.results_mbert[start_point:,:])

        return start_point + minv


def get_processed_text(table, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT processed FROM " + table)
    processed = [row[0] for row in cursor.fetchall()]
    return processed

def add_transformers_models(db_path, output_path , device=-1,
                            tables=["comment", "submission"],
                            start_point=0, end_point=None,
                            batch_size=1000, back_up_every=250):
    for table in tables:
        docs = get_processed_text(table, db_path)
        classifier = Classifier(len(docs), device=device)
        finished = False
        classifier.load(output_path+table)
        handled = classifier.find_next_row(start_point=start_point)
        end_point = len(docs) if end_point is None else min(end_point, len(docs))
        progress = tqdm(total=end_point)
        progress.update(handled)
        count = 0

        while not finished:
            batch_end = handled + batch_size
            if batch_end >= end_point:
                batch_end = end_point
                finished = True

            classifier(range(handled, batch_end), docs[handled:batch_end])

            progress.update(batch_end - handled)
            handled = batch_end

            count += 1
            if count % back_up_every == 0:
                classifier.store(output_path+table)

        classifier.store(output_path+table+"_final")
    return

def main_structured_topics(db_path, output_path):
    tables = ["comment", "submission"]
    start_point = 0
    end_point = None
    device = 0

    add_transformers_models(db_path, output_path, device=device, tables=tables, start_point=start_point, end_point=end_point)
    return

os.makedirs(output_path, exist_ok=True)
main_structured_topics(db_path, output_path)
