# COVID-19 Reddit Belgium

End-to-end data and analysis pipeline to build a Reddit dataset about COVID-19 measures in Belgium and reproduce the paper's figures/tables. The project:

- Ingests Pushshift-style .zst JSON lines for Reddit submissions and comments
- Builds a series of curated SQLite databases with topic labels and sentiment
- Runs Julia-based analyses for activity, time series, ECDFs, sentiment context, and SIEBC model fitting
- Exports CSVs and LaTeX tables in `results/`

The pipeline is a hybrid of Python (data ingest, HF models) and Julia (processing, statistics, MCMC with Turing).


## Repo layout

- Top-level scripts
  - `01_instantiate.sh` — one-time environment setup (Julia + Python venv)
  - `02_run-data.sh` — end-to-end data pipeline (build DBs, topics, sentiment, postprocess)
  - `03_run-results.sh` — run all result-generation Julia scripts
- Data pipeline (Python + Julia)
  - `data-pipeline/00_01_db_builder.py` — read `.zst` files to `data/01_belgium.db`
  - `data-pipeline/01_02_selection_processing.jl` — select COVID-related posts to `data/02_belgium.db`
  - `data-pipeline/02_03_db_topic.py` — topic classification (mBERT) → `data/03_topics/`
  - `data-pipeline/03_04_add_mbert.jl` — merge topic scores, filter → `data/04_belgium.db`
  - `data-pipeline/04_05_sentiment.py` — language + sentiment → `data/05_belgium.db`
  - `data-pipeline/05_06_postprocess.jl` — enrich (depth, cascade topics) → `data/06_belgium.db` + `data/db_sizes.csv`
- Results pipeline (Julia)
  - `results-pipeline/01_statistics.jl` — activity stats, power-law fits → `results/01_degree_data/`
  - `results-pipeline/02_timeseries.jl` — rolling activity, trends, negatives → `results/02_timeseries/`
  - `results-pipeline/03_activity.jl` — ECDF roots/replies → `results/03_ecdf/`
  - `results-pipeline/04_sentiment.jl` — context heatmaps, width sensitivity → `results/04_sentiment/`
  - `results-pipeline/05_SIEBC.jl` — model fitting (Turing) + interpretation → `results/05_siebc/` and `mcmc-chains/`
- Julia shared utilities: `util.jl`


## Prerequisites

- Julia 1.10.7 (see `Project.toml` compat)
- Python 3.10+ with `venv`
- System tools: `sqlite3` (libsqlite), ability to build Python wheels
- Optional GPU acceleration (highly recommended for transformers): CUDA-capable GPU + drivers

Model downloads happen automatically from Hugging Face on first run and can be several GB.


## Quick start

1) Prepare folders and environments

```bash
bash 01_instantiate.sh
```

What it does:
- Creates `data/` and `results/`
- Installs the Julia environment and `PowerLaws.jl`
- Creates and activates a Python virtualenv `.venv` and installs `requirements.txt`

2) Put input data in `data/00_zsts/`

- Place `.zst` files with JSON lines of Reddit data (Pushshift-style) in `data/00_zsts/`
- Filenames must contain `submission` or `comment` so the builder picks the target table

3) Run the full data pipeline

```bash
bash 02_run-data.sh
```

This will build `data/01_belgium.db → ... → data/06_belgium.db` and intermediate artifacts under `data/03_topics/` and `data/db_sizes.csv`.

4) Generate all results

```bash
bash 03_run-results.sh
```

- Uses `N_THREADS=16` by default; set to your CPU core count for best speed
- Outputs go to `results/01_*` through `results/05_*` and MCMC chains to `mcmc-chains/`


## Configuration knobs

Most defaults work out of the box, but you can tweak the following:

- Data window and constants: `util.jl`
  - `DATE_RANGE = Date(2020,1,1):Date(2022,6,30)`
  - DB and results paths are joined off repo root
- Subreddit and topics: in each results script
  - `subreddits = ["belgium"]`
  - `topics = [lockdown, mask, vaccin]`
- GPU/CPU selection for transformers
  - `data-pipeline/02_03_db_topic.py`: `device=0` (GPU 0). Use `-1` for CPU.
  - `data-pipeline/04_05_sentiment.py`: `device=0` for several pipelines. Change to `-1` for CPU.
- Which sentiment models to run (defaults are light): `data-pipeline/04_05_sentiment.py`
  - `enabled_models = { "transformers": True, "pattern": False, "vader": False, "perspective": False }`
  - To use Perspective API, set `perspective` to `True` and export `PERSPECTIVE_API_KEY` in your environment
- Threading for results
  - `03_run-results.sh` sets `N_THREADS=16` and passes `-t` to Julia


## What gets produced

- `results/01_degree_data/`
  - Power-law fit CSVs per topic and `*_number_table.tex`
- `results/02_timeseries/`
  - Smoothed daily activity per topic and trend control points
  - Negative days summary + per-day CSVs in `negative-days/`
- `results/03_ecdf/`
  - ECDF of first-root positions vs random baseline per topic
- `results/04_sentiment/`
  - Heatmaps (sentiment vs parent sentiment), diagonalness scores, context sensitivity, width sensitivity
- `results/05_siebc/`
  - Homophily and Wasserstein measures, alpha table, internal state evolution, histograms
- `mcmc-chains/`
  - Per-author `.jld2` chain files for SIEBC model variants (logistic and linear)
- `data/db_sizes.csv`
  - Counts per intermediate DB; useful for sanity checks


## Step-by-step DB lineage

- `data/01_belgium.db` — raw ingest of `.zst` into `submission` and `comment` tables
- `data/02_belgium.db` — COVID-related selection + basic cleaning
- `data/03_topics/*.npy` — mBERT topic scores for comments/submissions
- `data/04_belgium.db` — merge topic scores and filter to COVID topics
- `data/05_belgium.db` — add language + sentiment (BERT, optional VADER/Pattern/Perspective)
- `data/06_belgium.db` — postprocess: timestamps, depth, topic cascade; also updates `data/db_sizes.csv`


## Tips and troubleshooting

- Julia version mismatch
  - Scripts pin to `julia +1.10.7 --project`. Install Julia 1.10.7 (e.g., via juliaup) or adjust the scripts if you know what you’re doing.
- Hugging Face model downloads are large
  - First run will download several GB. Set `HF_HOME` to redirect cache. Use GPU for speed.
- CUDA OOM or no GPU
  - Switch models to CPU by setting `device=-1` in the Python scripts noted above.
- Perspective API errors
  - The Perspective integration is disabled by default. If you enable it, export a valid `PERSPECTIVE_API_KEY`. 400 errors for bad inputs are handled; rate limits are backoff-retried.
- Slow ingest of `.zst`
  - The builder streams and shows progress; large monthly dumps will still take time. Ensure enough disk space.
- Re-running parts of the pipeline
  - The scripts are idempotent-ish and will overwrite downstream DBs; topic/sentiment steps checkpoint their numpy outputs in `data/03_topics/`.


## Development notes

- Python deps are pinned in `requirements.txt`
- Julia deps and versions are pinned in `Project.toml` (includes `PowerLaws.jl` via url in `01_instantiate.sh`)
- Shared helper code (parent lookup, day aggregation, histogram utilities) is in `util.jl`


## Citation

If you use this code or results, please cite the accompanying paper or this repository. (Add citation details here.)


## License

Add a license file (e.g., MIT) if you plan to redistribute. Currently, no explicit license is included in the repository.
