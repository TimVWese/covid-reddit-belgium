source .venv/bin/activate

python data/00_01_db_builder.py
julia --project data/01_02_selection_processing.jl
python data/02_03_db_topic.py
julia --project data/03_04_add_mbert.jl
python data/04_05_sentiment.py
julia --project data/05_06_postprocess.jl
