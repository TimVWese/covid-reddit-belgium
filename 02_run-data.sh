source .venv/bin/activate

python data-pipeline/00_01_db_builder.py && \
julia +1.10.7 --project data-pipeline/01_02_selection_processing.jl && \
python data-pipeline/02_03_db_topic.py && \
julia +1.10.7 --project data-pipeline/03_04_add_mbert.jl && \
python data-pipeline/04_05_sentiment.py && \
julia +1.10.7 --project data-pipeline/05_06_postprocess.jl
