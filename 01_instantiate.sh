mkdir data
mkdir results

julia --project -e "using Pkg; Pkg.add(url="https://www.github.com/TimVWese/PowerLaws.jl"); Pkg.instantiate()"

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
