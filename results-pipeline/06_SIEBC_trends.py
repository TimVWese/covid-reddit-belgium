#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd 
import pymannkendall as mk 


# Mirrors KEYDATES in util.jl
KEYDATES = {
	"vaccin": [
		("2020-03-16", "First trials"),
		("2020-12-28", "Start campaign"),
		("2021-09-22", "Start booster"),
		("2021-11-01", "Healthcare obligation"),
	],
	"mask": [
		("2020-07-09", "General mandate"),
		("2021-09-17", "End in Flanders"),
		("2021-11-17", "Broad reintroduction"),
		("2022-03-04", "General end"),
	],
	"lockdown": [
		("2020-03-13", "Lockdown I"),
		("2020-06-08", ""),
		("2020-07-29", "Lockdown Antwerp"),
		("2020-08-26", ""),
		("2020-10-19", "Lockdown II"),
		("2021-06-09", ""),
		("2021-11-27", "Lockdown III"),
		("2022-02-18", ""),
	],
}

def infer_topic(path):
	# expected file name: belgium_<topic>_internal_state.csv
	stem = path.stem
	parts = stem.split("_")
	if len(parts) < 4:
		raise ValueError(f"Cannot infer topic from filename: {path.name}")
	return parts[1]

def build_periods(ts, topic):
	"""
	Match get_MK_trends logic in results-pipeline/05_SIEBC.jl.
	"""
	periods = [pd.Timestamp(d) for d, _ in KEYDATES[topic]]
	if ts.iloc[0] <= periods[0]:
		periods = [ts.iloc[0]] + periods
	if ts.iloc[-1] >= periods[-1]:
		periods = periods + [ts.iloc[-1]]
	return periods

def get_interval_label(start, end, topic):
	event_lookup = {pd.Timestamp(d): lbl for d, lbl in KEYDATES[topic]}
	s_lbl = event_lookup.get(start)
	e_lbl = event_lookup.get(end)
	s_txt = s_lbl if s_lbl else start.date().isoformat()
	e_txt = e_lbl if e_lbl else end.date().isoformat()
	return f"{s_txt} -> {e_txt}"

def getattr_or_none(obj, name):
	return getattr(obj, name) if hasattr(obj, name) else None

def run_hamed_rao_test(values):
	ys = pd.Series(list(values)).dropna().astype(float).to_numpy()
	out = mk.hamed_rao_modification_test(ys)
	return {
		"trend": getattr_or_none(out, "trend"),
		"h": getattr_or_none(out, "h"),
		"p": getattr_or_none(out, "p"),
		"z": getattr_or_none(out, "z"),
		"Tau": getattr_or_none(out, "Tau"),
		"s": getattr_or_none(out, "s"),
		"var_s": getattr_or_none(out, "var_s"),
		"slope": getattr_or_none(out, "slope"),
		"intercept": getattr_or_none(out, "intercept"),
	}

def process_file(path, alpha):
	topic = infer_topic(path)
	if topic not in KEYDATES:
		raise ValueError(f"Unknown topic '{topic}' for file: {path.name}")

	df = pd.read_csv(path)
	df["date"] = pd.to_datetime(df["date"])
	df = df.sort_values("date").reset_index(drop=True)

	periods = build_periods(df["date"], topic)
	rows: list[dict[str, object]] = []
	for i in range(len(periods) - 1):
		start = periods[i]
		end = periods[i + 1]

		# Same split convention as in get_MK_trends: [start, end)
		idx = (df["date"] >= start) & (df["date"] < end) & df["median"].notna()
		ys_values = pd.to_numeric(df.loc[idx, "median"], errors="coerce").dropna().tolist()  # type: ignore
		n = len(ys_values)

		if n <= 3:
			continue

		midpoint = start + (end - start) / 2
		test_summary = run_hamed_rao_test(ys_values)
		rows.append(
			{
				"topic": topic,
				"source_file": path.name,
				"segment_idx": i + 1,
				"start": start.date().isoformat(),
				"end_exclusive": end.date().isoformat(),
				"midpoint": midpoint.date().isoformat(),
				"segment_label": get_interval_label(start, end, topic),
				"n": n,
				"alpha": alpha,
				**test_summary,
			}
		)

	return pd.DataFrame(rows)

def find_input_files(base_dir):
	patterns = [
		base_dir / "results" / "05_siebc" / "belgium_*_internal_state.csv",
		base_dir / "results" / "05_SIEBC" / "belgium_*_internal_state.csv",
	]
	files: list[Path] = []
	for pattern in patterns:
		files.extend(sorted(pattern.parent.glob(pattern.name)))
	deduped = sorted(set(files))
	if not deduped:
		raise FileNotFoundError("No internal_state files found under results/05_siebc or results/05_SIEBC")
	return deduped


parser = argparse.ArgumentParser(
	description="Run Hamed-Rao Mann-Kendall test on internal-state median time series by key-event periods."
)
parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent)
parser.add_argument("--alpha", type=float, default=0.05)
parser.add_argument(
	"--output",
	type=Path,
	default=None,
	help="Output CSV path (default: <base-dir>/results/05_siebc/belgium_mk_advanced.csv)",
)
args = parser.parse_args()

files = find_input_files(args.base_dir)
frames = [process_file(path, args.alpha) for path in files]
result = pd.concat(frames, ignore_index=True)
result = result.sort_values(["topic", "segment_idx"]).reset_index(drop=True)

output = args.output
if output is None:
	output = args.base_dir / "results" / "05_siebc" / "belgium_mk_advanced.csv"
output.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(output, index=False)

print(f"Wrote {len(result)} rows to {output}")
with pd.option_context("display.max_rows", 30, "display.max_columns", None):
	print(result[["topic", "segment_idx", "segment_label", "n", "trend", "p", "Tau"]])

