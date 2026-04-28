from __future__ import annotations

import argparse
from pathlib import Path

from telemetry_analysis import analyze_iot_telemetry


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an automatic pandas trend report.")
    parser.add_argument(
        "--file",
        default="iot_telemetry_data.csv",
        help="Path to dataset file (default: iot_telemetry_data.csv)",
    )
    args = parser.parse_args()

    result = analyze_iot_telemetry(Path(args.file))

    summary = result.summary
    print("Summary:")
    print(
        f"- File: {summary['file']} | Shape before: {summary['shape_before']} | "
        f"Shape after: {summary['shape_after']} | Duplicates removed: {summary['duplicates_removed']}"
    )

    if summary["time_start"] and summary["time_end"]:
        print(f"- Time range: {summary['time_start']} to {summary['time_end']}")

    if summary["missing_nonzero"]:
        print(f"- Missing data found: {summary['missing_nonzero']}")
    else:
        print("- Missing data found: none")

    print("\nTrends:")
    for item in result.trends:
        print(f"- {item}")


if __name__ == "__main__":
    main()
