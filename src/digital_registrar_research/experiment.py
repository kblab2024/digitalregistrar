"""
experiment.py
~~~~~~~~~~~~~~~~~~~~~~
Batch-extraction entry point. Runs the DSPy cancer-extraction pipeline over
every `*.txt` in an input folder and writes one `<stem>_output.json` per file.

Invoke via the `registrar-pipeline` console script installed by pyproject.toml,
or directly: `python -m digital_registrar_research.experiment --input <folder>`.

Copyright 2025, Kai-Po Chang at Med NLP Lab, China Medical University.
"""
__version__ = "0.1.0"
__date__ = "2025-10-05"
__author__ = ["Kai-Po Chang"]
__copyright__ = "Copyright 2025, Med NLP Lab, China Medical University"
__license__ = "MIT"

import argparse
import csv
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path

from .pipeline import run_cancer_pipeline, setup_pipeline
from .util.logger import setup_logger


def create_experiment_folder(base_path: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_path = f"{base_path}/experiment_{timestamp}"
    Path(experiment_path).mkdir(parents=True, exist_ok=True)
    print(f"Created experiment folder at: {experiment_path}")
    return experiment_path


def read_random_report(file_path: str) -> tuple[str, str]:
    files = list(Path(file_path).glob("*.txt"))
    if not files:
        return "", ""
    random_file = random.choice(files)
    with open(random_file, encoding="utf-8") as f:
        return f.read(), random_file.stem


def run_folder(input_folder: str, output_folder: str, logger: logging.Logger, timingfile: str = "timing.csv"):
    for file in Path(input_folder).glob("*.txt"):
        logger.info(f"Processing file: {file.name}")
        with open(file, encoding="utf-8") as f:
            report = f.read()
        output, elapsed_time = run_cancer_pipeline(report=report, fname=file.stem)
        logger.log(logging.INFO, f"Processed {file.name} in {elapsed_time} seconds.")
        output_file = os.path.join(output_folder, f"{file.stem}_output.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"Output saved to: {output_file}")


def run_random_report(data_dir: str, experiment_folder: str, logger: logging.Logger, timingfile: str = "timing.csv"):
    example_report, example_filename = read_random_report(data_dir)
    if not example_report:
        logger.warning(f"No text files found in {data_dir}.")
        return "", False, 0.0
    example_filename = os.path.basename(example_filename)
    output, elapsed_time = run_cancer_pipeline(report=example_report, fname=example_filename)
    output_file = os.path.join(experiment_folder, f"{example_filename}_output.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Output saved to: {output_file}")
    logger.info(f"Elapsed time: {elapsed_time} seconds")
    with open(os.path.join(experiment_folder, timingfile), "a", encoding="utf-8", newline="") as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow([example_filename, output["cancer_excision_report"], elapsed_time])


def main():
    parser = argparse.ArgumentParser(description="Run cancer-extraction pipeline over a folder of pathology reports.")
    parser.add_argument("--input", type=str, help="Path to input folder containing *.txt reports.")
    parser.add_argument("--output", type=str, default=None, help="Output folder for *_output.json (default: ./experiment/<timestamp>).")
    parser.add_argument("--model", type=str, default="gpt", help="Model name from models/common.model_list (default: gpt).")
    args = parser.parse_args()

    setup_pipeline(args.model)

    if args.output:
        experiment_folder = args.output
        Path(experiment_folder).mkdir(parents=True, exist_ok=True)
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        experiment_folder = create_experiment_folder(os.path.join(base, "experiment"))

    log_file = os.path.join(experiment_folder, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logger(name="experiment_logger", level=logging.DEBUG, log_file=log_file, json_format=False)
    logger.info(f"Experiment started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input path '{args.input}' does not exist.")
            return
        print(f"Processing input folder: {input_path}")
        run_folder(input_folder=str(input_path), output_folder=experiment_folder, logger=logger)
        return

    # Default: walk all tcga* subfolders under the packaged example data, if present.
    from .paths import RAW_REPORTS
    if not RAW_REPORTS.exists():
        print(f"No --input given and default example data not found at {RAW_REPORTS}. Pass --input.")
        return
    for sub in sorted(RAW_REPORTS.iterdir()):
        if not sub.is_dir():
            continue
        subfolder_out = os.path.join(experiment_folder, sub.name)
        Path(subfolder_out).mkdir(parents=True, exist_ok=True)
        run_folder(input_folder=str(sub), output_folder=subfolder_out, logger=logger)


if __name__ == "__main__":
    main()
