"""
runner.py
~~~~~~~~~~~~~~~~~~~~~~
Production batch entry point. Runs the cancer-extraction pipeline over every
``*.txt`` in an input folder and writes one ``<stem>_output.json`` per file.

Two engines are available:
    --engine legacy   (default) uses the original ``pipeline.CancerPipeline``.
    --engine factory  uses the schema-driven ``pipeline_factory.CancerPipelineV2``.

Invoke via the ``registrar-pipeline`` console script installed by pyproject.toml,
or directly: ``python -m digital_registrar_research.runner --input <folder>``.

(Renamed from ``experiment.py`` — the unrelated 2026-04 *experimental study*
artifacts in ``docs/experiment_protocol.md`` are intentionally distinct and
keep their original wording.)
"""
__version__ = "0.2.0"
__date__ = "2026-05-07"
__author__ = ["Kai-Po Chang"]
__copyright__ = "Copyright 2025-2026, Med NLP Lab, China Medical University"
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
from .pipeline_factory import run_cancer_pipeline_v2, setup_pipeline_v2
from .util.logger import setup_logger


def create_run_folder(base_path: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = f"{base_path}/run_{timestamp}"
    Path(run_path).mkdir(parents=True, exist_ok=True)
    print(f"Created run folder at: {run_path}")
    return run_path


def read_random_report(file_path: str) -> tuple[str, str]:
    files = list(Path(file_path).glob("*.txt"))
    if not files:
        return "", ""
    random_file = random.choice(files)
    with open(random_file, encoding="utf-8") as f:
        return f.read(), random_file.stem


def _select_runner(engine: str, decomposition: str, jsonize_enabled: bool):
    """Return a callable with the legacy ``run_cancer_pipeline`` signature.

    For ``engine=='factory'``, decomposition + jsonize_enabled are bound here
    so the per-file loop in ``run_folder``/``run_random_report`` doesn't need
    to branch on engine.
    """
    if engine == "legacy":
        return run_cancer_pipeline

    def _run(report, fname=""):
        return run_cancer_pipeline_v2(
            report=report,
            fname=fname,
            decomposition=decomposition,
            jsonize_enabled=jsonize_enabled,
        )
    return _run


def run_folder(
    input_folder: str,
    output_folder: str,
    logger: logging.Logger,
    *,
    runner,
    timingfile: str = "timing.csv",
):
    for file in Path(input_folder).glob("*.txt"):
        logger.info(f"Processing file: {file.name}")
        with open(file, encoding="utf-8") as f:
            report = f.read()
        output, elapsed_time = runner(report=report, fname=file.stem)
        logger.log(logging.INFO, f"Processed {file.name} in {elapsed_time} seconds.")
        output_file = os.path.join(output_folder, f"{file.stem}_output.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"Output saved to: {output_file}")


def run_random_report(
    data_dir: str,
    run_folder_path: str,
    logger: logging.Logger,
    *,
    runner,
    timingfile: str = "timing.csv",
):
    example_report, example_filename = read_random_report(data_dir)
    if not example_report:
        logger.warning(f"No text files found in {data_dir}.")
        return "", False, 0.0
    example_filename = os.path.basename(example_filename)
    output, elapsed_time = runner(report=example_report, fname=example_filename)
    output_file = os.path.join(run_folder_path, f"{example_filename}_output.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Output saved to: {output_file}")
    logger.info(f"Elapsed time: {elapsed_time} seconds")
    with open(os.path.join(run_folder_path, timingfile), "a", encoding="utf-8", newline="") as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow([example_filename, output["cancer_excision_report"], elapsed_time])


def main():
    parser = argparse.ArgumentParser(
        description="Run cancer-extraction pipeline over a folder of pathology reports."
    )
    parser.add_argument("--input", type=str, help="Path to input folder containing *.txt reports.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output folder for *_output.json (default: ./runs/<timestamp>).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt",
        help="Model name from models/common.model_list (default: gpt).",
    )
    parser.add_argument(
        "--engine",
        choices=["legacy", "factory"],
        default="legacy",
        help="Which pipeline engine to use. Default: legacy. Use 'factory' for "
        "the v2 schema-driven pipeline (CancerPipelineV2).",
    )
    parser.add_argument(
        "--decomposition",
        choices=["per_group", "monolithic", "auto"],
        default="auto",
        help="Decomposition mode for the factory engine. Ignored when --engine "
        "is legacy. Default: auto.",
    )
    parser.add_argument(
        "--jsonize",
        dest="jsonize",
        action="store_true",
        help="Enable optional ReportJsonize preprocessing (factory engine only).",
    )
    parser.add_argument(
        "--no-jsonize",
        dest="jsonize",
        action="store_false",
        help="Disable ReportJsonize (default for factory engine).",
    )
    parser.set_defaults(jsonize=False)
    args = parser.parse_args()

    # Configure DSPy for the chosen engine.
    if args.engine == "factory":
        setup_pipeline_v2(args.model)
    else:
        setup_pipeline(args.model)

    runner = _select_runner(args.engine, args.decomposition, args.jsonize)

    if args.output:
        run_folder_path = args.output
        Path(run_folder_path).mkdir(parents=True, exist_ok=True)
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        run_folder_path = create_run_folder(os.path.join(base, "runs"))

    log_file = os.path.join(
        run_folder_path,
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    logger = setup_logger(name="runner_logger", level=logging.DEBUG, log_file=log_file, json_format=False)
    logger.info(
        "Run started at %s (engine=%s, model=%s, decomposition=%s, jsonize=%s)",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        args.engine, args.model, args.decomposition, args.jsonize,
    )

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input path '{args.input}' does not exist.")
            return
        print(f"Processing input folder: {input_path}")
        run_folder(
            input_folder=str(input_path),
            output_folder=run_folder_path,
            logger=logger,
            runner=runner,
        )
        return

    # Default: walk all tcga* subfolders under the packaged example data, if present.
    from .paths import RAW_REPORTS
    if not RAW_REPORTS.exists():
        print(f"No --input given and default example data not found at {RAW_REPORTS}. Pass --input.")
        return
    for sub in sorted(RAW_REPORTS.iterdir()):
        if not sub.is_dir():
            continue
        subfolder_out = os.path.join(run_folder_path, sub.name)
        Path(subfolder_out).mkdir(parents=True, exist_ok=True)
        run_folder(
            input_folder=str(sub),
            output_folder=subfolder_out,
            logger=logger,
            runner=runner,
        )


if __name__ == "__main__":
    main()
