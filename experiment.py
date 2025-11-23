"""
    experiment.py
    ~~~~~~~~~~~~~~~~~~~~~~
    This module provides functions to run LLM experiments using cancer structured data extraction pipeline from pipeline.py.
    
    Copyright 2025, Kai-Po Chang at Med NLP Lab, China Medical University, with aid from chatGPT.
"""
__version__ = "0.1.0"
__date__ = "2025-10-05"
__author__ = ["Kai-Po Chang"]
__copyright__ = "Copyright 2025, Med NLP Lab, China Medical University"
__license__ = "MIT"

import json
import csv 
from pipeline import setup_pipeline, run_cancer_pipeline
from datetime import datetime
from pathlib import Path
import os
import logging 
from util.logger import setup_logger
import random
import argparse

def create_experiment_folder(base_path: str) -> str:
    """
    Create a new experiment folder with a timestamp.
    
    :param base_path: Base directory to create the experiment folder in
    :return: Path to the created experiment folder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_path = f"{base_path}/experiment_{timestamp}"
    Path(experiment_path).mkdir(parents=True, exist_ok=True)
    print(f"Created experiment folder at: {experiment_path}")
    return experiment_path

def read_random_report(file_path: str) -> tuple[str, str]:
    """
    Read a random text file and its filename from a folder.
    """
    files = list(Path(file_path).glob("*.txt"))
    if not files:
        return "", ""
    random_file = random.choice(files)
    with open(random_file, "r", encoding="utf-8") as f:
        return f.read(), random_file.stem

def run_folder(input_folder: str, output_folder: str, logger: logging.Logger, timingfile: str = "timing.csv"):
    """
    Run the cancer pipeline on all text files in a folder.
    
    :param folder_path: Path to the folder containing text files
    """
    for file in Path(input_folder).glob("*.txt"):
        #print(f"Processing file: {file.name}")
        logger.info(f"Processing file: {file.name}")
        with open(file, "r", encoding="utf-8") as f:
            report = f.read()
        output, elapsed_time = run_cancer_pipeline(report=report, fname=file.stem)
        #print(f"Processed {file.name} in {elapsed_time} seconds.")
        logger.log(logging.INFO, f"Processed {file.name} in {elapsed_time} seconds.")
        # Here you can save the output as needed
        output_file = os.path.join(output_folder, f"{file.stem}_output.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        #print(f"Output saved to: {output_file}")
        logger.info(f"Output saved to: {output_file}")

def run_random_report(data_dir: str, experiment_folder: str, logger: logging.Logger, timingfile: str = "timing.csv"):
    """
    Run the cancer pipeline on a random report from a folder and save the output.
    
    :param data_dir: Directory containing text files
    :param experiment_folder: Directory to save the output
    """
    example_report, example_filename = read_random_report(data_dir)
    if not example_report:
        logger.warning(f"No text files found in {data_dir}.")
        return "", False, 0.0
    example_filename = os.path.basename(example_filename)
    output, elapsed_time = run_cancer_pipeline(report=example_report, fname=example_filename)    
    output_file = os.path.join(experiment_folder, f"{example_filename}_output.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    #print(f"Output saved to: {output_file}")
    logger.info(f"Output saved to: {output_file}")
    #print(f"Elapsed time: {elapsed_time} seconds")
import argparse

def create_experiment_folder(base_path: str) -> str:
    """
    Create a new experiment folder with a timestamp.
    
    :param base_path: Base directory to create the experiment folder in
    :return: Path to the created experiment folder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_path = f"{base_path}/experiment_{timestamp}"
    Path(experiment_path).mkdir(parents=True, exist_ok=True)
    print(f"Created experiment folder at: {experiment_path}")
    return experiment_path

def read_random_report(file_path: str) -> tuple[str, str]:
    """
    Read a random text file and its filename from a folder.
    """
    files = list(Path(file_path).glob("*.txt"))
    if not files:
        return "", ""
    random_file = random.choice(files)
    with open(random_file, "r", encoding="utf-8") as f:
        return f.read(), random_file.stem

def run_folder(input_folder: str, output_folder: str, logger: logging.Logger, timingfile: str = "timing.csv"):
    """
    Run the cancer pipeline on all text files in a folder.
    
    :param folder_path: Path to the folder containing text files
    """
    for file in Path(input_folder).glob("*.txt"):
        #print(f"Processing file: {file.name}")
        logger.info(f"Processing file: {file.name}")
        with open(file, "r", encoding="utf-8") as f:
            report = f.read()
        output, elapsed_time = run_cancer_pipeline(report=report, fname=file.stem)
        #print(f"Processed {file.name} in {elapsed_time} seconds.")
        logger.log(logging.INFO, f"Processed {file.name} in {elapsed_time} seconds.")
        # Here you can save the output as needed
        output_file = os.path.join(output_folder, f"{file.stem}_output.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        #print(f"Output saved to: {output_file}")
        logger.info(f"Output saved to: {output_file}")

def run_random_report(data_dir: str, experiment_folder: str, logger: logging.Logger, timingfile: str = "timing.csv"):
    """
    Run the cancer pipeline on a random report from a folder and save the output.
    
    :param data_dir: Directory containing text files
    :param experiment_folder: Directory to save the output
    """
    example_report, example_filename = read_random_report(data_dir)
    if not example_report:
        logger.warning(f"No text files found in {data_dir}.")
        return "", False, 0.0
    example_filename = os.path.basename(example_filename)
    output, elapsed_time = run_cancer_pipeline(report=example_report, fname=example_filename)    
    output_file = os.path.join(experiment_folder, f"{example_filename}_output.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    #print(f"Output saved to: {output_file}")
    logger.info(f"Output saved to: {output_file}")
    #print(f"Elapsed time: {elapsed_time} seconds")
    logger.info(f"Elapsed time: {elapsed_time} seconds")
    with open(os.path.join(experiment_folder, timingfile), "a", encoding="utf-8", newline="") as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow([example_filename, output["cancer_excision_report"], elapsed_time])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cancer pipeline experiment.")
    parser.add_argument("--input", type=str, help="Path to the input folder containing text files.")
    parser.add_argument("--model", type=str, default="gpt", help="Model name to use (default: gpt).")
    args = parser.parse_args()

    model_name = args.model
    setup_pipeline(model_name)    

    # Create an experiment folder at the parent directory of this script
    base_experiment_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    experiment_parent_folder = os.path.join(base_experiment_path, "experiment")
    experiment_folder = create_experiment_folder(experiment_parent_folder)
    
    # Set up logging
    log_file_path = os.path.join(experiment_folder, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logger(name="experiment_logger", level=logging.DEBUG, log_file=log_file_path, json_format=False)
    #print(f"Logging to: {log_file_path}")
    logger.info(f"Experiment started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input path '{args.input}' does not exist.")
        else:
            print(f"Processing input folder: {input_path}")
            run_folder(input_folder=str(input_path), output_folder=experiment_folder, logger=logger)
    else:
        # Default behavior if no input folder is provided
        print("No input folder specified. Running default experiment configuration.")
        """
        data_dir = r'E:\llmproject\totalregistrar\dataset\4'  # Example data directory
        for _ in range(10):
            run_random_report(data_dir=data_dir, experiment_folder=experiment_folder, logger=logger, timingfile="timing.csv")
        """
        
        for _ in range(5):
            data_dir = r'E:\llmproject\totalregistrar\dataset\tcga{}'.format(str(_+1))
            # Check if default data directory exists before running
            if os.path.exists(data_dir):
                experiment_subfolder = os.path.join(experiment_folder, str(_+1))
                #mkdir the subfolder
                Path(experiment_subfolder).mkdir(parents=True, exist_ok=True)
                run_folder(input_folder=data_dir, output_folder=experiment_subfolder, logger=logger)
            else:
                logger.warning(f"Default data directory not found: {data_dir}")