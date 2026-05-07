"""
pipeline.py
This script sets up a pipeline for processing pathology reports using large language models (LLMs). It includes functions for loading models, configuring the dspy library, and defining signatures for various cancer types. The pipeline is designed to extract structured information from pathology reports and convert it into JSON format.

author: Kai-Po Chang @ Med NLP Lab, China Medical University
date: 2025-10-05
"""
__version__ = "0.1.0"
__date__ = "2025-10-05"
__author__ = ["Kai-Po Chang"]
__copyright__ = "Copyright 2025, Med NLP Lab, China Medical University"
__license__ = "MIT"

import json
import logging
import time

import dspy

# Organ-specific DSPy signature classes are resolved dynamically via
# `globals().get(items)` below (see `organmodels` dict). They must be
# present in this module's global namespace, which the star imports
# provide. Do not convert to explicit imports without also refactoring
# the lookup in CancerPipeline.forward().
from .models.breast import *  # noqa: F401, F403
from .models.cervix import *  # noqa: F401, F403
from .models.colon import *  # noqa: F401, F403
from .models.common import *  # noqa: F401, F403
from .models.esophagus import *  # noqa: F401, F403
from .models.liver import *  # noqa: F401, F403
from .models.lung import *  # noqa: F401, F403
from .models.modellist import organmodels
from .models.pancreas import *  # noqa: F401, F403
from .models.prostate import *  # noqa: F401, F403
from .models.stomach import *  # noqa: F401, F403
from .models.thyroid import *  # noqa: F401, F403
from .util.predictiondump import dump_prediction_plain


def timeit(func):
    """
    Decorator to time a function's execution.
    """
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        #elapsed_str = (f"Execution time for {func.__name__}: {end_time - start_time:.4f} seconds")
        #print(elapsed_str)
        return result, end_time - start_time
    return wrapper

def setup_pipeline(model_name: str, overrides: dict | None = None):
    """
    Set up the pipeline by loading the specified model and configuring dspy.

    :param model_name: Name of the model to load
    :param overrides: Optional decoding-parameter overrides forwarded to load_model
    """
    autoconf_dspy(model_name, overrides=overrides)  # noqa: F405
    print("Pipeline setup complete.")

class CancerPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyzer_is_cancer = dspy.Predict(is_cancer)  # noqa: F405
        self.jsonize = dspy.Predict(ReportJsonize)  # noqa: F405

    def forward(self, report: str | list[str], logger: logging.Logger, fname: str = "") -> dict:
        """
        Process the full report to determine if it is a cancer excision and extract margins if applicable.
        Args:
            report (str): The pathology report to analyze.
        """
        print(f"Processing report: {fname}")
        logger.info(f"Processing report: {fname}")
        if isinstance(report, list):
            paragraphs = [p.strip() for p in report if isinstance(p, str) and p.strip()]
        else:
            paragraphs = report.split('\n\n')
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
        context_response = self.analyzer_is_cancer(report=paragraphs)
        if context_response.cancer_excision_report:
            output_report = {
                "cancer_excision_report": True,
                "cancer_category": context_response.cancer_category,
                "cancer_category_others_description": context_response.cancer_category_others_description,
                "cancer_data": {}
            }
            logger.info("This is a cancer excision report.")
            if context_response.cancer_category == 'others':
                logger.info(f"Cancer category is {context_response.cancer_category_others_description}, Currently not implemented.")
            elif context_response.cancer_category:
                logger.info(f"Cancer category is {context_response.cancer_category}.")
            try:
                json_response = self.jsonize(report=paragraphs, cancer_category=context_response.cancer_category)
                json_report = json_response.output

            except Exception:
                json_report = {}

            for items in organmodels.get(context_response.cancer_category, []):
                cls = globals().get(items)
                if cls is None:
                    logger.error(f"Model class {items} not found.")
                    continue
                logger.info(f"Processing organ-specific model: {cls.__name__} at {time.strftime('%Y-%m-%d %H:%M:%S')} for {context_response.cancer_category} cancer for {fname}")
                organ_analyzer = dspy.Predict(cls)
                try:
                    organ_response = organ_analyzer(report=paragraphs, report_jsonized=json_report)
                    organ_data = dump_prediction_plain(organ_response)
                    output_report["cancer_data"].update(organ_data)
                except Exception as e:
                    logger.error(f"Error processing {cls.__name__}: {e}")
                    continue

            return output_report
        else:
            #print("This is NOT a cancer excision report.")
            logger.info("This is NOT a cancer excision report.")
            output_report = {
                "cancer_excision_report": False,
                "cancer_category": None,
                "cancer_data": {}
            }
            print(json.dumps(output_report, indent=2, ensure_ascii=False))
            return output_report

@timeit
def run_pipeline(experiment_model: dspy.Module, **kwargs):
    """
    Run the pipeline with the provided model and additional keyword arguments.
    Args:
        experiment_model (dspy.Predict): The model to run.
        full_report (str): The full report to analyze.
    """
    response = experiment_model(**kwargs, logger=logging.getLogger("experiment_logger"))
    return response



def run_cancer_pipeline(report: str | list[str], fname: str = "") -> tuple[dict, str]:
    """
    Run the cancer pipeline on the provided report.

    :param report: The pathology report to analyze
    :param fname: Optional filename for logging purposes
    :return: Extracted structured data as a dictionary and timing string
    """
    cancer_pipeline = CancerPipeline()
    response, timing = run_pipeline(cancer_pipeline, report=report, fname=fname)
    return response, timing

if __name__ == "__main__":
    # For batch extraction across a folder, use the `registrar-pipeline` console
    # script (see experiment.py). This block is a minimal smoke test.
    setup_pipeline("gpt")
    print("Pipeline is ready for processing pathology reports.")
