# Digital Registrar
A pipeline for processing pathology reports using large language models (LLMs) to extract structured information and convert it into JSON format.
## Description
Digital Registrar is designed to automate the extraction of critical cancer data from pathology reports. It leverages the `dspy` library to orchestrate LLM calls and structure the output according to College of American Pathologists (CAP) protocols.
## Features
- **Multi-Cancer Support**: Capable of processing reports for 10 major cancer types:
  1. Lung
  2. Breast
  3. Colorectal
  4. Prostate
  5. Stomach
  6. Liver and Intrahepatic Bile Duct
  7. Thyroid
  8. Cervix Uteri
  9. Urinary Bladder
  10. Esophagus
- **CAP Protocol Compliance**: Uses CAP protocols (AJCC 8th Edition) for accurate data structuring.
  - *Note: Restricted to 2023-2024 IRB data, hence AJCC 8th Ed.*
- **Structured Output**: Converts unstructured text reports into structured JSON data.
## Directory Structure
- `models/`: Organ-specific cancer models and common utilities.
- `util/`: Utility scripts for logging and data processing.
- `pipeline.py`: Main pipeline script.
- `experiment.py`: Script for running experiments.
- `README.md`: Project documentation.
- `LICENSE`: Project license.
## Prerequisites
- **Ollama**: This project requires [Ollama](https://ollama.com/) to be installed and running to serve the LLMs.
## Installation
Ensure you have Python installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```
## Usage
### Running the Pipeline
You can run the pipeline using the `pipeline.py` script or by creating an experiment using `experiment.py`.
**Basic Usage:**
```python
from pipeline import setup_pipeline, run_cancer_pipeline
# Setup the pipeline with your desired model (e.g., 'gpt')
setup_pipeline("gpt")
# Run the pipeline on a report string
report_text = "..." # Your pathology report text here
result, timing = run_cancer_pipeline(report_text, fname="example_report")
print(result)
```
**Running an Experiment:**
The `experiment.py` script allows you to run the pipeline on a folder of text files. You can specify the input folder using the `--input` argument.
```bash
python experiment.py --input "path/to/your/dataset"
```
If no input folder is specified, it will default to the hardcoded paths in the script (for backward compatibility or testing).
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
## Authors
- **Kai-Po Chang** - [GitHub](https://github.com/kblab2024)
- **Hung-Kai Chen** - [GitHub](https://github.com/Walther-Chen)
- **Po-Yen Tseng** - [GitHub](https://github.com/ThomasTsengCMU)
All at **Med NLP Lab, China Medical University**
