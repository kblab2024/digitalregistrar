# The Digital Registrar: A model-agnostic, resource-efficient AI framework for comprehensive cancer surveillance from pathology reports

[](https://opensource.org/licenses/Apache-2.0)
[](https://www.python.org/)
[](https://dspy.ai/)
[](https://doi.org/10.5281/zenodo.17689362)

## Overview

**The Digital Registrar** is an open-source, locally deployable AI framework designed to automate the extraction of structured cancer registry data from unstructured surgical pathology reports.

This repository contains the source code, extraction logic (DSPy signatures), and benchmarking scripts associated with the manuscript: **"A Multicancer AI Framework for Comprehensive Cancer Surveillance from Pathology Reports"** (Submitted to *npj Digital Medicine*).

### Key Features

  * **Privacy-First:** Designed to run entirely on-premises using local LLMs (via Ollama), ensuring no PHI leaves the hospital firewall.
  * **Resource-Efficient:** Optimized for single-GPU medical workstations (NVIDIA RTX A6000, 48GB VRAM), resolving the "implementation trilemma" of deployment.
  * **Model Agnostic:** Built on [DSPy](https://github.com/stanfordnlp/dspy), allowing the underlying LLM to be swapped without rewriting extraction logic.
  * **Comprehensive:** Extracts 193+ CAP-aligned fields across 10 distinct cancer types, including complex nested data for margins and lymph nodes.

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

-----

##  Model Zoo & Performance

The following models were benchmarked in the study. We recommend **gpt-oss:20b** for the optimal balance of accuracy and latency on single-GPU setups.

| Model | Architecture | Total Params | Active Params | Rec. VRAM |
| :--- | :--- | :--- | :--- | :--- |
| **gpt-oss:20b** | **Sparse MoE** | **20B** | **\~2B** | **40GB** |
| Qwen3-30B-A3B | Sparse MoE | 30B | \~2.4B | 60GB\* |
| gemma3:27b | Dense | 27B | 27B | 48GB |

*\*Note: Qwen3-30B exceeds the 48GB VRAM limit of standard A6000 cards, leading to memory offloading and higher latency.*

-----

## Citation

If you use this code or the dataset in your research, please cite our preprint:

> **Chow, N.-H., Chang, H., Chen, H.-K., et al.** (2025). "A Multicancer AI Framework for Comprehensive Cancer Surveillance from Pathology Reports" *medRxiv*. DOI: [10.1101/2025.10.21.25338475](https://www.google.com/search?q=https://doi.org/10.1101/2025.10.21.25338475)

### BibTeX

```bibtex
@article {Chow2025.10.21.25338475,
	author = {Chow, Nan-Haw and Chang, Han and Chen, Hung-Kai and Lin, Chen-Yuan and Liu, Ying-Lung and Tseng, Po-Yen and Shiu, Li-Ju and Chu, Yen-Wei and Chung, Pau-Choo and Chang, Kai-Po},
	title = {A Multicancer AI Framework for Comprehensive Cancer Surveillance from Pathology Reports},
	elocation-id = {2025.10.21.25338475},
	year = {2025},
	doi = {10.1101/2025.10.21.25338475},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2025/11/23/2025.10.21.25338475},
	eprint = {https://www.medrxiv.org/content/early/2025/11/23/2025.10.21.25338475.full.pdf},
	journal = {medRxiv}
}
```

-----

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
## Authors
- **Kai-Po Chang** - [GitHub](https://github.com/kblab2024)
- **Hung-Kai Chen** - [GitHub](https://github.com/Walther-Chen)
- **Po-Yen Tseng** - [GitHub](https://github.com/ThomasTsengCMU)
All at **Med NLP Lab, China Medical University**
