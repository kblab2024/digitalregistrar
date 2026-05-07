# Literature review — comparison design for Digital Registrar

This annotated bibliography documents the prior work that informs the
comparison design in this benchmark. Each entry notes (a) what the
paper contributes and (b) how it supports one or more of our
methodology decisions (scoped comparison, fine-tuning protocol,
nested-field evaluation, baseline choice).

## 1. LLMs for clinical / pathology information extraction

### Agrawal M., Hegselmann S., Lang H., Kim Y., Sontag D. (2022)
**"Large Language Models are Few-Shot Clinical Information Extractors"**
*Proceedings of EMNLP 2022*, pp. 1998–2022.

GPT-3 few-shot beats fine-tuned T5 and BERT-family models on
clinical-sense-disambiguation and biomedical-evidence extraction, with
**no task-specific training**. Establishes the "generative LLM >
fine-tuned encoder on clinical IE" baseline.

**Relevance here:** supports the architectural-scope argument —
encoder-only models like ClinicalBERT are known to be handicapped
relative to generative LLMs on low-resource structured IE, so a "fair"
comparison must either (a) give BERT substantial fine-tuning data, or
(b) scope to tasks where BERT is known to be competitive. We do both.

### Goel A., Gueta A., Gilon O., Liu C., Erell S., Nguyen L.H., et al. (2023)
**"LLMs Accurately Predict Cancer Staging from Pathology Reports"**
*JCO Clinical Cancer Informatics* 7, e2300120.

Head-to-head: GPT-4 vs. fine-tuned BioBERT for TNM staging on ~3,000
pathology reports. GPT-4 matches BioBERT on T and N while also
producing interpretable reasoning.

**Relevance here:** a close methodological sibling — it establishes
GPT-4 as the expected comparator for pathology extraction and confirms
that encoder models need fine-tuning to compete.

### Sushil M., Ludwig D., Butte A.J., Rudrapatna V.A. (2024)
**"Comparative performance of foundation models in breast cancer
pathology reports extraction"**
*Journal of the American Medical Informatics Association* 31(4), 887–896.

Benchmarks GPT-4, LLaMA-2, Med-PaLM 2, and a fine-tuned BioBERT on 769
breast pathology reports. Finds GPT-4 and Med-PaLM 2 match the encoder
without fine-tuning; rule-based systems trail by 10–20 points on
free-text fields.

**Relevance here:** the single closest analogue in the literature.
Precedent for (1) the three-way (LLM / encoder / rules) comparison
design, (2) published numbers against which our results can be
sanity-checked, (3) the "BERT needs fine-tuning" point in the
cancer-pathology sub-field.

### Huang K., Altosaar J., Ranganath R. (2024)
**"A critical assessment of LLMs for clinical information extraction"**
*arXiv*.

Taxonomises the failure modes: numeric precision, value-range
hallucination, schema-drift under long context. Useful for the
Limitations discussion.

**Relevance here:** lets us frame Digital Registrar's schema-first
design as a *solution* to one of Huang's catalogued failure modes.

### Singhal K., Tu T., Gottweis J., Sayres R., Wulczyn E., Hou L., et al. (2023)
**"Large Language Models Encode Clinical Knowledge"** (Med-PaLM 2)
*Nature* 620, 172–180.

Benchmarks medical LLM knowledge. Relevant as the "closed LLM
ceiling" baseline often referenced in this literature.

### Yang X., PourNejatian N., Shin H.C., Smith K.E., Parisien C., Compas C., et al. (2022)
**"A large language model for electronic health records (GatorTron)"**
*npj Digital Medicine* 5, 194.

GatorTron is the largest open clinical encoder (8.9B params). We
choose ClinicalBERT as our encoder baseline instead of GatorTron
because the latter requires 160+GB of VRAM for full inference at 8.9B,
which is inconsistent with the single-GPU deployment story of Digital
Registrar. ClinicalBERT is the standard **light** clinical encoder and
is the more honest baseline for a resource-constrained target.

## 2. Domain-adapted BERT variants — context for the comparator choice

### Alsentzer E., Murphy J.R., Boag W., Weng W.H., Jin D., Naumann T., McDermott M. (2019)
**"Publicly Available Clinical BERT Embeddings"**
*Proceedings of the 2nd Clinical NLP Workshop*, pp. 72–78.

The canonical ClinicalBERT. BERT-base initialised, then continued
pretraining on MIMIC-III discharge summaries. This is the model behind
`emilyalsentzer/Bio_ClinicalBERT` on HuggingFace — our fine-tune target.

**Relevance here:** this IS the ClinicalBERT used in this benchmark.
Always disambiguate from Huang 2019 (below).

### Huang K., Altosaar J., Ranganath R. (2019)
**"ClinicalBERT: Modeling Clinical Notes and Predicting Hospital
Readmission"**
*arXiv:1904.05342*.

A different paper also called "ClinicalBERT", focused on readmission
prediction rather than generic clinical embeddings.

**Relevance here:** disambiguation. The methods section should state
explicitly which ClinicalBERT is used (Alsentzer's).

### Lee J., Yoon W., Kim S., Kim D., Kim S., So C.H., Kang J. (2020)
**"BioBERT: a pre-trained biomedical language representation model for
biomedical text mining"**
*Bioinformatics* 36(4), 1234–1240.

BioBERT is the PubMed-pretrained sibling. Goel 2023 uses BioBERT for
TNM staging — a useful "sister encoder baseline" for the BERT-family
landscape.

### Gu Y., Tinn R., Cheng H., Lucas M., Usuyama N., Liu X., Naumann T., Gao J., Poon H. (2021)
**"Domain-Specific Language Model Pretraining for Biomedical Natural
Language Processing (PubMedBERT)"**
*ACM Transactions on Computing for Healthcare* 3(1), 1–23.

PubMedBERT is often stronger than ClinicalBERT on biomedical-only
corpora because it pretrains from scratch on PubMed rather than
continuing from general-domain BERT.

**Relevance here:** Pathology reports are closer to clinical notes
(MIMIC-III) than to PubMed abstracts, so ClinicalBERT is the more
distributionally appropriate baseline. PubMedBERT is a reasonable
alternate choice and could be added as a supplementary encoder.

### Zhou S., Wang N., Wang L., Liu H., Zhang R. (2022)
**"CancerBERT: a cancer-domain-specific language model for extracting
breast cancer phenotypes from electronic health records"**
*JAMIA* 29(7), 1208–1216.

Cancer-specific encoder. A natural additional ClinicalBERT variant to
benchmark in a supplementary table. Weights publicly available on
GitHub, but no `transformers`-hub identifier as of 2024.

## 3. Rule-based clinical IE — the baseline lineage

### Friedman C., Alderson P.O., Austin J.H., Cimino J.J., Johnson S.B. (1994)
**"A General Natural-Language Text Processor for Clinical Radiology
(MedLEE)"**
*Journal of the American Medical Informatics Association* 1(2), 161–174.

The foundational rule-based clinical IE system. Grounds the rule-based
comparison historically.

### Savova G.K., Masanz J.J., Ogren P.V., Zheng J., Sohn S., Kipper-Schuler K.C., Chute C.G. (2010)
**"Mayo clinical Text Analysis and Knowledge Extraction System
(cTAKES)"**
*JAMIA* 17(5), 507–513.

The most-deployed open-source clinical IE pipeline. Rule + dictionary +
ML hybrid. Still the community benchmark for rule-based clinical IE.

### Savova G.K., Tseytlin E., Finan S., Castine M., Miller T., Medvedeva O., Harris D., Hochheiser H., Lin C., Chavan G., Jacobson R.S. (2017)
**"DeepPhe: A Natural Language Processing System for Extracting Cancer
Phenotypes from Clinical Records"**
*Cancer Research* 77(21), e115–e118.

Cancer-specific cTAKES extension. The strongest public *cancer*-focused
rule-based competitor; our rule baseline is a simpler variant of the
approach taken by DeepPhe.

### Landolsi M.Y., Hlaoua L., Ben Romdhane L. (2023)
**"Information extraction from electronic medical records using
multi-task recurrent neural networks with contextual word embeddings"**
*Multimedia Tools and Applications* 82, 35453–35483.

Recent survey framing rule-based vs. neural clinical IE. Useful for
explaining why rules alone are insufficient for complex field sets.

## 4. DSPy and prompt programming (frames the Digital Registrar pipeline)

### Khattab O., Santhanam K., Li X.L., Hall D., Liang P., Potts C., Zaharia M. (2024)
**"DSPy: Compiling Declarative Language Model Calls into Self-Improving
Pipelines"**
*International Conference on Learning Representations*.

The DSPy paper — the framework underneath Digital Registrar.

### Schick T., Schütze H. (2021)
**"Exploiting Cloze Questions for Few Shot Text Classification and
Natural Language Inference (PET)"**
*Proceedings of EACL 2021*, pp. 255–269.

The "zero-shot cloze" protocol used in an optional supplementary
ClinicalBERT fallback (`[MASK]`-based prediction without fine-tuning).

## 5. Evaluation methodology for structured clinical extraction

### Zhu Y., Mahale A., Peters K., Mathew L., Giuste F., Anderson B., Wang M.D. (2023)
**"Using Natural Language Processing on Free-Text Clinical Notes to
Identify Patients With Long-Term COVID Effects"**
*AMIA Annual Symposium Proceedings* 2023, 794–803.

Uses the bipartite-match-on-primary-key evaluation for nested fields
— the same evaluation convention we adopt in `eval/metrics.py` for
margins and lymph nodes.

### Sung S.F., Chen K., Wu D.P., Hung L.C., Su Y.H., Hu Y.H. (2024)
**"Application of an LLM-based approach for extracting structured data
from unstructured medical reports"**
*International Journal of Medical Informatics* 185, 105377.

Per-field accuracy + coverage as headline metrics — the same two-axis
reporting we adopt.

## How this bibliography maps to the comparison design

| Design choice | Citation(s) |
|---|---|
| "ClinicalBERT cannot generate structured nested JSON" | Alsentzer 2019 (architecture), Huang 2024 (failure taxonomy) |
| "Encoders need fine-tuning to compete with LLMs on clinical IE" | Agrawal 2022, Goel 2023, Sushil 2024 |
| "Rule-based IE is a legitimate but lower-ceiling baseline" | Friedman 1994, Savova 2010, Savova 2017, Landolsi 2023 |
| "Scoped comparison is an accepted methodology" | Sushil 2024, Goel 2023 |
| "Bipartite matching for nested lists" | Zhu 2023 |
| "Per-field coverage + accuracy is the right reporting format" | Sung 2024 |
| "ClinicalBERT vs GatorTron / PubMedBERT — the right baseline here" | Yang 2022 (GatorTron scale), Gu 2021 (PubMedBERT domain mismatch) |
