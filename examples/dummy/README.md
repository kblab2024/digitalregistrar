# Dummy fixture (synthetic, regenerable)

This tree is produced by `scripts/data/gen_dummy_skeleton.py`. The toolkit is
checked in; the output is not. Regenerate any time:

```
python scripts/data/gen_dummy_skeleton.py --out dummy --clean
```

Per-dataset organ scope is read from `configs/organ_code.yaml`:

- **CMUH**: 10 organs (1=pancreas, 2=breast, 3=cervix, 4=colorectal,
  5=esophagus, 6=liver, 7=lung, 8=prostate, 9=stomach, 10=thyroid).
- **TCGA**: 5 organs (1=breast, 2=colorectal, 3=esophagus, 4=stomach, 5=liver).

Defaults: cmuh = 100 cases per organ, tcga = 50 cases per organ,
80% cancer / 20% non-cancer, 3 LLM runs.
For a 10-run sweep matching the real experiments: `--llm-runs 10`.
To restrict to a subset: `--organs breast,colorectal,esophagus,stomach,liver`
(the cross-corpus common-5 set).

CMUH reports are clean key-value text. TCGA reports are chaotic
(dictation-style, abbreviations, shuffled sections) so the two datasets
exercise different parser robustness.
