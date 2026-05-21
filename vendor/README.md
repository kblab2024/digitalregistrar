# Vendored wheels

This directory holds locally-built Python wheels that are pinned as a
regular dependency in the project's [pyproject.toml](../pyproject.toml).

## `tnmhelper-0.1.0-py3-none-any.whl`

AJCC TNM cancer staging engine. Consumed by
`digital_registrar.staging` (see
[src/digital_registrar/staging/](../src/digital_registrar/staging/)).

- Source: `/Users/kbchang/localcode/tnmhelper`
- Source commit: `687c661fdc295e0f45de9f35a0ddd635f24e0822`
- Rebuild:
  ```bash
  cd /Users/kbchang/localcode/tnmhelper
  pip wheel . -w dist/
  cp dist/tnmhelper-0.1.0-py3-none-any.whl \
     /Users/kbchang/localcode/drr-next/vendor/
  ```
- Data bundle: the wheel intentionally does **not** include
  `tnmhelper_data/`. The compressed data zip lives at
  `src/digital_registrar/staging/data/tnmhelper_data.zip`
  and is rebuilt with:
  ```bash
  python -m tnmhelper.bundle export \
    --out /Users/kbchang/localcode/drr-next/src/digital_registrar/staging/data/tnmhelper_data.zip \
    --source /Users/kbchang/localcode/tnmhelper/tnmhelper_data \
    --verify
  ```

## Install

The repo's `Makefile` `install-dev` target installs the vendor wheel
first so the `tnmhelper` dependency resolves locally. Equivalent manual
install:

```bash
pip install vendor/tnmhelper-0.1.0-py3-none-any.whl
pip install -e .
```
