"""Cascade-gated evaluation orchestrator.

Single subcommand under ``scripts.eval.cli``. Walks ``(run, case)``,
invokes the cascade gates (Stage A: ``cancer_excision_report``;
Stage B: ``cancer_category``; Stage C: ``cancer_data`` field
extraction), and writes three chapter folders that mirror the
manuscript structure.

Entry points:
    register(subparsers)    Argparse registration for the CLI.
    main(args)              Argparse-driven main.

See ``docs/stat_methods.md`` and ``docs/eval/recipes.md`` for usage
examples.
"""
