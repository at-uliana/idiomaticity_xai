# Explainable AI: Idiomaticity Detection

Run `make-split` util to create zero-shot, one-shot or random data splits that comply with restriction on idioms.

To run `make-split` util:
```
python make-split --datadir my_data.tsv --setting zero-shot --outdir mydir --seed 42
```

To run fine-tuning:
```commandline
python fine-tune.py --config_file fine_tuning_config.json
```

To run grid search for hyperparameter optimization:
```commandline
python grid-search.py --config_file grid_search_config.json
```

See `configs` for sample config files.