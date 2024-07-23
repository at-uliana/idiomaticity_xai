apt update ; apt clean
pip install -r requirements.txt
python make-split.py --datadir "/netscratch/usentsova/idiomaticity/english.tsv" --outdir "split" --setting "zero-shot" --n_splits 5
