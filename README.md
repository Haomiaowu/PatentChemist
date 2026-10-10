# PatentChemist

**Learning Medicinal Chemists' Design Behavior from Patent Data**

PatentChemist is a BART-style encoder–decoder that generates analogues from a seed SMILES.

> Manuscript under review. Citation entry will be added upon publication.

## Installation

```bash
pip install -r requirements.txt
# pysmilesutils is not on PyPI; install from GitHub:
pip install git+https://github.com/MolecularAI/pysmilesutils.git
```

## Quick start: generate analogues

```bash
cd code
python generate.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --n 1000 --out analogs.csv
```

A single consumer-grade GPU (≥16 GB) is sufficient; CPU works but is slow.

## Training

Fine-tuning inputs are pickled dicts of the form `{"input": [...], "target": [...]}` (directed pairs, `smiles_A → smiles_B`).

```bash
cd code
# Stage 1: BART denoising pre-training.
python pretrain.py --corpus <corpus.smi> --out runs/pretrain \
    --epochs 5 --token_batch_size 12000 --seed 42

# Stage 2: molecular-pair fine-tuning
python train.py --train_pkl <train.pkl> --val_pkl <val.pkl> \
    --pretrained runs/pretrain/best.ckpt --out runs/finetuned \
    --epochs 2 --token_batch_size 12000 --lr 5e-4 --seed 42
```

Both scripts save raw `best.pt` and `best.ckpt` (`{state_dict, hyper_parameters}`); the latter is what `train.py` and `generate.py` load.

## License

- **Code**: MIT (see `LICENSE`).
- **Datasets & benchmark** (`data/`, `zo19/`): CC-BY-4.0 (see `LICENSE-DATA`).
- **Model weights** (`weights/`): CC-BY-4.0 (see `LICENSE-DATA`).
- The source data derive from SureChEMBL and follow its original license terms; redistribution conditions of the derived SMILES collections are given in `LICENSE-DATA`.

## Large files

`data/interpat_14m.parquet` (~193 MB) and `weights/patentchemist/best.ckpt` (760 MB) exceed GitHub's 100 MB per-file limit and are distributed as a single release asset. After cloning, download and unpack it in the repository root:

```bash
wget -O large-files.zip https://github.com/Haomiaowu/PatentChemist/releases/download/v1.0/large-files.zip
unzip -o large-files.zip && rm large-files.zip
```

