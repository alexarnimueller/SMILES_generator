# SMILES_generator
SMILES-based generative RNN for molecule de novo design

## Introduction
Inspired by natural language processing, this repository presents generative AI for _de novo_ design in drug discovery.
The here presented method utilizes generative recurrent neural networks (RNN) with long short‐term memory (LSTM) cells. 
Molecules are represented with a sequential syntax called SMILES that can be captured by the RNN model with close to
perfect accuracy.

After training, the learned SMILES character probabilities can be used for _de novo_ molecule generation. Compared to
classical design techniques, this method eliminates the need for virtual compound library enumeration and can be used 
for transfer learning, fragment growing or hit to lead optimization.

## Installation
```bash
git clone https://github.com/alexarnimueller/SMILES_generator.git
cd SMILES_generator
```

### Requirements

The project has the following requirements:
- `tensorflow == 1.14`
- `RDKit == 2019.09.3`
- optional: [Fréchet ChEMBLNET Distance](https://github.com/alexarnimueller/FCD) for analysis of the generated molecules

## Usage
Explenations on the different options are given inside the individual files.

### Training
Training a new model: `python train.py`

#### Example:
`python train.py --dataset data/test.csv --name test --train 20 --lr 0.005 --batch 512 --after 2 --sample 100 --augment 5 --preprocess True --stereo 1 --reinforce False --ref None --val 0.1 --seed 42`

### Sampling
Sampling SMILES strings from a trained model: `python sample.py`
#### Example:
`python sample.py --model checkpoint/test --out generated/test_sampled.csv --epoch 9 --num 1000 --temp 1.0 --frag ^ --seed 42`

### Finetuning
Finetuning (_aka transfer learning_) a trained model towards molecules of interest: `python finetune.py`
#### Example:
`python finetune.py --model checkpoint/test --dataset data/actives.csv --name test-finetune --lr 0.005 --epoch 19 -- train 20
--sample 100 --temp 1.0 --after 1 --augment 10 --batch 16 --preprocess False --stereo 1 --reinforce False --mw_filter 200-400
--reference None --val 0.0 --seed 42 --workers 1`

### Analysis
Analyzing sampled SMILES strings: `python analyze.py`
#### Example:
`python analyze.py --generated generated/test_sampled.csv --reference data/test.csv --name test --n 3 --fingerprint ECFP4`

## References
Publications employing similar techniques:

- Gupta, A., Müller, A. T., Huisman, B. J. H., Fuchs, J. A., Schneider, P. and Schneider, G. (2018) Generative recurrent networks for de novo drug design. Mol. Inf. 37, 1700111.
- Merk, D., Grisoni, F., Friedrich, L. and Schneider, G. (2018) Tuning artificial intelligence on the de novo design of natural-product-inspired retinoid X receptor modulators. Communications Chemistry 1, 68.
- Merk, D., Friedrich, L., Grisoni, F. and Schneider, G. (2018) De novo design of bioactive small molecules by artificial intelligence. Mol. Inf. 37, 1700153
