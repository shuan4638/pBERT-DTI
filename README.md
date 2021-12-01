# pBERT-DTI
### The winner project of 2021 Daewoong Foundation AI Big Data Hackathon
Predicting Drug-Target Interaction by DTI-BERT with pre-trained protBERT model [ProtBERT](https://www.computer.org/csdl/journal/tp/5555/01/09477085/1v2M3TwoN4A) and Pairwise Attention Layers (PALs) <br>

![](https://i.imgur.com/oycRczH.png)

## Developer
Shuan Chen<br>

## Dataset
Download the dataset first at [dropbox](https://www.dropbox.com/sh/552lndmllnxex4m/AAC3jUQIDRVIiiHaLlaisHTRa?dl=0) and put them in ./data directory <br>
DAVIS dataset includes 25,772 DTI pairs <br>
KIBA dataset includes 117,657D TI pairs <br>
DAVIS_KIBA_data.csv dataset means the model will be trained by all DAVIS data and tested on KIBA dataset <br>

## Quick start
Quick start by running
```
!python main.py
```
Use `-d` to change the dataset like
```
!python main.py -d KIBA_DAVIS
```
Use `-l` to test the effect of different layers of PALs.
```
!python main.py -d KIBA_DAVIS -l 5
```

## Results compared with previous methods (PRAUC)


GANsDTA (Front. Genet., 2020)
DeepCTA (Bioinform., 2020)
SimCNN-DTA (Sci. Rep., 2021)
pBERT-DTI (This propose)


| Method | DAVIS dataset | KIBA dataset |
| -------- | -------- |
| KronRLS (Brief. Bioinform., 2015) | 0.590 | 0.700 |
| DeepDTA (Bioinform., 2018) | 0.582 | 0.630 |
| GANsDTA (Front. Genet., 2020) | 0.493 | 0.600 |
| DeepCTA (Bioinform., 2020) | 0.580 | 0.626 |
| SimCNN-DTA (Sci. Rep., 2021) | 0.651 | 0.705 |
| pBERT-DTI (This propose) | 0.672 | **0.822** |

## DEMO
For the application, I wrote three kinds notebooks as follows:
1. 1D1T: one drug and one target DTI prediction - used to validate the model reliability
2. 1D3T: one drug and multiple targets - used for drug repurposing or side effect prediction
3. 1T3D: one target and multiple drugs - used for fast virtual screening of a certain target
