# DTI-BERT
Predicting Drug-Target Interaction by DTI-BERT with pretrained BERT model [ChemBERT](https://www.nature.com/articles/s41598-021-90259-7) and [ProtBERT](https://www.computer.org/csdl/journal/tp/5555/01/09477085/1v2M3TwoN4A) and Paiwise Attention Layers (PALs) <br>
## Developer
Shuan Chen<br>

## Dataset
Download the dataset first at https://www.dropbox.com/home/DTI-dataset and put them in ./data directory <br>
DAVIS dataset includes 25,772 DTI pairs <br>
KIBA dataset includes 117,657D TI pairs <br>
DAVIS_KIBA_data.csv dataset means the model will be trained by all DAVIS data and tested on KIBA dataset <br> 


## Quick start
Quick start by running 
```
!python main.py
```
Use `-d` to change the datasset like
```
!python main.py -d KIBA_DAVIS
```
Use `-l` to test the effect of different layers of PALs.
```
!python main.py -d KIBA_DAVIS -l 5
```
