import random
import numpy as np
import torch
from PAL import PairwiseAttentionLinear
from tqdm import tqdm
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools, DataStructs, AllChem, Draw

from pubchempy import get_compounds


def load_model(PATH):
    device = 'cpu'
    model = PairwiseAttentionLinear(n_layers = 4)
    model = model.to(device)
    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def drug2feats(drug, return_mol = False):
    smiles = get_compounds(drug, 'name')[0].isomeric_smiles
    arr = np.zeros((1,))
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits = 1024)
    DataStructs.ConvertToNumpyArray(fp, arr)
    if return_mol:
        return arr, mol
    else:
        return arr

def make_predictions(model, drug, gene):
    drug_feat = drug2feats(drug)
    gene_feat = np.load('../data/%s_feats.npy' % gene, allow_pickle = 'TRUE').item()[gene]
    model.eval()
    with torch.no_grad():
        cnt = 0
        feats = torch.cat([torch.FloatTensor(drug_feat), torch.FloatTensor(gene_feat)], dim = -1).unsqueeze(0)
        outputs = model(feats).reshape(-1)
        predictions = torch.sigmoid(outputs).data.numpy()[0]
    return float(predictions)

def make_predictions_1D3T(model, drug, targets):
    preds = {}
    model.eval()
    drug_feat, mol = drug2feats(drug, True)
    seq_feats = np.load('../data/%s_feats.npy' % targets, allow_pickle = 'TRUE').item()
    print ('Predicting %s on the following target genes' % drug)
    print (sorted(list(seq_feats.keys())))
    # print ('Predicting the interactions of %s drugs on the given target...' % (len(drug_feats)))
    with torch.no_grad():
        cnt = 0
        for seq,  seq_feat in tqdm(seq_feats.items()):
            feats = torch.cat([torch.FloatTensor(drug_feat), torch.FloatTensor(seq_feat)], dim = -1).unsqueeze(0)
            outputs = model(feats).reshape(-1)
            predictions = torch.sigmoid(outputs).data.numpy()[0]
            preds[seq] = float(predictions)
            cnt += 1
    return preds, mol

def print_results_1D3T(drug, preds, threshold = 0.1):
    ks = []
    vs = []
    i = 0
    for k, v in sorted(preds.items(), key=lambda item: -item[1]):
        if v < threshold:
            break
        else:
            ks.append(k)
            vs.append(v)
            i += 1
        
    print ('Selected %s/%s targets with possible DTI with drug %s with threshold %s' % (i, len(preds), drug, threshold))
    print ('----------------------------------------------------------------------------------')
    df = pd.DataFrame({'Target':ks, 'Score':vs})
    print (df)
    print ('----------------------------------------------------------------------------------')
    return df

def make_predictions_1T3D(model, target, drug_feats, n_drug = 3000):
    preds = {}
    seq_feat = np.load('../data/%s_feats.npy' % target, allow_pickle = 'TRUE').item()[target]
    
#     print (seq_feat)
#     return seq_feat
    model.eval()
    if n_drug > len(drug_feats):
        n_drug = len(drug_feats)
    shuffled = list(drug_feats.keys())[:n_drug]
    random.shuffle(shuffled)
    with torch.no_grad():
        cnt = 0
        for SMILES in tqdm(shuffled):
            drug_feat =  drug_feats[SMILES]
            feats = torch.cat([torch.FloatTensor(drug_feat), torch.FloatTensor(seq_feat)], dim = -1).unsqueeze(0)
            outputs = model(feats).reshape(-1)
            predictions = torch.sigmoid(outputs).data.numpy()[0]
            preds[SMILES] = float(predictions)
            cnt += 1
    return preds

def print_results_1T3D(target, preds, threshold = 0.1):
    ks = []
    vs = []
    i = 0
    for k, v in sorted(preds.items(), key=lambda item: -item[1]):
        if v < threshold:
            break
        else:
            ks.append(k)
            vs.append(v)
            i += 1
        
    print ('Selected %s/%s compounds with possible DTI with target %s with threshold %s' % (i, len(preds), target, threshold))
    print ('----------------------------------------------------------------------------------')
    df = pd.DataFrame({'SMILES':ks, 'Score':vs})
    PandasTools.AddMoleculeColumnToFrame(df, 'SMILES', 'Molecule')
    print ('----------------------------------------------------------------------------------')
    return df