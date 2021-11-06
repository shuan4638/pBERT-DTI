from argparse import ArgumentParser

import torch
import sklearn
from sklearn import metrics
import torch.nn as nn
import torch.nn.functional as F

from utils import *

def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_loss = 0
    pred_ys = []
    true_ys = []
    for batch_id, batch_data in enumerate(data_loader):
        drug_feats, prot_feats, labels = batch_data
        feats = torch.cat([torch.FloatTensor(drug_feats.float()), torch.FloatTensor(prot_feats.float())], dim = -1)
        labels = torch.LongTensor(labels)
        feats, labels = feats.to(args['device']), labels.to(args['device'])
        outputs = model(feats).reshape(-1)
        loss = loss_criterion(outputs, labels.float()).mean()
        train_loss += loss.item()
        optimizer.zero_grad()      
        loss.backward() 
        optimizer.step()
		
        
        pred_ys.append(F.sigmoid(outputs).cpu().detach().numpy())
        true_ys.append(labels.cpu().detach().numpy())

        if batch_id % args['print_every'] == 0:
            print('\repoch %d/%d, batch %d/%d, loss %.4f' % (epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss), end='', flush=True)
    precision, recall, thresholds = metrics.precision_recall_curve(true_ys, pred_ys)
    auc = metrics.auc(recall, precision)

    print('\nepoch %d/%d, training loss: %.4f, prauc: %.4f' % (epoch + 1, args['num_epochs'], train_loss/batch_id, auc))

def run_an_eval_epoch(args, model, data_loader, loss_criterion):
    model.eval()
    val_loss = 0
    val_acc = 0
    pred_ys = []
    true_ys = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            drug_feats, prot_feats, labels = batch_data
            feats = torch.cat([torch.FloatTensor(drug_feats.float()), torch.FloatTensor(prot_feats.float())], dim = -1)
            # labels = torch.LongTensor(labels)
            feats, labels = feats.to(args['device']), labels.to(args['device'])
            outputs = model(feats).reshape(-1)
            loss = loss_criterion(outputs, labels.float()).mean()
            val_loss += loss.item()
        
            # _, predictions = torch.max(outputs, 1)
            predictions = outputs
            # predictions = outputs[:, 1]
            pred_ys.append(F.sigmoid(predictions).cpu().detach().numpy())
            true_ys.append(labels.cpu().detach().numpy())
		
    # print ('validation loss: %.4f' % (val_loss/batch_id))
    # print (true_ys)
    # print (pred_ys)
    true_ys, pred_ys = np.concatenate(true_ys), np.concatenate(pred_ys)
    precision, recall, thresholds = metrics.precision_recall_curve(true_ys, pred_ys)
    auc = metrics.auc(recall, precision)
    # print ('AUC:', auc)
    # prauc = metrics.average_precision_score(true_ys, pred_ys)
    return auc


def main(args):
    model_name = '%s.pth' % args['dataset']
    args['model_path'] = 'models/' + model_name
    mkdir_p('models')                          
    model, loss_criterion, optimizer, stopper = load_model(args)   
    train_loader, val_loader, test_loader = load_dataloader(args)
    for epoch in range(args['num_epochs']):
        run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)
        prauc = run_an_eval_epoch(args, model, val_loader, loss_criterion)
        early_stop = stopper.step(prauc, model) 
        print('epoch %d/%d, validation prauc: %.4f, best prauc: %.4f' %  (epoch + 1, args['num_epochs'], prauc, stopper.best_score))
        if early_stop:
            print ('Early stopped!!')
            break

    stopper.load_checkpoint(model)
    prauc = run_an_eval_epoch(args, model, test_loader, loss_criterion)
    print('test prauc: %.4f' % prauc)
    
if __name__ == '__main__':
    parser = ArgumentParser('LocalRetro training arguements')
    parser.add_argument('-g', '--gpu', default='cuda:0', help='GPU device to use')
    parser.add_argument('-d', '--dataset', default='DAVIS', help='Dataset to use')
    parser.add_argument('-b', '--batch-size', default=128, help='Batch size of dataloader')                             
    parser.add_argument('-n', '--num-epochs', type=int, default=200, help='Maximum number of epochs for training')
    parser.add_argument('-p', '--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('-l', '--n_layers', type=int, default=3, help='Number of layer in PAL')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4, help='Learning rate of optimizer')
    parser.add_argument('-l2', '--weight-decay', type=float, default=1e-6, help='Weight decay of optimizer')
    parser.add_argument('-ss', '--schedule_step', type=int, default=10, help='Step size of learning scheduler')
    parser.add_argument('-pe', '--print-every', type=int, default=20, help='Print the training progress every X mini-batches')
    args = parser.parse_args().__dict__
    args['device'] = torch.device(args['gpu']) if torch.cuda.is_available() else torch.device('cpu')
    print ('Using device %s' % args['device'])
    main(args)