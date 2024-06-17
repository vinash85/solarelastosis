import numpy as np
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc, MLP
from models.model_clam import CLAM_SB, CLAM_MB
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def initiate_model(args, ckpt_path, device='cuda'):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, "embed_dim": args.embed_dim}
    if args.return_probs == 'yes':
        model_dict.update({'return_probs': True})
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    elif args.model_type == 'mil': 
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    else: # args.model_type == 'mlp'
        model = MLP(**model_dict)

    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    _ = model.to(device)
    _ = model.eval()
    print(device)
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, df, _ = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label) in enumerate(loader):
        print(f"label shape:{label.shape}")
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)
        
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error
        del data
    #test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger


def prediction(model, loader, args,ckpt_idx):
    model.eval()

    all_probs = np.zeros((len(loader), args.n_classes))
    all_preds = np.zeros(len(loader))
    slide_ids = loader.dataset.slide_data['slide_id']
    slides =[]
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data)
        
        probs = Y_prob.cpu().numpy()
        slides.append(slide_id)

        all_probs[batch_idx] = probs
        all_preds[batch_idx] = Y_hat.item()
    df = pd.DataFrame(data=all_probs,columns=['p0','p1'])
    df['y_hat'] = all_preds.tolist()
    df['case_id'] = slides
    df.to_csv(os.path.join('eval_results','EVAL_'+args.save_exp_code,'ckpt_'+str(ckpt_idx)+'results.csv'))
    return all_probs


def predictions_ckpt(dataset, args, ckpt_path,ckpt_idx):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    all_probs=prediction(model,loader,args,ckpt_idx)
    return all_probs

def get_shap_values(model,loader,device,args):
    model.eval()
    model.to(device)
    feature_names = [ 'HISTYPE_SR1', 'HISTYPE_DP1', 'BRES_SR1', 'BRES_DP1',
                      'SiteMel1', 'Specific site-Mel1', 'Specimen Type1', 
                      'Diagnosis1','Clark Level1', 'V Growth 1', 'Ulceration',
                      'Mitoses1', 'TIL1', 'Regression1', 'CNevus-1', 'Satellites1', 
                      'Pigmentation1', 'Specific site1', 'cv  BRES_DP1', 'cv BRES_SR1', 
                      'Geography', 'New histtype', 'CV bres1_new', 'Lymphct1', 
                      'IIF rule', 'Clark rule', 'Dl Breslow', 'DL Site', 'DL Histology',
                      'DLLU', 'D PATHOLG', 'D SPECIMN', 'D VERTGRW', 'D ULCERA', 'D REGRESS', 
                      'D NEVUS', 'D SATELLT', 'D PIGMENT', 'D clarklv', 'D SITE_SPEC', 'D Site_Spec_cat', 
                      'D breslow_cat', 'D AGE', 'mit1', 'mit2', 'stage1', 'max_ulc', 'max_mit', 'max_stage', 
                      'max_VERTGRW_new', 'VERTGRW1_new', 'max_REGRESS_new', 'REGRESS1_new', 'max_SATELLT_new', 
                      'SATELLT1_new', 'max_PIGMENT_new', 'PIGMENT1_new', 'NEVUS1_new', 'max_NEVUS_new']

    for batch_idx, (data, label) in enumerate(loader):
        data = data.to(device) 
        label = label.to(device) 
        print(data.shape)
        explainer = shap.DeepExplainer(model, data)
        shap_values = explainer.shap_values(data)
        #explanation = shap.Explanation(shap_values[:,:,6],data=data.detach().cpu().numpy())
        #shap.plots.waterfall(explanation, show=False)
        print(len(feature_names))
        shap.summary_plot(shap_values,data.detach().cpu().numpy(), show=False,feature_names=feature_names)
        plt.gcf().set_size_inches(30, 30)
        plt.savefig('summary_plot.png')
    return shap_values


def shap_ckpt(dataset, args,device,ckpt_path):
    model = initiate_model(args, ckpt_path)
    print('Init Loaders')
    loader = get_simple_loader(dataset,batch_size=len(dataset))
    get_shap_values(model,loader,device,args)
    


