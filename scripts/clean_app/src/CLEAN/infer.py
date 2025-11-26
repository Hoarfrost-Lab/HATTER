import torch
from .utils import * 
from .model import LayerNormNet
from .distance_map import *
from .evaluate import *
import pandas as pd
import warnings

def warn(*args, **kwargs):
    pass
warnings.warn = warn

    
def infer_pvalue(train_data, test_data, model=None, report_metrics = False, pretrained=True, model_name=None, gmm = None, train_path='./', test_path='./', emb_out_dir='/emb_data/', pretrained_weights_path='weights.pth', out_dir='./results/', train_emb=None, _format_esm=False, p_value=1e-5, nk_random=20, return_filename=False, train_mode=False):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float32
    id_ec_train, ec_id_dict_train = get_ec_id_dict(train_path+'/'+ train_data+'.csv')
    id_ec_test, _ = get_ec_id_dict(test_path+'/'+test_data+'.csv')
    # load checkpoints
    # NOTE: change this to LayerNormNet(512, 256, device, dtype) 
    # and rebuild with [python build.py install]
    # if inferencing on model trained with supconH loss
    
    #only load model if one is not passed
    if model is None:
        model = LayerNormNet(512, 128, device, dtype)
        if pretrained:
            try:
                checkpoint = torch.load(pretrained_weights_path, map_location=device)
            except FileNotFoundError as error:
                raise Exception('No pretrained weights for this training data')
            
        model.load_state_dict(checkpoint)

    model.eval()
    if train_emb is None:
        train_emb = load_embeddings(ec_id_dict_train, device, dtype, path=train_path, emb_out_dir=emb_out_dir, _format_esm=_format_esm)
    
    emb_train = model(train_emb)
        
    emb_test = model_embedding_test(id_ec_test, model, device, dtype, path=test_path, emb_out_dir=emb_out_dir, _format_esm=_format_esm)
    eval_dist = get_dist_map_test(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)

    if not train_mode:
        seed_everything()
    
    eval_df = pd.DataFrame.from_dict(eval_dist)
    rand_nk_ids, rand_nk_emb_train = random_nk_model(id_ec_train, ec_id_dict_train, emb_train, n=nk_random, weighted=True)
    random_nk_dist_map = get_random_nk_dist_map(emb_train, rand_nk_emb_train, ec_id_dict_train, rand_nk_ids, device, dtype)

    ensure_dirs(out_dir)
    out_filename = out_dir +'/'+ test_data

    write_pvalue_choices(eval_df, out_filename, random_nk_dist_map, p_value=p_value)
    
    if return_filename:
        return out_dir+'/'+test_data+'_maxsep.csv'
    
    if report_metrics:
        pred_label = get_pred_labels(out_filename, pred_type='_pvalue')
        pred_probs = get_pred_probs(out_filename, pred_type='_pvalue')
        true_label, all_label = get_true_labels(test_path+'/'+test_data)
        pre, rec, f1, auc, prc, acc, bacc = get_eval_metrics(pred_label, pred_probs, true_label, all_label)

        return pre, rec, f1, auc, prc, acc, bacc, pred_label, pred_probs, true_label, all_label
        #print("############ EC calling results using maximum separation ############")
        #print('-' * 75)
        #print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)} \n'
        #    f'>>> precision: {pre:.3} | recall: {rec:.3}'
        #    f'| F1: {f1:.3} | AUC: {roc:.3} ')
        #print('-' * 75)


def infer_maxsep(train_data, test_data, model=None, report_metrics = False, pretrained=True, model_name=None, gmm = None, train_path='./', test_path='./', emb_out_dir='/emb_data/', pretrained_weights_path='weights.pth', out_dir='./results/', train_emb=None, _format_esm=False, return_filename=False, train_mode=False):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float32
    id_ec_train, ec_id_dict_train = get_ec_id_dict(train_path+'/'+ train_data+'.csv')
    id_ec_test, _ = get_ec_id_dict(test_path+'/'+test_data+'.csv')
    # load checkpoints
    # NOTE: change this to LayerNormNet(512, 256, device, dtype) 
    # and rebuild with [python build.py install]
    # if inferencing on model trained with supconH loss
    
    #only load model if one is not passed
    if model is None:
        model = LayerNormNet(512, 128, device, dtype)
        if pretrained:
            try:
                checkpoint = torch.load(pretrained_weights_path, map_location=device)
            except FileNotFoundError as error:
                raise Exception('No pretrained weights for this training data')
            
        model.load_state_dict(checkpoint)

    model.eval()
    if train_emb is None:
        train_emb = load_embeddings(ec_id_dict_train, device, dtype, path=train_path, emb_out_dir=emb_out_dir, _format_esm=_format_esm)
    
    emb_train = model(train_emb)
        
    emb_test = model_embedding_test(id_ec_test, model, device, dtype, path=test_path, emb_out_dir=emb_out_dir, _format_esm=_format_esm)
    eval_dist = get_dist_map_test(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)
    
    if not train_mode:
        seed_everything()
    
    eval_df = pd.DataFrame.from_dict(eval_dist)
    ensure_dirs(out_dir)
    out_filename = out_dir +'/'+ test_data
    write_max_sep_choices(eval_df, out_filename, gmm=gmm)
    
    if return_filename:
        return out_dir+'/'+test_data+'_maxsep.csv'
    
    if report_metrics:
        pred_label = get_pred_labels(out_filename, pred_type='_maxsep')
        pred_probs = get_pred_probs(out_filename, pred_type='_maxsep')
        true_label, all_label = get_true_labels(test_path+'/'+test_data)
        pre, rec, f1, auc, prc, acc, bacc = get_eval_metrics(pred_label, pred_probs, true_label, all_label)

        return pre, rec, f1, auc, prc, acc, bacc, pred_label, pred_probs, true_label, all_label
        #print("############ EC calling results using maximum separation ############")
        #print('-' * 75)
        #print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)} \n'
        #    f'>>> precision: {pre:.3} | recall: {rec:.3}'
        #    f'| F1: {f1:.3} | AUC: {roc:.3} ')
        #print('-' * 75)
