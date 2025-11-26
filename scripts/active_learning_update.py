import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import warnings
import numpy as np
import sys
import pickle
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import json

from clean_app.src.CLEAN.utils import get_ec_id_dict, ensure_dirs, dump_info
from clean_app.src.CLEAN.infer import infer_maxsep, infer_pvalue

from dataloader import reformat_emb, update_ec_id_dicts
from utils import save_metrics, parse_infer_file
from train_loop import test_CLEAN_model, train_step_triplet, train_step_supconh, train_step_himulcone, validation_loop, save_end_of_training_metrics, reinit_CLEAN
from plots import plot_pca_by_uncertainty, plot_pca_by_class

def infer_initial_pool_ids(train_data_name, pool_data_name, model, train_data_path='./', pool_data_path='./', pretrained_weights=None, save_path='./', train_emb=None, maxsep=True, emb_dir='/esm_data/', use_old_naming_convention=True):
    if maxsep:
        infer = infer_maxsep
    else:
        infer = infer_pvalue
    
    infer_filename = infer(train_data_name, 
                 pool_data_name, 
                 model=model, 
                 report_metrics = False,
                 return_filename = True,
                 train_path=train_data_path, 
                 test_path=pool_data_path,  
                 pretrained_weights_path=pretrained_weights, 
                 out_dir=save_path, 
                 emb_out_dir=emb_dir,
                 train_emb=train_emb,
                 _format_esm=use_old_naming_convention)

    #parse labels and save new csv file in the right format
    infer_df = parse_infer_file(infer_filename)
    labels = [';'.join(labels) for labels in infer_df['EC number']]

    #add to unlabeled df
    df = pd.read_csv(pool_data_path+'/'+pool_data_name+'.csv', delimiter='\t')
    df['EC number'] = labels
    df.to_csv(pool_data_path+'/'+pool_data_name+'_labeled.csv', sep='\t', index=False)

    return pool_data_path+'/'+pool_data_name+'_labeled.csv'

def run_CLEAN_active_learning_init_step(model, al_strat, train_datamodule, pool_datamodule, train_data_path, pool_data_path, train_filename, pool_filename, n_instances=32, pca=None, label_encoder=None, plot_tuple=None, AL_round=0, save_path='./', generate_plots=False, correlation_file=None):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    #not enough instances to do requested amount so reset the indices
    if len(pool_datamodule.unlabeled_indices) < n_instances:
        #pool_datamodule.reset() #dont think reset is the right functionality from experimental context
        n_instances = len(pool_datamodule.unlabeled_indices) #sample whatever is left
        warnings.warn('Only able to sample {} number of points'.format(n_instances))
    
    indices, scores = al_strat.query(model=model, al_datamodule=pool_datamodule, acq_size=n_instances, return_utilities=True)
    pool_datamodule.update_annotations(indices)

    seq_ids = [pool_datamodule.query_dataset.full_list[i] for i in indices]
    ec_ids = [';'.join(pool_datamodule.id_ec[_id]) for _id in seq_ids]
    
    scores = scores.cpu()
    scores_dict = dict(zip(pool_datamodule.query_dataset.full_list, scores.tolist()))
    mapped_id_scores = [scores_dict[id_index] for id_index in seq_ids]

    assert(len(ec_ids) == len(seq_ids))

    seqs = [np.nan]*len(seq_ids) #option to write in main
    results = [np.nan]*len(seq_ids) #will get filled experimentally
    df = pd.DataFrame(columns=['Entry', 'EC number', 'Sequence', 'Result', 'Score'])
    
    df['Entry'] = seq_ids
    df['EC number'] = ec_ids
    df['Sequence'] = seqs
    df['Result'] = results
    df['Score'] = mapped_id_scores

    df.sort_values(by='EC number')
    df.to_csv(save_path+'/infer_ids.tsv', index=False, sep='\t')

    #not actually needed during init because we wont save recomputed embeddings since experiments may fail
    #train_datamodule, pool_datamodule = update_ec_id_dicts(train_datamodule,
    #                                                       pool_datamodule,
    #                                                       ec_ids,
    #                                                       seq_ids,
    #                                                       train_path=train_data_path,
    #                                                       pool_path=pool_data_path,
    #                                                       train_name=train_filename,
    #                                                       pool_name=pool_filename,
    #                                                       update_mode=False)

    if plot_tuple is not None and generate_plots:
        pool_ids_full, pool_embeddings_full, pool_ecs_full = plot_tuple

        to_plot = [list(pool_ids_full).index(_id) for _id in seq_ids]
        mapped_id_scores = [scores_dict[id_index] for id_index in pool_ids_full]

        plot_pca_by_uncertainty(pca, 
                    pool_embeddings_full, 
                    pool_embeddings_full[to_plot], 
                    mapped_id_scores, 
                    instance=AL_round, 
                    path=save_path)

        plot_pca_by_class(pca, 
                    label_encoder, 
                    pool_embeddings_full, 
                    pool_ecs_full, 
                    pool_embeddings_full[to_plot], 
                    np.array(pool_ecs_full)[to_plot], 
                    instance=AL_round, 
                    path=save_path)

    if correlation_file is not None:
        expression_df = pd.read_csv(correlation_file, header=0, sep='\t')
        mapped_id_scores = [scores_dict[id_index] for id_index in expression_df['Protein_ID']]
        
        s_p, p_p = pearsonr(mapped_id_scores, expression_df['Final_Score'])
        s_s, p_s = spearmanr(mapped_id_scores, expression_df['Final_Score'])
        
        output_dict = {'pearson' : {'statistic' : s_p, 'p-value' : p_p}, 'spearman' : {'statistic' : s_s, 'p-value' : p_s}}
        with open('expression_correlation.txt', 'w') as f:
            json.dump(output_dict, f)

def run_CLEAN_active_learning_update_step(model, criterion, optimizer, al_strat, train_datamodule, pool_datamodule, eval_dataloader=None, test_data_list=[], generate_plots=False, save_path='.', adaptive_rate=100, learning_rate=0.0001, checkpoint_and_eval=False, train_data_path='./', eval_data_path='./', pool_data_path='./', train_filename='train', eval_filename='eval', pool_filename='pool', pca=None, label_encoder=None, plot_tuple=None, save_recomputed_embeddings=False, cache_dir='distance_map', emb_dir='/esm_data/', maxsep=True, loss='triplet', AL_round=1, use_old_naming_convention=True, clip_norm=False, temp=0.1, n_pos=9, knn=30, model_name='CLEAN'):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    dtype=torch.float32

    if maxsep:
        infer = infer_maxsep
    else:
        infer = infer_pvalue

    if loss == 'triplet':
        train_step = train_step_triplet
    elif loss == 'supconh':
        train_step = train_step_supconh
    elif loss == 'himulcone':
        train_step = train_step_himulcone

    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)

    epoch_losses = []
    val_losses = []
    val_aucs = []
    val_prcs = []
    val_accuracies = []
    val_balanced_accuracies = []
    val_precs = []
    val_recs = []
    val_f1s = []
    val_hps = []
    val_hrs = []
    val_hf1s = []

    min_val_loss = 100000000

    lowest_val_loss_model = None

    #model.reset_states()
    model.to(device)
    model.train()

    torch.cuda.empty_cache()
    epoch_loss = 0.0
    vtotal = 0
    vcorrect = 0
    num_batches = 0

    update_indices = [pool_datamodule.query_dataset.full_list.index(_id) for _id in pool_datamodule.query_dataset.dataset.ids_for_update]
    pool_dataloader = DataLoader(pool_datamodule.query_dataset, sampler=update_indices, shuffle=False)
    #train one epoch
    #____________________________________________________________________#
    for i, item in enumerate(pool_dataloader): #need to use query dataset to target specific points in update mode
        batch_loss = train_step(item, device, optimizer, model, criterion, clip_norm=clip_norm, temp=temp, n_pos=n_pos)
        epoch_loss += batch_loss

        num_batches += 1

    # store epoch loss and accuracy
    avg_epoch_loss = epoch_loss / num_batches
    epoch_losses.append(avg_epoch_loss)

    print(f'Round {AL_round}, Training Loss: {avg_epoch_loss:.4f}')
    #____________________________________________________________________#

    #validate one epoch
    #____________________________________________________________________#
    if checkpoint_and_eval and eval_dataloader!=None:
        avg_val_loss, min_val_loss, vpre, vrec, vf1, vauc, vprc, vacc, vbacc, vhp, vhr, vhf1, lowest_val_loss_model = validation_loop(model, 
                    eval_dataloader, 
                    min_val_loss, 
                    device, 
                    criterion, 
                    save_path, 
                    model_name, 
                    infer, 
                    train_filename, 
                    eval_filename, 
                    train_data_path, 
                    eval_data_path, 
                    emb_dir, 
                    reformat_emb(train_datamodule.emb, train_datamodule.ec_id_dict), 
                    use_old_naming_convention, 
                    loss, 
                    AL_round,
                    active_learning_mode=True,
                    temp=temp,
                    n_pos=n_pos)

        val_losses.append(avg_val_loss)
        val_aucs.append(vauc)
        val_prcs.append(vprc)
        val_accuracies.append(vacc)
        val_balanced_accuracies.append(vbacc)
        val_precs.append(vpre)
        val_recs.append(vrec)
        val_f1s.append(vf1)
        val_hps.append(vhp)
        val_hrs.append(vhr)
        val_hf1s.append(vhf1)

        print(f'Round {AL_round}, Validation Loss: {avg_val_loss:.2f}, Validation Accuracy: {100*vacc:.2f}%')

    print('Finished Update')
    torch.save(model.model.state_dict(), save_path+'/'+model_name+'.pth')

    for (name, data_path, test_data) in test_data_list:
        test_CLEAN_model(model=model, 
                         train_data_path=train_data_path, 
                         test_data_path=data_path, 
                         device=device, 
                         train_name=train_filename, 
                         test_name=name, 
                         checkpoint_dir=save_path, 
                         metrics_save_path=name+'_metrics.json', 
                         train_emb=reformat_emb(train_datamodule.emb, train_datamodule.ec_id_dict),
                         emb_out_dir=emb_dir,
                         _format_esm=use_old_naming_convention,
                         maxsep=maxsep)
    
    #____________________________________________________________________#

    if plot_tuple is not None and generate_plots:
        pool_ids_full, pool_embeddings_full, pool_ecs_full = plot_tuple

    #update data lists to add annotations to train data -- this recomputes distmaps and copies over embeddings
    train_datamodule, pool_datamodule = update_ec_id_dicts(train_datamodule, 
                                                           pool_datamodule, 
                                                           pool_datamodule.query_dataset.dataset.ecs_for_update, 
                                                           pool_datamodule.query_dataset.dataset.ids_for_update, 
                                                           train_path=train_data_path, 
                                                           pool_path=pool_data_path, 
                                                           train_name=train_filename, 
                                                           pool_name=pool_filename,
                                                           update_mode=True)

    if generate_plots or save_recomputed_embeddings:
        #update points for next round
        pool_ids_full = []
        pool_ecs_full = []

        for ec in pool_datamodule.train_dataset.full_list:
            ids_for_query = list(pool_datamodule.ec_id_dict[ec])
            pool_ids_full.extend(ids_for_query)
            pool_ecs_full.extend([ec]*len(ids_for_query))

        pool_embeddings_full = reformat_emb(pool_datamodule.emb, pool_datamodule.ec_id_dict).cpu()

    if save_recomputed_embeddings:
        ensure_dirs(save_path)

        train_ids = []
        train_ecs = []
        for ec in train_datamodule.train_dataset.full_list:
            ids_for_query = list(train_datamodule.ec_id_dict[ec])
            train_ids.extend(ids_for_query)
            train_ecs.extend([ec]*len(ids_for_query))

        dump_info(train_ids, 
                train_ecs, 
                train_datamodule.dist_map, 
                reformat_emb(train_datamodule.emb, train_datamodule.ec_id_dict), 
                path=save_path, 
                train_file=train_filename, 
                use_old_naming_convention=use_old_naming_convention,
                cache_dir=cache_dir)

        dump_info(pool_ids_full, 
                pool_ecs_full, 
                pool_datamodule.dist_map, 
                pool_embeddings_full.to(device), 
                path=save_path, 
                train_file=pool_filename, 
                use_old_naming_convention=use_old_naming_convention,
                cache_dir=cache_dir)

    plot_tuple = (pool_ids_full, pool_embeddings_full, pool_ecs_full)

    return train_datamodule, pool_datamodule, plot_tuple
