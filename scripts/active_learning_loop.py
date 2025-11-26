import numpy as np
import torch
import os
import warnings
from tqdm import tqdm

from clean_app.src.CLEAN.infer import infer_maxsep, infer_pvalue
from clean_app.src.CLEAN.utils import ensure_dirs, dump_info

from train_loop import test_CLEAN_model, train_step_triplet, train_step_supconh, train_step_himulcone, validation_loop, save_end_of_training_metrics, reinit_CLEAN
from plots import plot_pca_by_uncertainty, plot_pca_by_class
from dataloader import reformat_emb, update_ec_id_dicts
from utils import save_metrics

def train_CLEAN_model_AL(model, criterion, optimizer, al_strat, train_datamodule, loss='triplet', eval_dataloader=None, test_data_list=[], num_epochs=100, batch_size=32, generate_plots=False, save_path='.', adaptive_rate=100, learning_rate=0.0001, checkpoint_and_eval=False, train_data_path='./', eval_data_path='./', train_filename='train', eval_filename='eval', save_recomputed_embeddings=False, maxsep=True, emb_dir='/emb_data/', cache_dir='/distance_map/', knn=30, shuffle=True, _format_esm=True, temp=0.1, n_pos=9, clip_norm=False, model_name='CLEAN', metrics_save_path='training_metrics.json'):

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

    if len(train_datamodule.unlabeled_indices) < batch_size:
        batch_size = max(len(train_datamodule.unlabeled_indices) / 3, 32) #use 1/3 the data per batch or 32 whichever is larger
        warnings.warn('Batch size decreased to {} due to data size.'.format(batch_size))

    min_val_loss = 100000000

    lowest_val_loss_model = None

    train_datamodule.random_init(n_samples=batch_size)
    
    for epoch in range(num_epochs):

        if epoch != 0:
            if len(train_datamodule.unlabeled_indices) < batch_size:
                train_datamodule.reset() #I think this is ok from a training context

            indices, scores = al_strat.query(model=model, al_datamodule=train_datamodule, acq_size=batch_size, return_utilities=True)
            train_datamodule.update_annotations(indices)

        #model.reset_states() # not sure why we should do this
        model.to(device)
        model.train()

        torch.cuda.empty_cache()
        epoch_loss = 0.0
        vtotal = 0
        vcorrect = 0
        num_batches = 0

        #train one epoch
        #____________________________________________________________________#
        for i, item in enumerate(train_datamodule.train_dataloader()):
            batch_loss = train_step(item, device, optimizer, model, criterion, clip_norm=clip_norm, temp=temp, n_pos=n_pos)
            epoch_loss += batch_loss

            num_batches += 1

        # store epoch loss and accuracy
        avg_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_epoch_loss:.4f}')
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
                    _format_esm, 
                    loss, 
                    epoch,
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

            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.2f}, Validation Accuracy: {100*vacc:.2f}%')

        if len(train_datamodule.unlabeled_indices) < batch_size:
            train_datamodule.reset() #redo the unlabaled/labeled indices to complete number of epochs

        #____________________________________________________________________#

    #if save_recomputed_embeddings:
        #ensure_dirs(save_path+'/'+cache_dir+'/recomputed/')

        #FIXME - not a priority for simulation
    
    save_end_of_training_metrics(model, 
            save_path, 
            model_name, 
            epoch_losses, 
            val_losses, 
            val_aucs, 
            val_prcs, 
            val_accuracies, 
            val_balanced_accuracies, 
            val_precs, 
            val_recs, 
            val_f1s, 
            val_hps, 
            val_hrs, 
            val_hf1s, 
            generate_plots, 
            num_epochs, 
            checkpoint_and_eval, 
            eval_dataloader,
            metrics_save_path,
            active_learning_mode=True)

    if checkpoint_and_eval and eval_dataloader != None:
        return model, lowest_val_loss_model, (epoch_losses, val_losses, val_accuracies)
    else:
        return model, None, (epoch_losses)

def run_CLEAN_active_learning_simulation(model, criterion, optimizer, al_strat, train_datamodule, pool_datamodule, eval_dataloader=None, n_instances=32, n_queries=3, generate_plots=False, save_path='.', adaptive_rate=100, learning_rate=0.0001, checkpoint_and_eval=False, train_data_path='./', eval_data_path='./', pool_data_path='./', train_filename='train', eval_filename='eval', pool_filename='./', pca=None, label_encoder=None, plot_tuple=None, save_recomputed_embeddings=False, maxsep=True, loss='triplet', model_name='CLEAN', metrics_save_path='training_metrics.json', emb_dir='/emb_data/', cache_dir='/distance_map/', clip_norm=False, temp=0.1, n_pos=9, _format_esm=False, test_data_list=[]):

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

    if plot_tuple is not None and generate_plots:
        pool_ids_full, pool_embeddings_full, pool_ecs_full = plot_tuple

    pool_datamodule.random_init(n_samples=n_instances)
    indices = pool_datamodule.labeled_indices

    break_early = False
    exit_round = False
    for i_cycle in range(n_queries+1):
        if break_early:
            exit_round = True

        if not os.path.exists(save_path+'/round_{}'.format(i_cycle)):
            os.makedirs(save_path+'/round_{}'.format(i_cycle))

        if i_cycle != 0:
            if generate_plots:
                seq_ids_before_query = [pool_datamodule.query_dataset.full_list[i] for i in pool_datamodule.unlabeled_indices] #need for plotting

            indices, scores = al_strat.query(model=model, al_datamodule=pool_datamodule, acq_size=n_instances, return_utilities=True)
            scores = scores.cpu()

            print(indices)
            print(scores)

            pool_datamodule.update_annotations(indices)

        #model.reset_states()
        model.to(device)
        model.train()

        torch.cuda.empty_cache()
        epoch_loss = 0.0
        vtotal = 0
        vcorrect = 0
        num_batches = 0

        #train one epoch
        #____________________________________________________________________#
        for i, item in enumerate(pool_datamodule.train_dataloader()):
            batch_loss = train_step(item, device, optimizer, model, criterion, clip_norm=clip_norm, temp=temp, n_pos=n_pos)
            epoch_loss += batch_loss

            num_batches += 1

        # store epoch loss and accuracy
        avg_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss)

        print(f'Round [{i_cycle}/{n_queries}], Training Loss: {avg_epoch_loss:.4f}')
        #____________________________________________________________________#

        #not enough instances to do requested amount so reset the indices
        if len(pool_datamodule.unlabeled_indices) < n_instances:
            #pool_datamodule.reset() #dont think reset is the right functionality from experimental context
            n_instances = len(pool_datamodule.unlabeled_indices) #sample whatever is left
            break_early=True
            warnings.warn('Only Completing {} number of rounds due to runnning out of pool data points.'.format(i_cycle+1))

        #validate one epoch
        #____________________________________________________________________#
        if checkpoint_and_eval and eval_dataloader!=None:
            avg_val_loss, min_val_loss, vpre, vrec, vf1, vauc, vprc, vacc, vbacc, vhp, vhr, vhf1, lowest_val_loss_model = validation_loop(model, 
                    eval_dataloader, 
                    min_val_loss, 
                    device, 
                    criterion, 
                    save_path+'/round_{}/'.format(i_cycle), 
                    model_name, 
                    infer, 
                    train_filename, 
                    eval_filename, 
                    train_data_path, 
                    eval_data_path, 
                    emb_dir, 
                    reformat_emb(train_datamodule.emb, train_datamodule.ec_id_dict), 
                    _format_esm, 
                    loss, 
                    i_cycle,
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

            print(f'Round [{i_cycle}/{n_queries}], Validation Loss: {avg_val_loss:.2f}, Validation Accuracy: {100*vacc:.2f}%')

        #____________________________________________________________________#

        seq_ids = [pool_datamodule.query_dataset.full_list[i] for i in indices]
        ec_ids = [pool_datamodule.id_ec[_id] for _id in seq_ids]

        assert(len(ec_ids) == len(seq_ids))

        save_metrics(seq_ids, save_path+'/round_{}/gene_ids.txt'.format(i_cycle))
        save_metrics(ec_ids, save_path+'/round_{}/ec_ids.txt'.format(i_cycle))
        
        if generate_plots and i_cycle != 0: #can only do after scores are calculated
            current_pool = [list(pool_ids_full).index(_id) for _id in seq_ids_before_query]
            pool_embeddings_current = pool_embeddings_full[current_pool]
            pool_ids_current = pool_ids_full[current_pool]
            pool_ecs_current = pool_ecs_full[current_pool]

            to_plot = [list(pool_ids_current).index(_id) for _id in seq_ids]
            scores_dict = dict(zip(seq_ids_before_query, scores))
            mapped_id_scores = [scores_dict[id_index] for id_index in pool_ids_current]

            plot_pca_by_uncertainty(pca, 
                    pool_embeddings_current, 
                    pool_embeddings_current[to_plot], 
                    mapped_id_scores, 
                    instance=i_cycle, 
                    path=save_path+'/round_{}/'.format(i_cycle))

            plot_pca_by_class(pca, 
                    label_encoder, 
                    pool_embeddings_current, 
                    pool_ecs_current, 
                    pool_embeddings_current[to_plot], 
                    np.array(pool_ecs_current)[to_plot], 
                    instance=i_cycle, 
                    path=save_path+'/round_{}'.format(i_cycle))

        #update data lists to add annotations to train data
        #FIXME - likely bugs; not needed for simulation; not currently supported
        #train_datamodule, pool_datamodule = update_ec_id_dicts(train_datamodule, 
        #        pool_datamodule, 
        #        ec_ids, 
        #        seq_ids, 
        #        train_path=train_data_path, 
        #        pool_path=pool_data_path, 
        #        train_name=train_filename, 
        #        pool_name=pool_filename)

        #FIXME - likely bugs; not needed for simulation; not currently supported
        #if generate_plots or save_recomputed_embeddings:
            #update points for next round
        #    pool_ids_full = []
        #    pool_ecs_full = []
        #    for ec in list(pool_datamodule.ec_id_dict.keys()):
        #        ids_for_query = list(pool_datamodule.ec_id_dict[ec])
        #        pool_ids_full.extend(ids_for_query)
        #        pool_ecs_full.extend([ec]*len(ids_for_query))

        #    pool_embeddings_full = reformat_emb(pool_datamodule.emb, pool_datamodule.ec_id_dict).cpu()

        #FIXME - likely bugs; not needed for simulation; not currently supported
        #if save_recomputed_embeddings:
        #    ensure_dirs(save_path+'/round_{}/{}/recomputed/'.format(cache_dir, i_cycle))

        #    train_ids = []
        #    train_ecs = []
        #    for ec in list(train_datamodule.ec_id_dict.keys()):
        #        ids_for_query = list(train_datamodule.ec_id_dict[ec])
        #        train_ids.extend(ids_for_query)
        #        train_ecs.extend([ec]*len(ids_for_query))

        #    dump_info(train_ids, 
        #            train_ecs, 
        #            train_datamodule.dist_map, 
        #            reformat_emb(train_datamodule.emb, train_datamodule.ec_id_dict), 
        #            path=save_path, 
        #            extra_folder='/round_{}/{}/recomputed/'.format(cache_dir, i_cycle), 
        #            train_file=train_filename)
            
        #    dump_info(pool_ids_full, 
        #            pool_ecs_full, 
        #            pool_datamodule.dist_map, 
        #            pool_embeddings_full.to(device), 
        #            path=save_path, 
        #            extra_folder='/round_{}/{}/recomputed/'.format(cache_dir, i_cycle), 
        #            train_file=pool_filename)

        for (test_data_name, test_data_path, test_data) in test_data_list:
            test_CLEAN_model(model=model, 
                         train_data_path=train_data_path, 
                         test_data_path=test_data_path, 
                         device=device, 
                         train_name=train_filename, 
                         test_name=test_data_name, 
                         checkpoint_dir=save_path+'/round_{}'.format(i_cycle), 
                         metrics_save_path=test_data_name+'_metrics.json',
                         train_emb=reformat_emb(train_datamodule.emb, train_datamodule.ec_id_dict),
                         emb_out_dir=emb_dir,
                         _format_esm=_format_esm,
                         maxsep=maxsep,
                         model_name=model_name)

        #exiting due to lack of points to complete round
        if exit_round:
            break

    save_end_of_training_metrics(model, 
            save_path, 
            model_name, 
            epoch_losses, 
            val_losses, 
            val_aucs, 
            val_prcs, 
            val_accuracies, 
            val_balanced_accuracies, 
            val_precs, 
            val_recs, 
            val_f1s, 
            val_hps, 
            val_hrs, 
            val_hf1s, 
            generate_plots, 
            n_queries, 
            checkpoint_and_eval, 
            eval_dataloader,
            metrics_save_path,
            active_learning_mode=True)

    if checkpoint_and_eval and eval_dataloader != None:
        return model, lowest_val_loss_model, (epoch_losses, val_losses, val_accuracies)
    else:
        return model, None, (epoch_losses)


############################## ORIGINAL - NO CLEAN ###############################################

#rewrote some functions for standard nn
from train_loop import test_model, train_step_standard, validation_loop_standard, save_end_of_training_metrics_standard

def train_standard_model_AL(model, criterion, optimizer, al_strat, train_datamodule, eval_dataloader=None, test_data_list=[], num_epochs=100, batch_size=32, generate_plots=False, save_path='.', learning_rate=0.0001, checkpoint_and_eval=False, train_data_path='./', eval_data_path='./', train_filename='train', eval_filename='eval', emb_dir='/emb_data/', cache_dir='/distance_map/', knn=30, shuffle=True, _format_esm=True, clip_norm=False, model_name='standard', metrics_save_path='training_metrics.json'):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    dtype=torch.float32
    train_step = train_step_standard
    le = train_datamodule.train_dataset.label_encoder

    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)

    epoch_losses = []
    val_losses = []
    val_accuracies = []
    val_precs = []
    val_recs = []
    val_f1s = []
    val_hps = []
    val_hrs = []
    val_hf1s = []
    min_val_loss = 100000000
    lowest_val_loss_model = None

    if len(train_datamodule.unlabeled_indices) < batch_size:
        batch_size = max(len(train_datamodule.unlabeled_indices) / 3, 32) #use 1/3 the data per batch or 32 whichever is larger
        warnings.warn('Batch size decreased to {} due to data size.'.format(batch_size))

    train_datamodule.random_init(n_samples=batch_size)
    
    for epoch in range(num_epochs):

        if epoch != 0:
            if len(train_datamodule.unlabeled_indices) < batch_size:
                train_datamodule.reset() #I think this is ok from a training context

            indices, scores = al_strat.query(model=model, al_datamodule=train_datamodule, acq_size=batch_size, return_utilities=True)
            train_datamodule.update_annotations(indices)

        #model.reset_states() # not sure why we should do this
        model.to(device)
        model.train()

        torch.cuda.empty_cache()
        epoch_loss = 0.0
        vtotal = 0
        vcorrect = 0
        num_batches = 0

        #train one epoch
        #____________________________________________________________________#
        for i, item in enumerate(tqdm(train_datamodule.train_dataloader(), desc='train loop', leave=False)):
            batch_loss = train_step(item, device, optimizer, model, criterion, clip_norm=clip_norm)
            epoch_loss += batch_loss

            num_batches += 1

        # store epoch loss and accuracy
        avg_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_epoch_loss:.4f}')
        #____________________________________________________________________#

        #validate one epoch
        #____________________________________________________________________#
        if checkpoint_and_eval and eval_dataloader!=None:
            avg_val_loss, min_val_loss, vacc, vprec, vrec, vf1, vhp, vhr, vhf1, lowest_val_loss_model = validation_loop_standard(model, 
                    eval_dataloader, 
                    min_val_loss, 
                    device, 
                    criterion, 
                    save_path, 
                    model_name, 
                    le, 
                    train_filename, 
                    eval_filename, 
                    train_data_path, 
                    eval_data_path, 
                    emb_dir, 
                    _format_esm, 
                    epoch,
                    active_learning_mode=True)

            val_losses.append(avg_val_loss)
            val_accuracies.append(vacc)
            val_precs.append(vprec)
            val_recs.append(vrec)
            val_f1s.append(vf1)
            val_hps.append(vhp)
            val_hrs.append(vhr)
            val_hf1s.append(vhf1)

            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.2f}, Validation Accuracy: {100*vacc:.2f}%')

        if len(train_datamodule.unlabeled_indices) < batch_size:
            train_datamodule.reset() #redo the unlabaled/labeled indices to complete number of epochs

        #____________________________________________________________________#

    save_end_of_training_metrics_standard(model, 
            save_path, 
            model_name, 
            epoch_losses, 
            val_losses, 
            val_accuracies, 
            val_precs,
            val_recs,
            val_f1s,
            val_hps, 
            val_hrs, 
            val_hf1s, 
            generate_plots, 
            num_epochs, 
            checkpoint_and_eval, 
            eval_dataloader,
            metrics_save_path,
            active_learning_mode=True)

    if checkpoint_and_eval and eval_dataloader != None:
        return model, lowest_val_loss_model, (epoch_losses, val_losses, val_accuracies)
    else:
        return model, None, (epoch_losses)

def run_standard_active_learning_simulation(model, criterion, optimizer, al_strat, train_datamodule, pool_datamodule, eval_dataloader=None, n_instances=32, n_queries=3, generate_plots=False, save_path='.', learning_rate=0.0001, checkpoint_and_eval=False, train_data_path='./', eval_data_path='./', pool_data_path='./', train_filename='train', eval_filename='eval', pool_filename='./', pca=None, plot_tuple=None, model_name='standard', metrics_save_path='training_metrics.json', emb_dir='/emb_data/', cache_dir='/distance_map/', clip_norm=False, _format_esm=False, test_data_list=[]):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    dtype=torch.float32

    train_step = train_step_standard
    le = train_datamodule.train_dataset.label_encoder

    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)

    epoch_losses = []
    val_losses = []
    val_accuracies = []
    val_precs = []
    val_recs = []
    val_f1s = []
    val_hps = []
    val_hrs = []
    val_hf1s = []
    min_val_loss = 100000000
    lowest_val_loss_model = None

    if plot_tuple is not None and generate_plots:
        pool_ids_full, pool_embeddings_full, pool_ecs_full = plot_tuple

    pool_datamodule.random_init(n_samples=n_instances)
    indices = pool_datamodule.labeled_indices

    break_early = False
    exit_round = False
    for i_cycle in range(n_queries+1):
        if break_early:
            exit_round = True

        if not os.path.exists(save_path+'/round_{}'.format(i_cycle)):
            os.makedirs(save_path+'/round_{}'.format(i_cycle))

        if i_cycle != 0:
            if generate_plots:
                seq_ids_before_query = [pool_datamodule.query_dataset.dataset.full_list_ids[i] for i in pool_datamodule.unlabeled_indices] #need for plotting

            indices, scores = al_strat.query(model=model, al_datamodule=pool_datamodule, acq_size=n_instances, return_utilities=True)
            scores = scores.cpu()

            pool_datamodule.update_annotations(indices)

        #model.reset_states()
        model.to(device)
        model.train()

        torch.cuda.empty_cache()
        epoch_loss = 0.0
        vtotal = 0
        vcorrect = 0
        num_batches = 0

        #train one epoch
        #____________________________________________________________________#
        for i, item in enumerate(pool_datamodule.train_dataloader()):
            batch_loss = train_step(item, device, optimizer, model, criterion, clip_norm=clip_norm)
            epoch_loss += batch_loss

            num_batches += 1

        # store epoch loss and accuracy
        avg_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss)

        print(f'Round [{i_cycle}/{n_queries}], Training Loss: {avg_epoch_loss:.4f}')
        #____________________________________________________________________#

        #not enough instances to do requested amount so reset the indices
        if len(pool_datamodule.unlabeled_indices) < n_instances:
            #pool_datamodule.reset() #dont think reset is the right functionality from experimental context
            n_instances = len(pool_datamodule.unlabeled_indices) #sample whatever is left
            break_early=True
            warnings.warn('Only Completing {} number of rounds due to runnning out of pool data points.'.format(i_cycle+1))

        #validate one epoch
        #____________________________________________________________________#
        if checkpoint_and_eval and eval_dataloader!=None:
            avg_val_loss, min_val_loss, vacc, vprec, vrec, vf1, vhp, vhr, vhf1, lowest_val_loss_model = validation_loop_standard(model, 
                    eval_dataloader, 
                    min_val_loss, 
                    device, 
                    criterion, 
                    save_path+'/round_{}/'.format(i_cycle), 
                    model_name, 
                    le, 
                    train_filename, 
                    eval_filename, 
                    train_data_path, 
                    eval_data_path, 
                    emb_dir, 
                    _format_esm, 
                    i_cycle,
                    active_learning_mode=True)

            val_losses.append(avg_val_loss)
            val_accuracies.append(vacc)
            val_precs.append(vprec)
            val_recs.append(vrec)
            val_f1s.append(vf1)
            val_hps.append(vhp)
            val_hrs.append(vhr)
            val_hf1s.append(vhf1)

            print(f'Round [{i_cycle}/{n_queries}], Validation Loss: {avg_val_loss:.2f}, Validation Accuracy: {100*vacc:.2f}%')

        #____________________________________________________________________#

        seq_ids = [pool_datamodule.query_dataset.dataset.full_list_ids[i] for i in indices]
        ec_ids = [pool_datamodule.train_dataset.id_ec[_id] for _id in seq_ids]

        assert(len(ec_ids) == len(seq_ids))

        save_metrics(seq_ids, save_path+'/round_{}/gene_ids.txt'.format(i_cycle))
        save_metrics(ec_ids, save_path+'/round_{}/ec_ids.txt'.format(i_cycle))
        
        if generate_plots and i_cycle != 0 and al_strat.subset_size == None: #can only do after scores are calculated; NOTE: subset_size messes up indexing below. Need to fix before I can plot when subsets are taken.
            current_pool = [list(pool_ids_full).index(_id) for _id in seq_ids_before_query]
            pool_embeddings_current = pool_embeddings_full[current_pool]
            pool_ids_current = pool_ids_full[current_pool]
            pool_ecs_current = pool_ecs_full[current_pool]

            to_plot = [list(pool_ids_current).index(_id) for _id in seq_ids]
            scores_dict = dict(zip(seq_ids_before_query, scores))
            mapped_id_scores = [scores_dict[id_index] for id_index in pool_ids_current]

            plot_pca_by_uncertainty(pca, 
                    pool_embeddings_current, 
                    pool_embeddings_current[to_plot], 
                    mapped_id_scores, 
                    instance=i_cycle, 
                    path=save_path+'/round_{}/'.format(i_cycle))

            plot_pca_by_class(pca, 
                    le, 
                    pool_embeddings_current, 
                    pool_ecs_current, 
                    pool_embeddings_current[to_plot], 
                    np.array(pool_ecs_current)[to_plot], 
                    instance=i_cycle, 
                    path=save_path+'/round_{}'.format(i_cycle))

        for (test_data_name, test_data_path, test_data) in test_data_list:
            test_model(model, 
                    criterion, 
                    test_data, 
                    device, 
                    checkpoint_dir=save_path+'/round_{}'.format(i_cycle), 
                    metrics_save_path=test_data_name+'_metrics.json')

        #exiting due to lack of points to complete round
        if exit_round:
            break

    save_end_of_training_metrics_standard(model, 
            save_path, 
            model_name, 
            epoch_losses, 
            val_losses, 
            val_accuracies, 
            val_precs,
            val_recs, 
            val_f1s,
            val_hps, 
            val_hrs, 
            val_hf1s, 
            generate_plots, 
            n_queries, 
            checkpoint_and_eval, 
            eval_dataloader,
            metrics_save_path,
            active_learning_mode=True)

    if checkpoint_and_eval and eval_dataloader != None:
        return model, lowest_val_loss_model, (epoch_losses, val_losses, val_accuracies)
    else:
        return model, None, (epoch_losses)
