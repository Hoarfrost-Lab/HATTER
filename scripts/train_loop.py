import sys
import os
import glob
import torch
from torch import autocast
from itertools import zip_longest
import pandas as pd
from sklearn.metrics import classification_report
from scipy.stats import spearmanr
from hiclass.metrics import precision as h_precision, recall as h_recall, f1 as h_F1
from tqdm import tqdm

from clean_app.src.CLEAN.distance_map import get_dist_map_test, get_dist_map
from clean_app.src.CLEAN.infer import infer_maxsep, infer_pvalue

#from labels import convert_string_to_list
from utils import save_metrics, save_model, convert_string_to_list
from plots import make_loss_plot, make_accuracy_plot, make_metrics_plot
from dataloader import create_CLEAN_dataloader

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def reinit_CLEAN(epoch, adaptive_rate, num_epochs, optimizer, learning_rate, model, checkpoint_dir, model_name, train_dataloader, train_emb, dtype, train_data_path, batch_size, shuffle, knn, loss, _format_esm, active_learning_mode=False):
    model.train()
    if epoch % adaptive_rate == 0 and epoch != num_epochs + 1 and epoch != 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

        # save updated model
        if active_learning_mode:
            torch.save(model.model.state_dict(), checkpoint_dir+'/'+model_name+'_'+str(epoch)+'.pth')
        else:
            torch.save(model.state_dict(), checkpoint_dir+'/'+model_name+'_'+str(epoch)+'.pth')
            
        # delete last model checkpoint
        if epoch != adaptive_rate:
            os.remove(checkpoint_dir+'/'+model_name+'_'+str(epoch-adaptive_rate)+'.pth')
            
        # sample new distance map
        new_distmap = get_dist_map(train_dataloader.dataset.ec_id_dict, train_emb, device, dtype, model=model)
        train_dataloader = create_CLEAN_dataloader(new_distmap, 
                train_dataloader.dataset.id_ec, 
                train_dataloader.dataset.ec_id, 
                train_dataloader.dataset.ec_id_dict, 
                path=train_data_path, 
                batch_size=batch_size, 
                shuffle=shuffle, 
                knn=knn, 
                emb_out_dir=train_dataloader.dataset.emb_out_dir, 
                loss='triplet', 
                _format_esm=_format_esm, 
                return_anchor=active_learning_mode)

    return optimizer, train_dataloader

def train_step_triplet(item, device, optimizer, model, criterion, clip_norm=False, temp=None, n_pos=None):
    model.train()
    anchor, positive, negative = item
    anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

    optimizer.zero_grad()
    with autocast(device_type=device.type, dtype=torch.float32):
        anchor_out = model(anchor)
        pos_out = model(positive)
        neg_out = model(negative)

        loss = criterion(anchor_out, pos_out, neg_out)
        loss.backward()

        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        # store loss and accuracy
        batch_loss = loss.item()

    return batch_loss

def train_step_supconh(item, device, optimizer, model, criterion, clip_norm=False, temp=0.1, n_pos=9):
    model.train()
    item = item.to(device)

    optimizer.zero_grad()
    with autocast(device_type=device.type, dtype=torch.float32):
        out = model(item)

        loss = criterion(out, temp, n_pos)
        loss.backward()

        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        # store loss and accuracy
        batch_loss = loss.item()

    return batch_loss

def train_step_himulcone(item, device, optimizer, model, criterion, clip_norm=False, temp=None, n_pos=None):
    model.train()
    data, labels = item
    data, labels = data.to(device), labels.to(device)

    optimizer.zero_grad()
    with autocast(device_type=device.type, dtype=torch.float32):
        out = model(data)

        loss = criterion(out, labels)
        loss.backward()

        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        # store loss and accuracy
        batch_loss = loss.item()

    return batch_loss

def validation_loop(model, eval_dataloader, min_val_loss, device, criterion, checkpoint_dir, model_name, infer, train_filename, eval_filename, train_data_path, eval_data_path, emb_out_dir, train_emb, _format_esm, loss, epoch, active_learning_mode=False, temp=0.1, n_pos=9):

    model.eval()
    lowest_val_loss_model = model 

    with torch.no_grad():
        num_batches = 0
        val_loss = 0.0
            
        for i, item in enumerate(tqdm(eval_dataloader, desc='val loop', leave=False)):

            if loss == 'himulcone':
                e_data, e_labels = item
                e_data, e_labels = e_data.to(device), e_labels.to(device)

                with autocast(device_type=device.type, dtype=torch.float32):
                    e_out = model(e_data)
                    loss = criterion(e_out, e_labels)
            
            if loss == 'supconh':
                e_data = item
                e_data = e_data.to(device)

                with autocast(device_type=device.type, dtype=torch.float32):
                    e_out = model(e_data)
                    loss = criterion(e_out, temp, n_pos)
    
            if loss == 'triplet':
                anchor, positive, negative = item
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                with autocast(device_type=device.type, dtype=torch.float32):
                    anchor_out = model(anchor)
                    pos_out = model(positive)
                    neg_out = model(negative)

                    loss = criterion(anchor_out, pos_out, neg_out)

                # store loss and accuracy
                val_batch_loss = loss.item()
                val_loss += val_batch_loss

            num_batches += 1

        # store epoch loss and accuracy
        avg_val_loss = val_loss / num_batches

        if avg_val_loss < min_val_loss:
            for f in glob.glob(checkpoint_dir+'/'+model_name+'_lowest_loss_epoch_*.pth'):
                os.remove(f)

            # save updated model
            if active_learning_mode:
                torch.save(model.model.state_dict(), checkpoint_dir+'/'+model_name+'_lowest_loss_epoch_'+str(epoch)+'.pth')
            else:
                torch.save(model.state_dict(), checkpoint_dir+'/'+model_name+'_lowest_loss_epoch_'+str(epoch)+'.pth')
            
            min_val_loss = avg_val_loss
            lowest_val_loss_model = model        

        vpre, vrec, vf1, vauc, vprc, vacc, vbacc, pred_label, pred_probs, true_label, all_label = infer(train_filename, 
                                                                                                          eval_filename, 
                                                                                                          report_metrics = True, 
                                                                                                          train_path=train_data_path, 
                                                                                                          test_path=eval_data_path, 
                                                                                                          model=model, 
                                                                                                          out_dir=checkpoint_dir, 
                                                                                                          emb_out_dir=emb_out_dir,
                                                                                                          train_emb=train_emb.clone().detach(),
                                                                                                          _format_esm=_format_esm,
                                                                                                          train_mode=True)
        decoded_labels = [list(map(convert_string_to_list, x))[0] for x in true_label]
        decoded_preds = [list(map(convert_string_to_list, x))[0] for x in pred_label]

        vhp = h_precision(decoded_labels, decoded_preds)
        vhr = h_recall(decoded_labels, decoded_preds)
        vhf1 = h_F1(decoded_labels, decoded_preds)
        
    return avg_val_loss, min_val_loss, vpre, vrec, vf1, vauc, vprc, vacc, vbacc, vhp, vhr, vhf1, lowest_val_loss_model

def save_end_of_training_metrics(model, checkpoint_dir, model_name, epoch_losses, val_losses, val_aucs, val_prcs, val_accuracies, val_balanced_accuracies, val_precs, val_recs, val_f1s, val_hps, val_hrs, val_hf1s, generate_plots, num_epochs, checkpoint_and_eval, eval_dataloader, metrics_save_path, active_learning_mode=False):

    print('Finished Training')
    if active_learning_mode:
        torch.save(model.model.state_dict(), checkpoint_dir+'/'+model_name+'_'+'final'+'.pth')
    else:
        torch.save(model.state_dict(), checkpoint_dir+'/'+model_name+'_'+'final'+'.pth')

    if isinstance(epoch_losses, torch.Tensor):
        epoch_losses = epoch_losses.cpu().numpy()
    if isinstance(val_losses, torch.Tensor):
        val_losses = val_losses.cpu().numpy()
    if isinstance(val_aucs, torch.Tensor):
        val_aucs = val_aucs.cpu().numpy()
    if isinstance(val_prcs, torch.Tensor):
        val_prcs = val_prcs.cpu().numpy()
    if isinstance(val_accuracies, torch.Tensor):
        val_accuracies = val_accuracies.cpu().numpy()
    if isinstance(val_balanced_accuracies, torch.Tensor):
        val_balanced_accuracies = val_balanced_accuracies.cpu().numpy()
    if isinstance(val_precs, torch.Tensor):
        val_precs = val_precs.cpu().numpy()
    if isinstance(val_recs, torch.Tensor):
        val_recs = val_recs.cpu().numpy()
    if isinstance(val_f1s, torch.Tensor):
        val_f1s = val_f1s.cpu().numpy()
    if isinstance(val_hps, torch.Tensor):
        val_hps = val_hps.cpu().numpy()
    if isinstance(val_hrs, torch.Tensor):
        val_hrs = val_hrs.cpu().numpy()
    if isinstance(val_hf1s, torch.Tensor):
        val_hf1s = val_hf1s.cpu().numpy()

    if generate_plots and num_epochs > 1:
        if checkpoint_and_eval and eval_dataloader != None:
            make_accuracy_plot(val_accuracies, val_balanced_accuracies, file_path=checkpoint_dir+'/val_accuracy_plot.png', model_name=model_name)
            make_metrics_plot(val_aucs, val_prcs, val_precs, val_recs, val_f1s, val_hps, val_hrs, val_hf1s, file_path=checkpoint_dir+'/val_metrics_plot.png', model_name=model_name)
            make_loss_plot(epoch_losses, val_losses, file_path=checkpoint_dir+'/lossplot.png', model_name=model_name)
        else:
            make_loss_plot(epoch_losses, file_path=checkpoint_dir+'/lossplot.png', model_name=model_name)

    # save metrics to a file
    metrics = dict(zip(list(range(1, len(epoch_losses)+1)), map(list, zip_longest(epoch_losses, val_losses, val_aucs, val_accuracies, val_balanced_accuracies, val_precs, val_recs, val_f1s, val_hps, val_hrs, val_hf1s))))

    save_metrics(metrics, checkpoint_dir+'/'+metrics_save_path)

    if checkpoint_and_eval and eval_dataloader != None:
        train_corr = spearmanr(val_losses, val_balanced_accuracies)
    else:
        train_corr = None

    save_metrics({'valloss_valacc_correlation' : train_corr}, checkpoint_dir+'/corr.json')


def train_CLEAN_model(model, criterion, optimizer, train_dataloader, train_emb, train_distmap, num_epochs=25, adaptive_rate=100, learning_rate=0.0001, metrics_save_path='training_metrics.json', checkpoint_dir='.', checkpoint_and_eval=False, eval_dataloader=None, clip_norm=False, model_name='CLEAN', generate_plots=False, train_data_path='./', train_filename=None, eval_data_path='./', eval_filename=None, dtype=torch.float32, batch_size=6000, shuffle=False, knn=30, _format_esm=False, temp=0.1, n_pos=9, maxsep=True, loss='triplet'):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

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

    if checkpoint_dir is not None and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Initialize containers to store metrics
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

    model.to(device)
    lowest_val_loss_model = None

    for epoch in range(num_epochs):

        #reinitialize every 100 epochs (following CLEAN paper)
        # -------------------------------------------------------------------- #
        optimizer, train_dataloader = reinit_CLEAN(epoch, 
                adaptive_rate, 
                num_epochs, 
                optimizer, 
                learning_rate, 
                model, 
                checkpoint_dir, 
                model_name, 
                train_dataloader, 
                train_emb, 
                dtype, 
                train_data_path, 
                batch_size, 
                shuffle, 
                knn, 
                loss, 
                _format_esm)
        # -------------------------------------------------------------------- #

        torch.cuda.empty_cache()

        model.train()
        epoch_loss = 0.0
        vtotal = 0
        vcorrect = 0
        num_batches = 0

        print('Loading Embeddings...')
        #train one epoch
        #____________________________________________________________________#
        for i, item in enumerate(tqdm(train_dataloader, desc='train loop', leave=False)):
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
                    checkpoint_dir, 
                    model_name, 
                    infer, 
                    train_filename, 
                    eval_filename, 
                    train_data_path, 
                    eval_data_path, 
                    train_dataloader.dataset.emb_out_dir, 
                    train_emb, 
                    _format_esm, 
                    loss, 
                    epoch,
                    active_learning_mode=False,
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

        #____________________________________________________________________#

    save_end_of_training_metrics(model, 
            checkpoint_dir, 
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
            metrics_save_path)

    if checkpoint_and_eval and eval_dataloader != None:
        return model, lowest_val_loss_model, (epoch_losses, val_losses, val_accuracies)
    else:
        return model, None, (epoch_losses)

def test_CLEAN_model(model, train_data_path='./', test_data_path='./', device='cpu', dtype=torch.float32, train_name='train', test_name='test', model_name='CLEAN', checkpoint_dir='./', metrics_save_path='test_metrics.json', train_emb=None, emb_out_dir='/emb_data/', _format_esm=False, maxsep=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if maxsep:
        infer = infer_maxsep
    else:
        infer = infer_pvalue

    if checkpoint_dir is not None and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


    model.eval()
    with torch.no_grad():
        vpre, vrec, vf1, vauc, vprc, vacc, vbacc, pred_label, pred_probs, true_label, all_label = infer(train_name, 
                                                                                                  test_name, 
                                                                                                  report_metrics = True, 
                                                                                                  train_path=train_data_path, 
                                                                                                  test_path=test_data_path, 
                                                                                                  model=model, 
                                                                                                  out_dir=checkpoint_dir, 
                                                                                                  emb_out_dir=emb_out_dir,
                                                                                                  train_emb=train_emb,
                                                                                                  _format_esm=_format_esm)

        decoded_labels = [list(map(convert_string_to_list, x))[0] for x in true_label]
        decoded_preds = [list(map(convert_string_to_list, x))[0] for x in pred_label]

        vhp = h_precision(decoded_labels, decoded_preds)
        vhr = h_recall(decoded_labels, decoded_preds)
        vhf1 = h_F1(decoded_labels, decoded_preds)


    print(f'Test Accuracy: {100*vacc:.2f}%')

    metrics = {
        'auc': vauc,
        'prc': vprc,
        'accuracy': vacc,
        'balanced-accuracy': vbacc,
        'precision': vpre,
        'recall': vrec,
        'f1-score': vf1,
        'h-precision' : vhp,
        'h-recall' : vhr,
        'h-f1-score' : vhf1
    }

    save_metrics(metrics, checkpoint_dir+'/'+metrics_save_path)


###################################### ##########################################
# ORIGINAL IMPLEMENTATION - NO CLEAN
# Adding back to get explicit comparison of simulations versus CLEAN simulations
# Many features not implemented using this code
###################################### ##########################################

def train_step_standard(item, device, optimizer, model, criterion, clip_norm=False):
    model.train()
    data, labels, index = item
    data, labels = data.to(device), labels.to(device)

    optimizer.zero_grad()
    with autocast(device_type=device.type, dtype=torch.float32):
        out = model(data)

        loss = criterion(out, labels)
        loss.backward()

        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        # store loss and accuracy
        batch_loss = loss.item()

    return batch_loss

def validation_loop_standard(model, eval_dataloader, min_val_loss, device, criterion, checkpoint_dir, model_name, le, train_filename, eval_filename, train_data_path, eval_data_path, emb_out_dir, _format_esm, epoch, active_learning_mode=False):

    lowest_val_loss_model = model        
    running_vloss = 0.0
    running_acc = 0.0
    running_prec = 0.0
    running_rec = 0.0
    running_f1 = 0.0
    running_hp = 0.0
    running_hr = 0.0
    running_hf1 = 0.0
    vtotal = 0
    vcorrect = 0

    if active_learning_mode:
        model.eval()
        model.model.eval()
    else:
        model.eval()
    
    model.to(device)
    with torch.no_grad():
        for j, (vemb, vlabels, vindex) in enumerate(tqdm(eval_dataloader, desc='val loop', leave=False)):
            vemb, vlabels = vemb.to(device), vlabels.to(device)

            with autocast(device_type=device.type, dtype=torch.float32):
                voutputs = model(vemb)

                _v, vpredicted = torch.max(voutputs.data, 1)
                vtotal += vlabels.size(0)
                vcorrect += (vpredicted == vlabels).sum().item()

                vloss = criterion(voutputs, vlabels).item()
                running_vloss += vloss

                vlabels = vlabels.detach().cpu().numpy()
                vpredicted = vpredicted.detach().cpu().numpy()
                        
                report = classification_report(vlabels, vpredicted, output_dict=True, zero_division=0.0)
                        
                decoded_labels = le.inverse_transform(vlabels)
                decoded_preds = le.inverse_transform(vpredicted)

                decoded_labels = list(map(convert_string_to_list, decoded_labels))
                decoded_preds = list(map(convert_string_to_list, decoded_preds))
    
                hp = h_precision(decoded_labels, decoded_preds)
                hr = h_recall(decoded_labels, decoded_preds)
                hf1 = h_F1(decoded_labels, decoded_preds)
                
                running_prec += report['weighted avg']['precision']
                running_rec += report['weighted avg']['recall']
                running_f1 += report['weighted avg']['f1-score']
                running_hp += hp
                running_hr += hr
                running_hf1 += hf1

        avg_vacc = vcorrect / vtotal
        avg_vloss = running_vloss / (j + 1)
        avg_prec = running_hp / (j + 1)
        avg_rec = running_hr / (j + 1)
        avg_f1 = running_hf1 / (j + 1)
        avg_hp = running_hp / (j + 1)
        avg_hr = running_hr / (j + 1)
        avg_hf1 = running_hf1 / (j + 1)

        if avg_vacc < min_val_loss:
            for f in glob.glob(checkpoint_dir+'/'+model_name+'_lowest_loss_epoch_*.pth'):
                os.remove(f)

            # save updated model
            if active_learning_mode:
                torch.save(model.model.state_dict(), checkpoint_dir+'/'+model_name+'_lowest_loss_epoch_'+str(epoch)+'.pth')
            else:
                torch.save(model.state_dict(), checkpoint_dir+'/'+model_name+'_lowest_loss_epoch_'+str(epoch)+'.pth')
            
            min_val_loss = avg_vacc
            lowest_val_loss_model = model        

    return avg_vloss, min_val_loss, avg_vacc, avg_prec, avg_rec, avg_f1, avg_hp, avg_hr, avg_hf1, lowest_val_loss_model

def save_end_of_training_metrics_standard(model, checkpoint_dir, model_name, epoch_losses, val_losses, val_accuracies, val_precs, val_recs, val_f1s, val_hps, val_hrs, val_hf1s, generate_plots, num_epochs, checkpoint_and_eval, eval_dataloader, metrics_save_path, active_learning_mode=False):

    print('Finished Training')
    if active_learning_mode:
        torch.save(model.model.state_dict(), checkpoint_dir+'/'+model_name+'_'+'final'+'.pth')
    else:
        torch.save(model.state_dict(), checkpoint_dir+'/'+model_name+'_'+'final'+'.pth')

    if isinstance(epoch_losses, torch.Tensor):
        epoch_losses = epoch_losses.cpu().numpy()
    if isinstance(val_losses, torch.Tensor):
        val_losses = val_losses.cpu().numpy()
    if isinstance(val_accuracies, torch.Tensor):
        val_accuracies = val_accuracies.cpu().numpy()
    if isinstance(val_precs, torch.Tensor):
        val_precs = val_precs.cpu().numpy()
    if isinstance(val_recs, torch.Tensor):
        val_recs = val_recs.cpu().numpy()
    if isinstance(val_f1s, torch.Tensor):
        val_f1s = val_f1s.cpu().numpy()
    if isinstance(val_hps, torch.Tensor):
        val_hps = val_hps.cpu().numpy()
    if isinstance(val_hrs, torch.Tensor):
        val_hrs = val_hrs.cpu().numpy()
    if isinstance(val_hf1s, torch.Tensor):
        val_hf1s = val_hf1s.cpu().numpy()

    if generate_plots and num_epochs > 1:
        if checkpoint_and_eval and eval_dataloader != None:
            make_accuracy_plot(val_accuracies, file_path=checkpoint_dir+'/val_accuracy_plot.png', model_name=model_name)
            make_metrics_plot(precs=val_precs, recs=val_recs, f1s=val_f1s, hps=val_hps, hrs=val_hrs, hf1s=val_hf1s, file_path=checkpoint_dir+'/val_metrics_plot.png', model_name=model_name)
            make_loss_plot(epoch_losses, val_losses, file_path=checkpoint_dir+'/lossplot.png', model_name=model_name)
        else:
            make_loss_plot(epoch_losses, file_path=checkpoint_dir+'/lossplot.png', model_name=model_name)

    # save metrics to a file
    metrics = dict(zip(list(range(1, len(epoch_losses)+1)), map(list, zip_longest(epoch_losses, val_losses, val_accuracies, val_precs, val_recs, val_f1s, val_hps, val_hrs, val_hf1s))))

    save_metrics(metrics, checkpoint_dir+'/'+metrics_save_path)

    if checkpoint_and_eval and eval_dataloader != None:
        train_corr = spearmanr(val_losses, val_accuracies)
    else:
        train_corr = None

    save_metrics({'valloss_valacc_correlation' : train_corr}, checkpoint_dir+'/corr.json')

def train_model(model, criterion, optimizer, train_loader, device, num_epochs=25, metrics_save_path='training_metrics.json', checkpoint_dir='.', checkpoint_and_eval=False, eval_loader=None, clip_norm=False, model_name='standard', generate_plots=False, active_learning_mode=False, train_filename=None, eval_filename=None, train_data_path='./', eval_data_path='./', _format_esm=False):

    device = torch.device(device)

    if active_learning_mode:
        le = train_loader.train_dataset.label_encoder
    else:
        le = train_loader.dataset.label_encoder

    if checkpoint_dir is not None and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Initialize containers to store metrics
    epoch_losses = []
    val_losses = []
    epoch_accuracies = []
    epoch_hps = []
    epoch_hrs = []
    epoch_hf1s = []
    val_accuracies = []
    val_precs = []
    val_recs = []
    val_f1s = []
    val_hps = []
    val_hrs = []
    val_hf1s = []

    min_val_loss = 100000000

    model.to(device)
    lowest_val_loss_model = None

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()

        model.train()
        epoch_loss = 0.0
        vtotal = 0
        vcorrect = 0
        num_batches = 0


        print('Loading Embeddings...')
        #train one epoch
        #____________________________________________________________________#
        for i, item in enumerate(tqdm(train_loader, desc='train loop', leave=False)):
            batch_loss = train_step_standard(item, device, optimizer, model, criterion, clip_norm=clip_norm)        
            epoch_loss += batch_loss

            num_batches += 1
        
        # store epoch loss and accuracy
        avg_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_epoch_loss:.4f}')
        #____________________________________________________________________#

        #validate one epoch
        #____________________________________________________________________#
        if checkpoint_and_eval and eval_loader!=None:
            avg_vloss, min_val_loss, avg_vacc, avg_prec, avg_rec, avg_f1, avg_hp, avg_hr, avg_hf1, lowest_val_loss_model = validation_loop_standard(model, 
                    eval_loader, 
                    min_val_loss, 
                    device, 
                    criterion, 
                    checkpoint_dir, 
                    model_name, 
                    le,
                    train_filename, 
                    eval_filename, 
                    train_data_path, 
                    eval_data_path, 
                    train_loader.dataset.emb_out_dir, 
                    _format_esm, 
                    epoch,
                    active_learning_mode=False)

            val_losses.append(avg_vloss)
            val_accuracies.append(avg_vacc)
            val_precs.append(avg_prec)
            val_recs.append(avg_rec)
            val_f1s.append(avg_f1)
            val_hps.append(avg_hp)
            val_hrs.append(avg_hr)
            val_hf1s.append(avg_hf1)

            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_vloss:.4f}, Accuracy: {100*avg_vacc:.2f}%')

        #____________________________________________________________________#

    save_end_of_training_metrics_standard(model, 
            checkpoint_dir, 
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
            eval_loader,
            metrics_save_path)

    if checkpoint_and_eval and eval_loader != None:
        return model, lowest_val_loss_model, (epoch_losses, epoch_accuracies, val_losses, val_accuracies)
    else:
        return model, None, (epoch_losses, epoch_accuracies)

def test_model(model, criterion, test_loader, device, model_name='standard', checkpoint_dir='./', metrics_save_path='test_metrics.csv'):
    device = torch.device(device)
    le = test_loader.train_dataset.label_encoder

    if checkpoint_dir is not None and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    model.eval()
    model.to(device)

    total_loss = 0.0
    all_labels = []
    all_predictions = []
    batch_losses = []

    with torch.no_grad():
        for i, (emb, labels, index) in enumerate(test_loader.get_dataloader()):
            emb, labels = emb.to(device), labels.to(device)

            with autocast(device_type=device.type, dtype=torch.float32):
                outputs = model(emb)
                loss = criterion(outputs, labels)

            # Store batch loss
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            total_loss += batch_loss

            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    #calculate heirarchical metrics
    decoded_labels = le.inverse_transform(all_labels)
    decoded_preds = le.inverse_transform(all_predictions)
     
    decoded_labels = list(map(convert_string_to_list, decoded_labels))
    decoded_preds = list(map(convert_string_to_list, decoded_preds))

    hp = h_precision(decoded_labels, decoded_preds)
    hr = h_recall(decoded_labels, decoded_preds)
    hf1 = h_F1(decoded_labels, decoded_preds)

    #calculate average loss
    avg_loss = total_loss / (i+1)

    #calculate standard metrics
    accuracy = torch.tensor(all_labels).eq(torch.tensor(all_predictions)).sum().item() / len(all_labels)
    report = classification_report(all_labels, all_predictions, output_dict=True, zero_division=0.0)


    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {100*accuracy:.2f}%')

    metrics = {
        'average_loss': avg_loss,
        'accuracy': accuracy,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1-score': report['weighted avg']['f1-score'],
        'h-precision' : hp,
        'h-recall' : hr,
        'h-f1-score' : hf1
    }

    save_metrics(metrics, checkpoint_dir+'/'+metrics_save_path)

