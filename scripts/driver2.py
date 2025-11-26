import torch
import argparse
import os
import numpy as np
import sys
import pickle
import pandas as pd
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from dal_toolbox.utils import seed_everything as seed_everything2
from dal_toolbox.models.deterministic import DeterministicModel

from clean_app.src.CLEAN.utils import seed_everything
from clean_app.src.CLEAN.losses import SupConHardLoss

from setup import get_standard_NN, get_standard_data_module, setup_CLEAN_active_learning_model
#from embedding import get_tokenizer_and_encoder
from train_loop import train_model, test_model
from active_learning_loop import train_standard_model_AL, run_standard_active_learning_simulation
#from active_learning_update import infer_initial_pool_ids, run_CLEAN_active_learning_init_step, run_CLEAN_active_learning_update_step
from dataloader import reformat_emb
from utils import dump_seqs

RANDOM_STATE_SEED = 1234
np.random.seed(RANDOM_STATE_SEED)
torch.manual_seed(RANDOM_STATE_SEED)
seed_everything(seed=RANDOM_STATE_SEED)
seed_everything2(seed=RANDOM_STATE_SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a functional prediction model with various choices')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference', 'simulation'], default='simulation', help='Stage of active learning. Init mode is for the initial points to run the experiment. Update mode is for after the experiment to query the next set of points. Train mode is simply to pre-train the model if using custom data (will not perform any active learning). Pre-training can also be done in init mode by specifying --perform_pretraining.')
    parser.add_argument('--active_type', type=str, choices=['uncertainty_sampling', 'entropy_sampling', 'margin_sampling', 'random_sampling', 'bayesian', 'BALD', 'BADGE', 'typiclust', 'QBC', 'bio-inspired'], required=True, help='Type of embedding to use')
    parser.add_argument('--train_csv_path', type=str, required=True, help='Path to the train CSV file')
    parser.add_argument('--pool_csv_path', type=str, help='Path to the unlabeled pool CSV file')
    parser.add_argument('--embedding_type', type=str, choices=['lookingglassv2', 'evo', 'esm1b', 'esm2', 'protgpt2'], default='esm1b', help='Type of embedding to use. NOT IMPLEMENTED.')
    parser.add_argument('--model_load_path', type=str, help='Path to the pre-trained NN')
    parser.add_argument('--cache_path', type=str, default='/distance_map/', help='Cache path for saving/loading cached embeddings')
    parser.add_argument('--emb_path', type=str, default='/esm_data/', help='Emb path for saving/loading cached embeddings')
    parser.add_argument('--num_learners', type=int, default=3, help='Number of learners (if making a committee)')
    parser.add_argument('--num_queries', type=int, default=4, help='Number of iterations to perform active learning')
    parser.add_argument('--num_instances', type=int, default=32, help='Number of instances (i.e. batch size) to grab from the pool on each iteration')
    parser.add_argument('--valid_csv_path', type=str, default=None, help='Path to the valid CSV file')
    parser.add_argument('--batch_size', type=int, default=6000, help='Batch size used to calculate embeddings and perform training')
    parser.add_argument('--knn', type=int, default=30, help='K-nearest neighbors used in dataloader.')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for the optimizer')
    parser.add_argument('--adaptive_rate', type=int, default=100, help='Number of epochs after which to reinitialize the optimizer')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden layer size')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of epochs to train on when passing training data')
    parser.add_argument('--save_path', type=str, default='./train_dir/', help='Path to args.save_path in which to save the trained model and any specified outputs (e.g. graphs and test dataset )')
    parser.add_argument('--plot_name', type=str, default='TwoLayerClassifier', help='To be used in the titles of plots.')
    parser.add_argument('--generate_plots', action='store_true', help='Specifies if plots should be generated after each query iteration')
    parser.add_argument('--checkpoint_and_eval', action='store_true', help='Whether to save intermediary checkpoints based on evaluation dataset performance')
    parser.add_argument('--precomputed', action='store_true', help='CLEAN ONLY! If the ESM embeddings and distance maps (train) were precomputed for ALL data.')
    parser.add_argument('--use_old_naming_convention', action='store_true', help='CLEAN ONLY! Use CLEAN naming convention for precomputed embeddings and distmaps.')
    parser.add_argument('--perform_pretraining', action='store_true', help='CLEAN ONLY! Use the training data for additional model weight updates (pretraining) using traditional training methods. If false, train data is only for inference.')
    parser.add_argument('--perform_active_learning_pretraining', action='store_true', help='CLEAN ONLY! Use the training data for additional model weight updates (pretraining) using the specified active learning algorithm. If false, train data is only for inference.')
    parser.add_argument('--test_data_path', default=None, help='Path (or paths -- can specify multiple separated by spaces) of the data files to calculate test on at each iteration', nargs='+')

    #setup stuff
    #----------------------------------------------------------------------------------#
    args = parser.parse_args()

    #set the embedding size hardcoded based on transformer training
    if args.embedding_type == 'esm2' or args.embedding_type == 'esm1b':
        input_size = 1280 #esm - change if loading smaller/larger checkpoint
    # the below sizes are not implemented (yet)
    elif args.embedding_type == 'protgpt2':
        input_size = 1280 #gpt
    elif args.embedding_type == 'evo':
        input_size = 512 #evo
    else:
        input_size = 768 #bert models
    
    #loss will be standard cross entropy for classification
    criterion = torch.nn.CrossEntropyLoss()

    if args.active_type in ['bayesian', 'BALD', 'BADGE']:
        mc_dropout = True
    else:
        mc_dropout = False

    #not supported yet FIXME
#    if not args.use_old_naming_convention:
#        tokenizer, encoder = get_tokenizer_and_encoder(embedding_type=args.embedding_type)    
#    else:
#        tokenizer, encoder = None, None #precomputed by ESM-1b

    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.mode == 'train' and not args.perform_pretraining and not args.perform_active_learning_pretraining:
        args.perform_pretraining = True #set standard pretraining as default if one is not specified

    #----------------------------------------------------------------------------------#

    #data loading
    #----------------------------------------------------------------------------------#
    #get training data if specified
    if args.train_csv_path is not None:
        train_data_name, train_data_path, train_datamodule = get_standard_data_module(args.train_csv_path,
                                                                                  batch_size=args.batch_size,
                                                                                  precomputed=True, #FIXME
                                                                                  shuffle=True,
                                                                                  seed=RANDOM_STATE_SEED,
                                                                                  emb_dir=args.emb_path, #not supported yet 
                                                                                  cache_dir=args.cache_path, #not supported yet
                                                                                  knn=args.knn, 
                                                                                  use_old_naming_convention=args.use_old_naming_convention)
        
        if args.perform_pretraining: #need special dataloader
            train_dataloader = train_datamodule.get_dataloader(batch_size=args.batch_size, shuffle=True)

    else:
        train_data_name = train_data_path = train_datamodule = None

    if args.pool_csv_path is not None and args.mode == 'simulation':
        pool_data_name, pool_data_path, pool_datamodule = get_standard_data_module(args.pool_csv_path,
                                                                                  batch_size=args.batch_size,
                                                                                  precomputed=True, #FIXME
                                                                                  shuffle=False, #I dont think it should shuffle because it will mess with the AL
                                                                                  seed=RANDOM_STATE_SEED,
                                                                                  emb_dir=args.emb_path, #not supported yet 
                                                                                  cache_dir=args.cache_path, #not supported yet
                                                                                  knn=args.knn, 
                                                                                  use_old_naming_convention=args.use_old_naming_convention) #not supported yet
    else:
        pool_data_name = pool_data_path = pool_datamodule = None

    if args.valid_csv_path is not None:
        valid_data_name, valid_data_path, valid_datamodule = get_standard_data_module(args.valid_csv_path,
                                                                                  batch_size=args.batch_size,
                                                                                  precomputed=True, #FIXME
                                                                                  shuffle=False,
                                                                                  seed=RANDOM_STATE_SEED,
                                                                                  emb_dir=args.emb_path,
                                                                                  cache_dir=args.cache_path,
                                                                                  knn=args.knn,
                                                                                  use_old_naming_convention=args.use_old_naming_convention)

        eval_dataloader = valid_datamodule.get_dataloader(batch_size=args.batch_size, shuffle=False)

    else:
        valid_data_name = valid_data_path = valid_datamodule = eval_dataloader = None
 
    #get any necessary testing sets
    test_data_list = []
    if args.test_data_path is not None:
        for test_csv_path in args.test_data_path:
            test_tuple = get_standard_data_module(test_csv_path, 
                                                batch_size=args.batch_size,
                                                precomputed=True, #FIXME
                                                shuffle=False,
                                                seed=RANDOM_STATE_SEED,
                                                emb_dir=args.emb_path,
                                                cache_dir=args.cache_path,
                                                knn=args.knn,
                                                use_old_naming_convention=args.use_old_naming_convention)

            test_data_list.append(test_tuple)

    #create label encoder
    full_labels = set(train_datamodule.train_dataset.full_list_labels)
    if valid_datamodule is not None:
        full_labels.update(set(valid_datamodule.train_dataset.full_list_labels))
    if pool_datamodule is not None:
        full_labels.update(set(pool_datamodule.train_dataset.full_list_labels))
    for _,_, test_datamodule in test_data_list:
        full_labels.update(set(test_datamodule.train_dataset.full_list_labels))

    le = LabelEncoder()
    le.fit(list(full_labels))
    
    train_datamodule.train_dataset.label_encoder = le
    train_datamodule.train_dataset.transform_labels()
    if valid_datamodule is not None:
        valid_datamodule.train_dataset.label_encoder = le
        valid_datamodule.train_dataset.transform_labels()
    if pool_datamodule is not None:
        pool_datamodule.train_dataset.label_encoder = le
        pool_datamodule.train_dataset.transform_labels()
    for _,_, test_datamodule in test_data_list:
        test_datamodule.train_dataset.label_encoder = le
        test_datamodule.train_dataset.transform_labels()

    #get model
    model = get_standard_NN(input_size=input_size, 
                         hidden_size=args.hidden_size, 
                         learning_rate=args.learning_rate, 
                         output_embedding_size=len(le.classes_), 
                         pretrained_weights=args.model_load_path,
                         mc_dropout=mc_dropout)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))  # Using Adam optimizer

    #prepares active learning wrapped model
    if args.mode != 'inference' and not args.perform_pretraining:
        model = DeterministicModel(model, loss_fn=criterion, optimizer=optimizer)
        learner = setup_CLEAN_active_learning_model(active_type=args.active_type) #not specific to CLEAN just dont want to change legacy code name

    #need to make sure train and pool ids are precomputed first so we can match order by id
    seq_path = None
    if args.active_type == 'bio-inspired':
        if args.perform_active_learning_pretraining:
            seq_path = dump_seqs(args.train_csv_path, cache_dir=args.cache_path, id_list=train_datamodule.query_dataset.dataset.full_list_ids)
            
            if args.pool_csv_path is not None:
                alternate_seq_path = dump_seqs(args.pool_csv_path, cache_dir=args.cache_path, id_list=pool_datamodule.query_dataset.dataset.full_list_ids) #will be set again before active learning simulation
            learner.seq_path = seq_path
        else:
            seq_path = dump_seqs(args.pool_csv_path, cache_dir=args.cache_path, id_list=pool_datamodule.query_dataset.dataset.full_list_ids)


    #----------------------------------------------------------------------------------#

    #execute train and inference modes
    #----------------------------------------------------------------------------------#

    #-----------------------performing pretraining-------------------------------------#
    if args.mode != 'inference':
        if args.perform_pretraining:

            #------------------------------------default pretraining---------------------------------#
            model, lowest_loss_model, _ = train_model(model=model, 
                                                    criterion=criterion, 
                                                    optimizer=optimizer, 
                                                    train_loader=train_dataloader,
                                                    device=device,
                                                    num_epochs=args.num_train_epochs, 
                                                    checkpoint_dir=args.save_path+'/train_dir/', 
                                                    checkpoint_and_eval=args.checkpoint_and_eval, 
                                                    eval_loader=eval_dataloader, 
                                                    generate_plots=args.generate_plots, 
                                                    model_name=args.plot_name,
                                                    train_filename=train_data_name,
                                                    eval_filename=valid_data_name,
                                                    train_data_path=train_data_path,
                                                    eval_data_path=valid_data_path,
                                                    _format_esm=args.use_old_naming_convention)

        #---------------------------active learning inspired pretraining-------------------------#
        elif args.perform_active_learning_pretraining:
            model, lowest_loss_model, _ = train_standard_model_AL(model, 
                    criterion, 
                    optimizer, 
                    learner, 
                    train_datamodule, 
                    eval_dataloader=eval_dataloader, 
                    num_epochs=args.num_train_epochs, 
                    batch_size=args.batch_size, 
                    generate_plots=args.generate_plots, 
                    save_path=args.save_path+'/train_dir/', 
                    learning_rate=args.learning_rate, 
                    checkpoint_and_eval=args.checkpoint_and_eval, 
                    train_data_path=train_data_path, 
                    eval_data_path=valid_data_path, 
                    train_filename=train_data_name, 
                    eval_filename=valid_data_name, 
                    emb_dir=args.emb_path, 
                    cache_dir=args.cache_path, 
                    knn=args.knn, 
                    _format_esm=args.use_old_naming_convention,
                    model_name=args.plot_name)

        #-------------------------------evaluate pretraining performance--------------------------------#
        if args.checkpoint_and_eval and (args.perform_active_learning_pretraining or args.perform_pretraining) and lowest_loss_model != None:
            for (test_data_name, test_data_path, test_data) in test_data_list:
                test_model(lowest_loss_model, 
                    criterion, 
                    test_data, 
                    device, 
                    model_name=args.plot_name,
                    checkpoint_dir=args.save_path+'/train_dir/', 
                    metrics_save_path=test_data_name+'_lowest_loss_metrics.json')

    #-----------------------------------perform inference---------------------------------------#
    if args.perform_pretraining or args.perform_active_learning_pretraining or args.mode == 'inference':
        if args.mode == 'inference':
            checkpoint_dir = args.save_path
        else:
            checkpoint_dir = args.save_path+'/train_dir/'

        for (test_data_name, test_data_path, test_data) in test_data_list:
            test_model(model, 
                    criterion, 
                    test_data, 
                    device, 
                    model_name=args.plot_name,
                    checkpoint_dir=checkpoint_dir, 
                    metrics_save_path=test_data_name+'_metrics.json')

    #------------------------exit early from train/inference modes------------------------------#
    if args.mode == 'train' or args.mode == 'inference':
        quit()

    #----------------------------begin active learning modes-----------------------------------#

    #specific case for bio-inspired learner
    if args.perform_active_learning_pretraining and args.active_type == 'bio-inspired':
        learner.seq_path = alternate_seq_path #update where to pull sequences for computing smith waterman distance

    #need to now convert the model to AL type
    if args.perform_pretraining:
        model = DeterministicModel(model, loss_fn=criterion, optimizer=optimizer)
        learner = setup_CLEAN_active_learning_model(active_type=args.active_type) #this isnt specific to CLEAN just dont want to mess up legacy code by changing the name

    #execute simulation mode
    #-------------------------------------------------------------------------------------------#

    if args.mode == 'simulation': 
        if args.generate_plots:
            id_list = np.array(pickle.load(open(pool_data_path+'/'+args.cache_path+'/'+pool_data_name+'_ids.pkl', 'rb')))
            ec_list = np.array(pickle.load(open(pool_data_path+'/'+args.cache_path+'/'+pool_data_name+'_ecs.pkl', 'rb')))
            pool_embeddings = pickle.load(open(pool_data_path+'/'+args.cache_path+'/'+pool_data_name+'_esm.pkl', 'rb')).to(device=device, dtype=torch.float32).cpu()

            pca = PCA(n_components=2).fit(pool_embeddings)
            plot_tuple = (id_list, pool_embeddings, ec_list)
        else:
            pca=None
            plot_tuple=None

        #running active learning simulation post-training
        model, lowest_loss_model, _ = run_standard_active_learning_simulation(model, 
                                   criterion, 
                                   optimizer, 
                                   learner, 
                                   train_datamodule, 
                                   pool_datamodule, 
                                   eval_dataloader=eval_dataloader, 
                                   test_data_list=test_data_list,
                                   n_instances=args.num_instances, 
                                   n_queries=args.num_queries, 
                                   generate_plots=args.generate_plots, 
                                   save_path=args.save_path, 
                                   learning_rate=args.learning_rate, 
                                   checkpoint_and_eval=args.checkpoint_and_eval, 
                                   emb_dir=args.emb_path,
                                   cache_dir=args.cache_path,
                                   train_data_path=train_data_path, 
                                   eval_data_path=valid_data_path, 
                                   pool_data_path=pool_data_path, 
                                   train_filename=train_data_name, 
                                   eval_filename=valid_data_name, 
                                   pool_filename=pool_data_name, 
                                   pca=pca, 
                                   plot_tuple=plot_tuple, 
                                   model_name=args.plot_name,
                                   _format_esm=args.use_old_naming_convention)

        if args.checkpoint_and_eval and lowest_loss_model != None:
            for (test_data_name, test_data_path, test_data) in test_data_list:
                test_model(lowest_loss_model, 
                    criterion, 
                    test_data, 
                    device, 
                    model_name=args.plot_name,
                    checkpoint_dir=args.save_path,
                    metrics_save_path=test_data_name+'_lowest_loss_metrics.json')

        for (test_data_name, test_data_path, test_data) in test_data_list:
            test_model(model, 
                    criterion, 
                    test_data, 
                    device, 
                    model_name=args.plot_name,
                    checkpoint_dir=args.save_path,
                    metrics_save_path=test_data_name+'_metrics.json')

        quit() #exit after simulation



################################ NOT IMPLEMENTED BELOW HERE ####################################################
# Would need to add a lot of code/debugging around data loaders and exact implementation

'''
    #execute init mode
    #-------------------------------------------------------------------------------------------#

    if args.mode == 'init':
        #-------------------------------------infer labels-----------------------------------------#
        if not args.labeled:
            new_pool_labels_path = infer_initial_pool_ids(train_data_name, 
                                                            pool_data_name, 
                                                            model, 
                                                            train_data_path=train_data_path, 
                                                            pool_data_path=pool_data_path, 
                                                            pretrained_weights=args.model_load_path, 
                                                            save_path=args.save_path, 
                                                            train_emb=reformat_emb(train_datamodule.emb, train_datamodule.ec_id_dict),
                                                            maxsep=maxsep, 
                                                            emb_dir=args.emb_path, 
                                                            use_old_naming_convention=args.use_old_naming_convention)
        else:
            new_pool_labels_path = args.pool_csv_path
    
        if args.generate_plots:
            new_pool_data_name = new_pool_labels_path.split('/')[-1].split('.csv')[0]

            if args.use_old_naming_convention:
                id_list = np.array(pickle.load(open(pool_data_path+'/'+args.cache_path+'/'+new_pool_data_name+'_ids.pkl', 'rb')))
                ec_list = np.array(pickle.load(open(pool_data_path+'/'+args.cache_path+'/'+new_pool_data_name+'_ecs.pkl', 'rb')))
                pool_embeddings = pickle.load(open(pool_data_path+'/'+args.cache_path+'/'+new_pool_data_name+'_esm.pkl', 'rb')).to(device=device, 
                        dtype=torch.float32).cpu()
            else:
                id_list = np.array(pickle.load(open(pool_data_path+'/'+args.cache_path+'/'+new_pool_data_name+'_ids.pkl', 'rb')))
                ec_list = np.array(pickle.load(open(pool_data_path+'/'+args.cache_path+'/'+new_pool_data_name+'_ecs.pkl', 'rb')))
                pool_embeddings = pickle.load(open(pool_data_path+'/'+args.cache_path+'/'+new_pool_data_name+'_emb.pkl', 'rb')).to(device=device, 
                        dtype=torch.float32).cpu()

            pca = PCA(n_components=2).fit(pool_embeddings)
            le = LabelEncoder().fit(ec_list)
            plot_tuple = (id_list, pool_embeddings, ec_list)
        else:
            pca=None
            plot_tuple=None
            le = None

        #------------------------------fix dataloader to have infered labels----------------------------#
        pool_data_name, pool_data_path, pool_datamodule = get_full_data_module(new_pool_labels_path,
                                                                            batch_size=1, #previously was 1 but should consider changing
                                                                            precomputed=True,
                                                                            shuffle=False,
                                                                            seed=RANDOM_STATE_SEED,
                                                                            emb_dir=args.emb_path, #not supported yet 
                                                                            cache_dir=args.cache_path, #not supported yet
                                                                            mutate_for_training=True,
                                                                            knn=args.knn, 
                                                                            loss=args.loss,
                                                                            compute_distmaps_only=True,
                                                                            train_mode=False,
                                                                            use_old_naming_convention=args.use_old_naming_convention) #not supported yet

        #-----------------------------setup code to plot queried points---------------------------------#
        if args.generate_plots:
            id_list = np.array(pickle.load(open(pool_data_path+args.cache_path+pool_data_name+'_ids.pkl', 'rb')))
            ec_list = np.array(pickle.load(open(pool_data_path+args.cache_path+pool_data_name+'_ecs.pkl', 'rb')))
            pool_embeddings = pickle.load(open(pool_data_path+args.cache_path+pool_data_name+'_esm.pkl', 'rb')).to(device=device, dtype=torch.float32).cpu()
            
            pca = PCA(n_components=2).fit(pool_embeddings)
            le = LabelEncoder().fit(ec_list)
            plot_tuple = (id_list, pool_embeddings, ec_list)
        else:
            pca=None
            plot_tuple=None
            le = None

        #---------------------------query initial (round 0) points and save-----------------------------#
        run_CLEAN_active_learning_init_step(model, 
                                            learner, 
                                            train_datamodule=train_datamodule,
                                            pool_datamodule=pool_datamodule, 
                                            train_data_path=train_data_path, 
                                            pool_data_path=pool_data_path, 
                                            train_filename=train_data_name,
                                            pool_filename=pool_data_name,
                                            n_instances=args.num_instances, 
                                            save_path=args.save_path, 
                                            generate_plots=args.generate_plots, 
                                            pca=pca, 
                                            label_encoder=le, 
                                            plot_tuple=plot_tuple,
                                            AL_round=args.round_number,
                                            correlation_file=args.correlation_file_path)

        #------------------------write sequences associated with queries------------------------------#
        if args.write_sequences:
            infer_df = pd.read_csv(args.save_path+'/infer_ids.tsv', sep='\t', header=0)
            seq_df = pd.read_csv(args.pool_csv_path, sep='\t', header=0)

            seqs = []
            for entry in infer_df['Entry']:
                seqs.append(seq_df.loc[seq_df['Entry'] == entry, 'Sequence'].item())

            assert(len(seqs) == len(infer_df))
            infer_df['Sequence'] = seqs

            infer_df.to_csv(args.save_path+'/infer_ids.tsv', sep='\t', header=True, index=False)

    #execute update mode
    #---------------------------------------------------------------------------------------------#
    elif args.mode == 'update':

        if args.distmap_recomputed_path is not None:
            distmap_path = args.distmap_recomputed_path
        else:
            distmap_path = pool_data_path+'/'+args.cache_path+'/'

        #-----------------------------setup code to plot queried points---------------------------------#
        if args.generate_plots:
            if args.use_old_naming_convention:
                id_list = np.array(pickle.load(open(distmap_path+pool_data_name+'_ids.pkl', 'rb')))
                ec_list = np.array(pickle.load(open(distmap_path+pool_data_name+'_ecs.pkl', 'rb')))
                pool_embeddings = pickle.load(open(distmap_path+pool_data_name+'_esm.pkl', 'rb')).to(device=device, dtype=torch.float32).cpu()
            else:
                id_list = np.array(pickle.load(open(distmap_path+pool_data_name+'_ids.pkl', 'rb')))
                ec_list = np.array(pickle.load(open(distmap_path+pool_data_name+'_ecs.pkl', 'rb')))
                pool_embeddings = pickle.load(open(distmap_path+pool_data_name+'_emb.pkl', 'rb')).to(device=device, dtype=torch.float32).cpu()

            pca = PCA(n_components=2).fit(pool_embeddings)
            le = LabelEncoder().fit(ec_list)
            plot_tuple = (id_list, pool_embeddings, ec_list)
        else:
            pca=None
            plot_tuple=None
            le = None

        #------------------------------update points from previous round-------------------------------#
        if args.model_load_path is not None:
            _model_name = args.model_load_path.rpartition('/')[-1].split('.pth')[0]

        new_train_datamodule, new_pool_datamodule, new_plot_tuple = run_CLEAN_active_learning_update_step(model,
                                              criterion,
                                              optimizer,
                                              learner,
                                              train_datamodule=train_datamodule,
                                              pool_datamodule=pool_datamodule,
                                              eval_dataloader=eval_dataloader,
                                              test_data_list=test_data_list,
                                              save_path=args.save_path,
                                              adaptive_rate=args.adaptive_rate,
                                              learning_rate=args.learning_rate,
                                              checkpoint_and_eval=args.checkpoint_and_eval,
                                              generate_plots=args.generate_plots,
                                              train_data_path=train_data_path,
                                              eval_data_path=valid_data_path,
                                              pool_data_path=pool_data_path,
                                              train_filename=train_data_name,
                                              eval_filename=valid_data_name,
                                              pool_filename=pool_data_name,
                                              pca=pca,
                                              label_encoder=le,
                                              plot_tuple=plot_tuple,
                                              save_recomputed_embeddings=args.save_recomputed_embeddings,
                                              cache_dir=args.cache_path, 
                                              emb_dir=args.emb_path, 
                                              maxsep=maxsep, 
                                              loss=args.loss, 
                                              AL_round=args.round_number, 
                                              use_old_naming_convention=args.use_old_naming_convention,
                                              model_name=_model_name)

        #---------------------------perform second init step to repeat the active learning------------------------------#
        if args.update_and_requery:
            if args.generate_plots:
                id_list, pool_embeddings, ec_list = new_plot_tuple

                pca = PCA(n_components=2).fit(pool_embeddings)
                le = LabelEncoder().fit(ec_list)
            
            run_CLEAN_active_learning_init_step(model,
                                                learner,
                                                train_datamodule=new_train_datamodule,
                                                pool_datamodule=new_pool_datamodule,
                                                train_data_path=train_data_path,
                                                pool_data_path=pool_data_path,
                                                train_filename=train_data_name,
                                                pool_filename=pool_data_name,
                                                n_instances=args.num_instances,
                                                save_path=args.save_path,
                                                generate_plots=args.generate_plots,
                                                pca=pca,
                                                label_encoder=le,
                                                plot_tuple=new_plot_tuple,
                                                AL_round=args.round_number)
            
        #------------------------write sequences associated with queries------------------------------#
        if args.write_sequences:
            infer_df = pd.read_csv(args.save_path+'/infer_ids.tsv', sep='\t', header=0)
            seq_df = pd.read_csv(args.pool_csv_path, sep='\t', header=0)

            seqs = []
            for entry in infer_df['Entry']:
                seqs.append(seq_df.loc[seq_df['Entry'] == entry, 'Sequence'].item())

            assert(len(seqs) == len(infer_df))
            infer_df['Sequence'] = seqs

            infer_df.to_csv(args.save_path+'/infer_ids.tsv', sep='\t', header=True, index=False)
'''
