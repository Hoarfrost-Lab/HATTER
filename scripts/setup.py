import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import gc
import pandas as pd
import pickle
import io
import os
from tqdm.auto import tqdm

from dal_toolbox.models.deterministic import DeterministicModel
from dal_toolbox.models.utils.mcdropout import ConsistentMCDropout
from dal_toolbox.models.deterministic.simplenet import SimpleNet as TwoLayerClassifier
from dal_toolbox.models.mc_dropout.simplenet import SimpleMCNet as TwoLayerMCClassifier

from clean_app.src.CLEAN.utils import get_ec_id_dict, ensure_dirs, csv_to_fasta, mutate_single_seq_ECs, dump_info, load_embeddings, retrieve_esm1b_embedding
from clean_app.src.CLEAN.distance_map import get_dist_map

from models import CLEANLayerNormNet, CLEANBatchNormNet, CLEANInstanceNormNet, CLEANStandardNet, CLEANConvNet, CLEANSNGPNet
from models import CLEANLayerNormNetMC, CLEANBatchNormNetMC, CLEANInstanceNormNetMC, CLEANStandardNetMC, CLEANConvNetMC, CLEANSNGPNetMC
from dataloader import create_CLEAN_dataloader, create_CLEAN_active_learning_data_module, create_CLEAN_active_learning_data_module_w_update, create_standard_active_learning_data_module
from embedding import generate_embeddings
from utils import convert_to_AA, read_fasta, create_hierarchy_dict
from wrappers import get_sampling_active_learner, get_badge_active_learner, get_bald_active_learner, get_bayesian_active_learner, get_committee_active_learner, get_clust_active_learner, get_bio_active_learner

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DeterministicCLEANModel(DeterministicModel):
    def __init__(
            self,
            model: nn.Module,
            loss_fn: nn.Module = nn.CrossEntropyLoss(),
            optimizer: torch.optim.Optimizer = None,
            lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
            train_metrics: dict = None,
            val_metrics: dict = None,
            scheduler_interval = 'epoch',
            bayesian=False
    ):
        super().__init__(model, loss_fn=loss_fn, optimizer=optimizer, lr_scheduler=lr_scheduler, train_metrics=train_metrics, val_metrics=val_metrics, scheduler_interval=scheduler_interval)

        self.bayesian = bayesian

    # TODO(dhuseljic): Discuss
    @torch.inference_mode()
    def get_logits(self, *args, **kwargs):
        if 'device' not in kwargs:
            kwargs['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not hasattr(self.model, 'get_logits'):
            raise NotImplementedError('The `get_logits` method is not implemented.')
        
        if self.bayesian==True:
            return self.model.get_logits_bayesian(*args, **kwargs)

        return self.model.get_logits(*args, **kwargs)
    
    def training_step(self, batch):
        anchor, pos, neg = batch

        anchor_out = model(anchor)
        pos_out = model(pos)
        neg_out = model(neg)

        loss = self.loss_fn(anchor_out, pos_out, neg_out)
        self.log('train_loss', loss, prog_bar=True, batch_size=len(anchor))

        return loss

    def validation_step(self, batch, batch_idx):
        anchor, pos, neg = batch

        anchor_out = model(anchor)
        pos_out = model(pos)
        neg_out = model(neg)

        loss = self.loss_fn(anchor_out, pos_out, neg_out)
        self.log('val_loss', loss, prog_bar=True, batch_size=len(anchor))

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        anchor, pos, neg = batch

        anchor_out = model(anchor)
        pos_out = model(pos)
        neg_out = model(neg)

        return anchor_out, pos_out, neg_out

def get_CLEAN_NN(model_name='layernorm', dropout_rate=0.001, input_size=1280, hidden_size=512, output_embedding_size=128, learning_rate=0.01, momentum=0.98, pretrained_weights=None, device='cpu', dtype=torch.float32, mc_dropout=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_name == 'standard':
        if mc_dropout:
            model = CLEANStandardNetMC(input_dim=input_size, hidden_dim=hidden_size, out_dim=output_embedding_size, device=device, dtype=dtype, drop_out=dropout_rate)
        else:
            model = CLEANStandardNet(input_dim=input_size, hidden_dim=hidden_size, out_dim=output_embedding_size, device=device, dtype=dtype, drop_out=dropout_rate)
    elif model_name == 'cnn':
        if mc_dropout:
            model = CLEANConvNetMC(input_dim=input_size, hidden_dim=hidden_size, out_dim=output_embedding_size, device=device, dtype=dtype, drop_out=dropout_rate)
        else:
            model = CLEANConvNet(input_dim=input_size, hidden_dim=hidden_size, out_dim=output_embedding_size, device=device, dtype=dtype, drop_out=dropout_rate)
    elif model_name == 'sngp':
        if mc_dropout:
            model = CLEANSNGPNetMC(input_dim=input_size, hidden_dim=hidden_size, out_dim=output_embedding_size, device=device, dtype=dtype, drop_out=dropout_rate)
        else:
            model = CLEANSNGPNet(input_dim=input_size, hidden_dim=hidden_size, out_dim=output_embedding_size, device=device, dtype=dtype, drop_out=dropout_rate)
    elif model_name == 'layernorm':
        if mc_dropout:
            model = CLEANLayerNormNetMC(input_dim=input_size, hidden_dim=hidden_size, out_dim=output_embedding_size, device=device, dtype=dtype, drop_out=dropout_rate)
        else:
            model = CLEANLayerNormNet(input_dim=input_size, hidden_dim=hidden_size, out_dim=output_embedding_size, device=device, dtype=dtype, drop_out=dropout_rate)
    elif model_name == 'batchnorm':
        if mc_dropout:
            model = CLEANBatchNormNetMC(input_dim=input_size, hidden_dim=hidden_size, out_dim=output_embedding_size, device=device, dtype=dtype, drop_out=dropout_rate)
        else:
            model = CLEANBatchNormNet(input_dim=input_size, hidden_dim=hidden_size, out_dim=output_embedding_size, device=device, dtype=dtype, drop_out=dropout_rate)
    elif model_name == 'instancenorm':
        if mc_dropout:
            model = CLEANInstanceNormNetMC(input_dim=input_size, hidden_dim=hidden_size, out_dim=output_embedding_size, device=device, dtype=dtype, drop_out=dropout_rate)
        else:
            model = CLEANInstanceNormNet(input_dim=input_size, hidden_dim=hidden_size, out_dim=output_embedding_size, device=device, dtype=dtype, drop_out=dropout_rate)

    if pretrained_weights is not None:
        model.load_state_dict(torch.load(pretrained_weights))

    return model

def setup_CLEAN_active_learning_model(active_type='uncertainty_sampling', seq_path=None):
    #set the proper type of active learning
    if active_type == 'uncertainty_sampling':
        learner = get_sampling_active_learner()
    elif active_type == 'entropy_sampling':
        learner = get_sampling_active_learner(query_strategy='entropy')
    elif active_type == 'margin_sampling':
        learner = get_sampling_active_learner(query_strategy='margin')
    elif active_type == 'random_sampling':
        learner = get_sampling_active_learner(query_strategy='random')
    elif active_type == 'bayesian':
        learner = get_bayesian_active_learner()
    elif active_type == 'BALD':
        learner = get_bald_active_learner()
    elif active_type == 'BADGE':
        learner = get_badge_active_learner()
    elif active_type == 'typiclust':
        learner = get_clust_active_learner()
    elif active_type == 'QBC':
        learner = get_committee_active_learner()
    elif active_type == 'bio-inspired':
        learner = get_bio_active_learner(seq_path)
        
    return learner

def get_full_data_module(csv_path, batch_size=64, precomputed=False, seed=10986223, train_mode=False, train_datamodule=None, result_of_experiment_path=None, compute_distmaps_only=False, mutate_for_training=True, loss='triplet', _format_esm=True, emb_dir='./', cache_dir='./', knn=30, use_old_naming_convention=False, shuffle=False, percentage=None):

    if result_of_experiment_path is not None:
        result_df = pd.read_csv(result_of_experiment_path, sep='\t', header=0)
        ids_to_update = result_df['Entry'].to_list()
        ecs_to_update = result_df['EC number'].to_list()
        result_of_experiment = result_df['Result'].to_list()

        assert(len(ids_to_update) == len(result_of_experiment))
        assert(len(ecs_to_update) == len(result_of_experiment))

    data_name = csv_path.split('/')[-1].split('.')[0]
    data_path = csv_path.rpartition('/')[0]

    if train_datamodule is not None:
        al_data = create_CLEAN_active_learning_data_module_w_update(data_name, 
                train_datamodule, 
                ids_to_update=ids_to_update, 
                ecs_to_update=ecs_to_update,
                result_of_experiment=result_of_experiment, 
                batch_size=batch_size, 
                precomputed_emb=precomputed, 
                compute_distmaps_only=compute_distmaps_only, 
                path=data_path, 
                seed=seed, 
                mutate_for_training=mutate_for_training, 
                loss=loss,
                knn=knn,
                emb_dir=emb_dir,
                cache_dir=cache_dir,
                shuffle=shuffle,
                _format_esm=_format_esm,
                use_old_naming_convention=use_old_naming_convention, 
                percentage=percentage)
        
        #al_data.pool_mode()
    
    else:
        al_data = create_CLEAN_active_learning_data_module(data_name, 
                batch_size=batch_size, 
                precomputed_emb=precomputed, 
                compute_distmaps_only=compute_distmaps_only, 
                path=data_path, 
                seed=seed, 
                mutate_for_training=mutate_for_training, 
                loss=loss, 
                knn=knn,
                emb_dir=emb_dir,
                cache_dir=cache_dir,
                shuffle=shuffle,
                _format_esm=_format_esm,
                use_old_naming_convention=use_old_naming_convention,
                percentage=percentage)

    if train_mode:
        al_data.train_mode()

    return (data_name, data_path, al_data)

def get_validation_only_data_module(csv_path, precomputed=False, emb_dir='./'):
    data_name = csv_path.split('/')[-1].split('.csv')[0]
    data_path = csv_path.rpartition('/')[0]

    if not precomputed:
        csv_to_fasta(data_path+'/'+data_name+'.csv', data_path+'/'+data_name+'.fasta')
        retrieve_esm1b_embedding(data_name, path=data_path, emb_out_dir=emb_dir)

    al_data, _ = get_ec_id_dict(csv_path) #al_valid is actually just ec map and will be loaded in the train loop

    return (data_name, data_path, al_data)


###################ORIGINAL - NO CLEAN##########################

def get_standard_data_module(csv_path, batch_size=64, precomputed=False, seed=10986223, train_mode=False, train_datamodule=None, _format_esm=True, emb_dir='./', cache_dir='./', knn=30, use_old_naming_convention=False, shuffle=False):

    data_name = csv_path.split('/')[-1].split('.')[0]
    data_path = csv_path.rpartition('/')[0]

    al_data = create_standard_active_learning_data_module(data_name, 
                batch_size=batch_size, 
                precomputed_emb=precomputed, 
                path=data_path, 
                seed=seed, 
                knn=knn,
                emb_dir=emb_dir,
                cache_dir=cache_dir,
                shuffle=shuffle,
                _format_esm=_format_esm,
                use_old_naming_convention=use_old_naming_convention)

    return (data_name, data_path, al_data)

class MyTwoLayerMCClassifier(TwoLayerMCClassifier):
    def __init__(self,
            num_classes: int,
            dropout_rate: int = .2,
            feature_dim: int = 128,
            in_dimension = 1280
        ):
        super().__init__(num_classes=num_classes, dropout_rate=dropout_rate, feature_dim=feature_dim)

        self.in_dimension=in_dimension
        self.feature_dim=feature_dim
        self.num_classes=num_classes
        self.use_mc_dropout=True
        self.first = nn.Linear(in_dimension, feature_dim)
        self.first_dropout = ConsistentMCDropout(dropout_rate)
        self.hidden = nn.Linear(feature_dim, feature_dim)
        self.hidden_dropout = ConsistentMCDropout(dropout_rate)
        self.last = nn.Linear(feature_dim, num_classes)
        self.act = nn.ReLU()
        self.n_passes=5

    def forward(self, x):
        if not self.use_mc_dropout:
            # deterministic forward
            x = self.act(self.first(x))
            x = F.dropout(x, p=self.first_dropout.p, training=self.training)
            x = self.act(self.hidden(x))
            x = F.dropout(x, p=self.hidden_dropout.p, training=self.training)
            out = self.last(x)
            return out
        else:
            # MC dropout forward
            x = self.act(self.first(x))
            x = self.first_dropout(x)
            x = self.act(self.hidden(x))
            x = self.hidden_dropout(x)
            out = self.last(x)
            return out

    # override eval() to disable MC dropout automatically
    def eval(self):
        super().eval()           # standard PyTorch behavior
        self.use_mc_dropout = False
        return self

    # override train() to enable MC dropout automatically
    def train(self, mode: bool = True):
        super().train(mode)      # standard PyTorch behavior
        if mode:
            self.use_mc_dropout = True
        return self


def get_standard_NN(dropout_rate=0.001, input_size=1280, hidden_size=512, output_embedding_size=128, learning_rate=0.01, momentum=0.98, pretrained_weights=None, mc_dropout=False):

    if mc_dropout:
        model = MyTwoLayerMCClassifier(in_dimension=input_size, feature_dim=hidden_size, num_classes=output_embedding_size, dropout_rate=dropout_rate)
    else:
        model = TwoLayerClassifier(in_dimension=input_size, feature_dim=hidden_size, num_classes=output_embedding_size, dropout_rate=dropout_rate)

    if pretrained_weights is not None:
        model.load_state_dict(torch.load(pretrained_weights))

    return model
