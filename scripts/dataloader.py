from torch.utils.data import Dataset, DataLoader
import torch
import shutil
import random
import pickle
from lightning.pytorch.utilities import rank_zero_warn

from clean_app.src.CLEAN.dataloader import mine_hard_negative, Triplet_dataset_with_mine_EC, MultiPosNeg_dataset_with_mine_EC, mine_negative, random_positive
from clean_app.src.CLEAN.utils import ensure_dirs, csv_to_fasta, retrieve_esm1b_embedding, mutate_single_seq_ECs, compute_emb_distance, get_ec_id_dict, mutate_incorrect_seq_ECs, format_esm
from clean_app.src.CLEAN.distance_map import get_dist_map

from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.utils import setup_rng

from dataset import Heirarchical_dataset_with_mine_EC, Active_learning_triplet_dataset_with_mine_EC, Active_learning_supconh_dataset_with_mine_EC, Active_learning_heirarchical_dataset_with_mine_EC, MyCLEANQueryDataset, StandardDataset, MyQueryDataset


class CLEANActiveLearningDataModule(ActiveLearningDataModule):
    def __init__(
            self,
            train_dataset: Dataset,
            query_dataset: Dataset = None,
            train_batch_size: int = 64,
            predict_batch_size: int = 256,
            seed: int = None,
            fill_train_loader_batch: bool = True,
            collator=None, 
            ec_id=None,
            id_ec=None,
            emb=None,
            dist_map=None, 
            ec_id_dict=None
    ):
        super().__init__(train_dataset)
        self.train_dataset = train_dataset
        self.query_dataset = MyCLEANQueryDataset(dataset=train_dataset)
        self.train_batch_size = train_batch_size
        self.predict_batch_size = predict_batch_size
        self.fill_train_loader_batch = fill_train_loader_batch
        self.collator = collator
        self.ec_id=ec_id
        self.id_ec=id_ec
        self.emb=emb
        self.dist_map=dist_map
        self.ec_id_dict=ec_id_dict


        if query_dataset is None:
            rank_zero_warn('Using train_dataset for queries. Ensure that there are no augmentations used.')

        self.rng = setup_rng(seed)
        self.unlabeled_indices = list(range(len(self.query_dataset)))
        self.labeled_indices = []

    def get_CLEAN_dataloader(self, batch_size=6000, shuffle=True):
        params = {'batch_size': batch_size, 'shuffle': shuffle}
        loader = DataLoader(self.train_dataset, **params)
        return loader

    def train_mode(self):
        self.labeled_indices = self.unlabeled_indices

    def pool_mode(self): #FIXME
        unlabeled_indices = self.unlabeled_indices 
        indices = [self.query_dataset.full_list.index(_id) for _id in self.query_dataset.dataset.ids_for_update]
        self.update_annotations(indices)

    def reset(self):
        self.unlabeled_indices = list(range(len(self.query_dataset)))
        self.labeled_indices = []

    def update_all(self, ec_id, ec_id_dict, id_ec, emb, dist_map, neg, train_ec_id=None, train_id_ec=None):
        self.ec_id = ec_id
        self.ec_id_dict = ec_id_dict
        self.id_ec = id_ec
        self.emb = emb
        self.dist_map = dist_map

        if train_ec_id is not None and train_id_ec is not None:
            self.train_dataset.update_ecs_and_ids(id_ec, ec_id, ec_id_dict, neg, train_ec_id=train_ec_id, train_id_ec=train_id_ec)
        else:
            self.train_dataset.update_ecs_and_ids(id_ec, ec_id, ec_id_dict, neg)
        
        self.query_dataset = MyCLEANQueryDataset(dataset=self.train_dataset)

        self.unlabeled_indices = list(range(len(self.query_dataset)))

def reformat_emb(emb, ec_id_dict):
    tmp_list = []

    for ec in ec_id_dict.keys():
        ids_to_append = list(ec_id_dict[ec])

        for s_id in ids_to_append:
            tmp_list.append(emb[s_id])

    return torch.stack(tmp_list, dim=0)

def update_ec_id_dicts(train_datamodule, pool_datamodule, ec_ids, seq_ids, train_path='./', pool_path='./', train_name='split100', pool_name='pool', knn=30, update_mode=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype=torch.float32
    
    pool_ec_id = pool_datamodule.ec_id
    pool_ec_id_dict = pool_datamodule.ec_id_dict
    pool_id_ec = pool_datamodule.id_ec
    pool_emb = pool_datamodule.emb

    train_ec_id = train_datamodule.ec_id
    train_ec_id_dict = train_datamodule.ec_id_dict
    train_id_ec = train_datamodule.id_ec
    train_emb = train_datamodule.emb

    seq_set = set([])
    for ec, seq in zip(ec_ids, seq_ids):
        
        if seq in seq_set: #duplicate id skip
            continue

        seq_set.add(seq)

        #update id_ec by removing id from pool and adding to train
        ec_list = pool_id_ec.pop(seq)
        train_id_ec[seq] = ec_list

        for _ec in set(ec_list):
            #remove the used sequence from its pool ec and add to train
            pool_ec_id[_ec].pop(pool_ec_id[_ec].index(seq))
            if len(pool_ec_id[_ec]) == 0:
                pool_ec_id.pop(_ec)

            pool_ec_id_dict[_ec].discard(seq)
            if len(pool_ec_id_dict[_ec]) == 0:
                pool_ec_id_dict.pop(_ec)

            if _ec not in train_ec_id_dict.keys():
                train_ec_id[_ec] = [seq]
                train_ec_id_dict[_ec] = set([seq])
            else:
                train_ec_id[_ec].append(seq)
                train_ec_id_dict[_ec].add(seq)

        #update the embeddings
        item = pool_emb.pop(seq)
        train_emb[seq] = item

    #need to also copy embeddings to train location so they are available in next pass
    for _id in seq_set:
        shutil.copy2(pool_path+'/'+pool_datamodule.train_dataset.emb_out_dir+'/'+_id+'.pt', train_path+'/'+train_datamodule.train_dataset.emb_out_dir+'/'+_id+'.pt')

    train_fasta_file = mutate_single_seq_ECs(id_ec=train_id_ec, ec_id=train_ec_id_dict, path=train_path, name=train_name, emb_out_dir=train_datamodule.train_dataset.emb_out_dir)
    retrieve_esm1b_embedding(train_fasta_file, path=train_path, emb_out_dir=train_datamodule.train_dataset.emb_out_dir)
    train_dist_map = get_dist_map(train_ec_id_dict, reformat_emb(train_emb, train_ec_id_dict), device, dtype)
#    train_negative = mine_hard_negative(train_dist_map, knn)

    pool_fasta_file = mutate_single_seq_ECs(id_ec=pool_id_ec, ec_id=pool_ec_id_dict, path=pool_path, name=pool_name, emb_out_dir=pool_datamodule.train_dataset.emb_out_dir)
    retrieve_esm1b_embedding(pool_fasta_file, path=pool_path, emb_out_dir=pool_datamodule.train_dataset.emb_out_dir)
    pool_dist_map = get_dist_map(pool_ec_id_dict, reformat_emb(pool_emb, pool_ec_id_dict), device, dtype)
#    pool_negative = mine_hard_negative(pool_dist_map, knn)

    #temporary to see if bug is fixed #FIXME
    batch_size=6000
    shuffle=False
    return_anchor=False
    loss='triplet'
    _format_esm=True
    emb_dir='/esm_data/'
    seed=1234

    new_train_dataset = create_CLEAN_dataloader(train_dist_map, 
            train_id_ec, 
            train_ec_id, 
            train_ec_id_dict,
            batch_size=batch_size, 
            shuffle=shuffle, 
            knn=knn, 
            return_dataset_only=True, 
            path=train_path, 
            return_anchor=return_anchor, 
            loss=loss, 
            _format_esm=_format_esm,
            emb_out_dir=emb_dir)

    new_train_datamodule = CLEANActiveLearningDataModule(new_train_dataset, 
            train_batch_size=batch_size, 
            seed=seed, 
            ec_id=train_ec_id, 
            id_ec=train_id_ec, 
            emb=train_emb, 
            dist_map=train_dist_map, 
            ec_id_dict=train_ec_id_dict)

    #FIXME - this does not maintain the previously labeled instances in pool (i.e. from round i-1); I think its ok but may depend what the researcher wants
    new_pool_dataset = create_CLEAN_dataloader(pool_dist_map, 
            pool_id_ec, 
            pool_ec_id, 
            pool_ec_id_dict,
            batch_size=batch_size, 
            shuffle=shuffle, 
            knn=knn, 
            return_dataset_only=True, 
            path=pool_path, 
            return_anchor=return_anchor, 
            loss=loss, 
            _format_esm=_format_esm,
            emb_out_dir=emb_dir)

    new_pool_datamodule = CLEANActiveLearningDataModule(new_pool_dataset, 
            train_batch_size=batch_size, 
            seed=seed, 
            ec_id=pool_ec_id, 
            id_ec=pool_id_ec, 
            emb=pool_emb, 
            dist_map=pool_dist_map, 
            ec_id_dict=pool_ec_id_dict)


#    if update_mode:
#        pool_datamodule.update_all(pool_ec_id, pool_ec_id_dict, pool_id_ec, pool_emb, pool_dist_map, pool_negative, train_ec_id=train_ec_id, train_id_ec=train_id_ec)
#    else:
#        pool_datamodule.update_all(pool_ec_id, pool_ec_id_dict, pool_id_ec, pool_emb, pool_dist_map, pool_negative)

#    train_datamodule.update_all(train_ec_id, train_ec_id_dict, train_id_ec, train_emb, train_dist_map, train_negative)

    return new_train_datamodule, new_pool_datamodule


def create_CLEAN_dataloader(dist_map, id_ec, ec_id, ec_id_dict, batch_size=6000, shuffle=True, knn=30, return_dataset_only=False, path='./', emb_out_dir='/emb_data/', return_anchor=False, loss='triplet', n_pos=9, n_neg=30, _format_esm=True):
    params = {'batch_size': batch_size, 'shuffle': shuffle}

    negative = mine_hard_negative(dist_map, knn)

    if loss == 'triplet':
        train_data = Triplet_dataset_with_mine_EC(id_ec, ec_id, ec_id_dict, negative, path=path, emb_out_dir=emb_out_dir, return_anchor=return_anchor, _format_esm=_format_esm)
    elif loss == 'supconh':
        train_data = MultiPosNeg_dataset_with_mine_EC(id_ec, ec_id, ec_id_dict, negative, n_pos, n_neg, path=path, emb_out_dir=emb_out_dir, return_anchor=return_anchor, _format_esm=_format_esm)
    elif loss == 'himulcone':
        train_data = Heirarchical_dataset_with_mine_EC(id_ec, ec_id, ec_id_dict, negative, path=path, emb_out_dir=emb_out_dir, return_anchor=return_anchor, _format_esm=_format_esm)

    if return_dataset_only:
        return train_data

    train_loader = DataLoader(train_data, **params)
    
    return train_loader

def create_CLEAN_dataloader_w_update(train_dist_map, train_id_ec, train_ec_id, dist_map, id_ec, ec_id, ec_id_dict, ids_to_update, ecs_to_update, result_of_experiment, batch_size=6000, shuffle=False, knn=30, return_dataset_only=False, path='./', emb_out_dir='/emb_data/', return_anchor=False, pool_data_name='pool', loss='triplet', n_pos=9, n_neg=30, _format_esm=True):
    params = {'batch_size': batch_size,'shuffle': shuffle}

    train_negative = mine_hard_negative(train_dist_map, knn)
    negative = mine_hard_negative(dist_map, knn)

    if loss == 'triplet':
        pool_data = Active_learning_triplet_dataset_with_mine_EC(id_ec, ec_id, ec_id_dict, negative, train_id_ec=train_id_ec, train_ec_id=train_ec_id, train_mine_neg=train_negative, ids_for_update=ids_to_update, ecs_for_update=ecs_to_update, result_of_experiment=result_of_experiment, path=path, emb_out_dir=emb_out_dir, return_anchor=return_anchor, name=pool_data_name, _format_esm=_format_esm)
    elif loss == 'supconh':
        pool_data = Active_learning_supconh_dataset_with_mine_EC(id_ec, ec_id, ec_id_dict, negative, n_pos, n_neg, train_id_ec=train_id_ec, train_ec_id=train_ec_id, ids_for_update=ids_to_update, ecs_for_update=ecs_to_update, result_of_experiment=result_of_experiment, path=path, emb_out_dir=emb_out_dir, return_anchor=return_anchor, name=pool_data_name, _format_esm=_format_esm)
    elif loss == 'himulcone':
        pool_data = Active_learning_heirarchical_dataset_with_mine_EC(id_ec, ec_id, ec_id_dict, negative, train_id_ec=train_id_ec, train_ec_id=train_ec_id, ids_for_update=ids_to_update, ecs_for_update=ecs_to_update, result_of_experiment=result_of_experiment, path=path, emb_out_dir=emb_out_dir, return_anchor=return_anchor, name=pool_data_name, _format_esm=_format_esm)

    if return_dataset_only:
        return pool_data

    pool_loader = torch.utils.data.DataLoader(pool_data, **params)

    return pool_loader


def filter_ec_mappings(id_ec, ec_id, ec_id_dict, dist_map, percent=0.1, seed=None):
    """
    Randomly select a subset of IDs and filter EC mappings.

    Parameters:
        id_ec (dict): ID -> EC mapping
        ec_id (dict): EC -> list of IDs mapping
        ec_id_dict (dict): CLEAN EC prediction mapping (ID -> predicted EC)
        percent (float): fraction of IDs to select (0.0 - 1.0)
        seed (int, optional): random seed for reproducibility

    Returns:
        tuple: (filtered_id_ec, filtered_ec_id, filtered_ec_id_dict, selected_ids)
    """
    if seed is not None:
        random.seed(seed)

    # --- Step 1: Select random subset of IDs ---
    all_ids = list(id_ec.keys())
    n_select = max(1, int(len(all_ids) * percent))
    selected_ids = set(random.sample(all_ids, n_select))

    # --- Step 2: Filter id_ec ---
    filtered_id_ec = {k: v for k, v in id_ec.items() if k in selected_ids}

    # --- Step 3: Filter ec_id ---
    filtered_ec_id = {}
    for ec, ids in ec_id.items():
        filtered_ids = [i for i in ids if i in selected_ids]
        if filtered_ids:
            filtered_ec_id[ec] = filtered_ids

    # --- Step 4: Filter ec_id_dict ---
    filtered_ec_id_dict = {key: set(filtered_ec_id[key]) for key in filtered_ec_id.keys()}
    filtered_dist_map = {ec: {ec2: dist2 for ec2, dist2 in dist.items() if ec2 in filtered_ec_id.keys()} for ec, dist in dist_map.items() if ec in filtered_ec_id.keys()}

    # --- Step 5: Consistency checks ---
    ids_from_id_ec = set(filtered_id_ec.keys())
    ids_from_ec_id = {i for ids in filtered_ec_id.values() for i in ids}
    ids_from_ec_id_dict = {i for ids in filtered_ec_id_dict.values() for i in ids}

    # All IDs must be subsets of selected_ids
    assert ids_from_id_ec.issubset(selected_ids), "id_ec has IDs outside selected_ids"
    assert ids_from_ec_id.issubset(selected_ids), "ec_id has IDs outside selected_ids"
    assert ids_from_ec_id_dict.issubset(selected_ids), "ec_id_dict has IDs outside selected_ids"

    # All three dicts should cover the same set of IDs
    assert ids_from_id_ec == ids_from_ec_id == ids_from_ec_id_dict, (
        f"Inconsistent IDs:\n"
        f"id_ec keys: {ids_from_id_ec}\n"
        f"ec_id values: {ids_from_ec_id}\n"
        f"ec_id_dict keys: {ids_from_ec_id_dict}"
    )

    # Dist map check: all its ecs must match filtered_ec_id
    assert set(filtered_dist_map.keys()).issubset(set(filtered_ec_id.keys())), (
        "dist_map contains ecs not in filtered_ec_id"
    )

    # Collect all valid ECs
    valid_ecs = set(filtered_ec_id.keys())

    # Check that all negative entries are in valid_ecs
    for ec, data in filtered_dist_map.items():
        keys = data.keys()
        invalid = [n for n in keys if n not in valid_ecs]
        assert not invalid, f"{ec} has invalid keys: {invalid}"

    return filtered_id_ec, filtered_ec_id, filtered_ec_id_dict, filtered_dist_map, selected_ids


def create_CLEAN_active_learning_data_module(train_data_name, batch_size=128, shuffle=False, knn=30, precomputed_emb=False, path='./', seed=0, dtype=torch.float32, mutate_for_training=True, compute_distmaps_only=False, loss='triplet', _format_esm=True, emb_dir='/emb_data/', cache_dir='/distance_map/', use_old_naming_convention=False, return_anchor=False, percentage=None):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    id_ec, ec_id_dict = get_ec_id_dict(path+'/'+train_data_name+'.csv')
    ec_id = {key: list(ec_id_dict[key]) for key in ec_id_dict.keys()}


    if compute_distmaps_only:
        if mutate_for_training:
            train_fasta_file = mutate_single_seq_ECs(name=train_data_name, path=path, emb_out_dir=emb_dir)
            retrieve_esm1b_embedding(train_fasta_file, path=path, emb_out_dir=emb_dir)

        compute_emb_distance(train_data_name, 
                ec_id_dict=ec_id_dict, 
                path=path, 
                device=device, 
                dtype=dtype, 
                emb_out_dir=emb_dir, 
                cache_dir=cache_dir, 
                use_old_naming_convention=use_old_naming_convention)

        if use_old_naming_convention:
            ids = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'_ids.pkl', 'rb'))
            emb = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'_esm.pkl', 'rb')).to(device=device, dtype=dtype)
            dist_map = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'.pkl', 'rb'))
        else:
            ids = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'_ids.pkl', 'rb'))
            emb = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'_emb.pkl', 'rb')).to(device=device, dtype=dtype)
            dist_map = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'_dist.pkl', 'rb'))

    elif precomputed_emb:
        if use_old_naming_convention:
            ids = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'_ids.pkl', 'rb'))
            emb = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'_esm.pkl', 'rb')).to(device=device, dtype=dtype)
            dist_map = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'.pkl', 'rb'))
        else:
            ids = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'_ids.pkl', 'rb'))
            emb = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'_emb.pkl', 'rb')).to(device=device, dtype=dtype)
            dist_map = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'_dist.pkl', 'rb'))

    else:
        ensure_dirs(path)
        csv_to_fasta(path+'/'+train_data_name+'.csv', path+'/'+train_data_name+'.fasta')
        retrieve_esm1b_embedding(train_data_name, path=path, emb_out_dir=emb_dir)

        if mutate_for_training:
            train_fasta_file = mutate_single_seq_ECs(name=train_data_name, path=path, emb_out_dir=emb_dir)
            retrieve_esm1b_embedding(train_fasta_file, path=path, emb_out_dir=emb_dir)

        compute_emb_distance(train_data_name, 
                ec_id_dict=ec_id_dict, 
                path=path, 
                device=device, 
                dtype=dtype, 
                emb_out_dir=emb_dir, 
                cache_dir=cache_dir, 
                use_old_naming_convention=use_old_naming_convention)

        if use_old_naming_convention:
            ids = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'_ids.pkl', 'rb'))
            emb = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'_esm.pkl', 'rb')).to(device=device, dtype=dtype)
            dist_map = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'.pkl', 'rb'))
        else:
            ids = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'_ids.pkl', 'rb'))
            emb = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'_emb.pkl', 'rb')).to(device=device, dtype=dtype)
            dist_map = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'_dist.pkl', 'rb'))


    if percentage is not None:
        id_ec, ec_id, ec_id_dict, dist_map, selected_ids = filter_ec_mappings(id_ec, ec_id, ec_id_dict, dist_map, percent=percentage, seed=seed)
        redo_fasta_file = mutate_single_seq_ECs(id_ec=id_ec, ec_id=ec_id, name=train_data_name, path=path, emb_out_dir=emb_dir)
        retrieve_esm1b_embedding(redo_fasta_file, path=path, emb_out_dir=emb_dir)
        ids, emb, dist_map = compute_emb_distance(train_data_name, 
                ec_id_dict=ec_id_dict, 
                path=path, 
                device=device, 
                dtype=dtype, 
                emb_out_dir=emb_dir, 
                cache_dir=cache_dir, 
                use_old_naming_convention=use_old_naming_convention,
                dont_save=True)

    emb = dict(zip(ids, emb))

    dataset = create_CLEAN_dataloader(dist_map, 
            id_ec, 
            ec_id, 
            ec_id_dict,
            batch_size=batch_size, 
            shuffle=shuffle, 
            knn=knn, 
            return_dataset_only=True, 
            path=path, 
            return_anchor=return_anchor, 
            loss=loss, 
            _format_esm=_format_esm,
            emb_out_dir=emb_dir)

    al_datamodule = CLEANActiveLearningDataModule(dataset, 
            train_batch_size=batch_size, 
            seed=seed, 
            ec_id=ec_id, 
            id_ec=id_ec, 
            emb=emb, 
            dist_map=dist_map, 
            ec_id_dict=ec_id_dict)

    return al_datamodule

def create_CLEAN_active_learning_data_module_w_update(pool_data_name, train_datamodule, ids_to_update, ecs_to_update, result_of_experiment, batch_size=128, shuffle=False, knn=30, precomputed_emb=False, path='./', seed=0, dtype=torch.float32, mutate_for_training=True, compute_distmaps_only=False, loss='triplet', _format_esm=True, emb_dir='/emb_data/', cache_dir='/distance_map/', use_old_naming_convention=False, return_anchor=False):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    id_ec, ec_id_dict = get_ec_id_dict(path+'/'+pool_data_name+'.csv')
    ec_id = {key: list(ec_id_dict[key]) for key in ec_id_dict.keys()}

    if compute_distmaps_only:
        if mutate_for_training:
            pool_fasta_file = mutate_single_seq_ECs(name=pool_data_name, path=path, emb_out_dir=emb_dir)
            retrieve_esm1b_embedding(pool_fasta_file, path=path, emb_out_dir=emb_dir)

        compute_emb_distance(pool_data_name, 
                ec_id_dict=ec_id_dict, 
                path=path, 
                device=device, 
                dtype=dtype, 
                emb_out_dir=emb_dir, 
                cache_dir=cache_dir, 
                use_old_naming_convention=use_old_naming_convention)

        if use_old_naming_convention:
            ids = pickle.load(open(path+'/'+cache_dir+'/'+pool_data_name+'_ids.pkl', 'rb'))
            emb = pickle.load(open(path+'/'+cache_dir+'/'+pool_data_name+'_esm.pkl', 'rb')).to(device=device, dtype=dtype)
            dist_map = pickle.load(open(path+'/'+cache_dir+'/'+pool_data_name+'.pkl', 'rb'))
        else:
            ids = pickle.load(open(path+'/'+cache_dir+'/'+pool_data_name+'_ids.pkl', 'rb'))
            emb = pickle.load(open(path+'/'+cache_dir+'/'+pool_data_name+'_emb.pkl', 'rb')).to(device=device, dtype=dtype)
            dist_map = pickle.load(open(path+'/'+cache_dir+'/'+pool_data_name+'_dist.pkl', 'rb'))

    elif precomputed_emb:
        if use_old_naming_convention:
            ids = pickle.load(open(path+'/'+cache_dir+'/'+pool_data_name+'_ids.pkl', 'rb'))
            emb = pickle.load(open(path+'/'+cache_dir+'/'+pool_data_name+'_esm.pkl', 'rb')).to(device=device, dtype=dtype)
            dist_map = pickle.load(open(path+'/'+cache_dir+'/'+pool_data_name+'.pkl', 'rb'))
        else:
            ids = pickle.load(open(path+'/'+cache_dir+'/'+pool_data_name+'_ids.pkl', 'rb'))
            emb = pickle.load(open(path+'/'+cache_dir+'/'+pool_data_name+'_emb.pkl', 'rb')).to(device=device, dtype=dtype)
            dist_map = pickle.load(open(path+'/'+cache_dir+'/'+pool_data_name+'_dist.pkl', 'rb'))

    else:
        ensure_dirs(path)
        csv_to_fasta(path+'/'+pool_data_name+'.csv', path+'/'+pool_data_name+'.fasta')
        retrieve_esm1b_embedding(pool_data_name, path=path, emb_out_dir=emb_dir)

        if mutate_for_training:
            pool_fasta_file = mutate_single_seq_ECs(name=pool_data_name, path=path, emb_out_dir=emb_dir)
            retrieve_esm1b_embedding(pool_fasta_file, path=path, emb_out_dir=emb_dir)

        compute_emb_distance(pool_data_name, 
                ec_id_dict=ec_id_dict, 
                path=path, 
                device=device, 
                dtype=dtype, 
                emb_out_dir=emb_dir, 
                cache_dir=cache_dir, 
                use_old_naming_convention=use_old_naming_convention)

        if use_old_naming_convention:
            ids = pickle.load(open(path+'/'+cache_dir+'/'+pool_data_name+'_ids.pkl', 'rb'))
            emb = pickle.load(open(path+'/'+cache_dir+'/'+pool_data_name+'_esm.pkl', 'rb')).to(device=device, dtype=dtype)
            dist_map = pickle.load(open(path+'/'+cache_dir+'/'+pool_data_name+'.pkl', 'rb'))
        else:
            ids = pickle.load(open(path+'/'+cache_dir+'/'+pool_data_name+'_ids.pkl', 'rb'))
            emb = pickle.load(open(path+'/'+cache_dir+'/'+pool_data_name+'_emb.pkl', 'rb')).to(device=device, dtype=dtype)
            dist_map = pickle.load(open(path+'/'+cache_dir+'/'+pool_data_name+'_dist.pkl', 'rb'))

    emb = dict(zip(ids, emb))
    dataset = create_CLEAN_dataloader_w_update(train_datamodule.dist_map, 
            train_datamodule.id_ec, 
            train_datamodule.ec_id, 
            dist_map,
            id_ec, 
            ec_id, 
            ec_id_dict,
            ids_to_update, 
            ecs_to_update,
            result_of_experiment, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            return_dataset_only=True, 
            path=path, 
            return_anchor=return_anchor, 
            pool_data_name=pool_data_name, 
            loss=loss, 
            _format_esm=_format_esm,
            emb_out_dir=emb_dir)

    al_datamodule = CLEANActiveLearningDataModule(dataset, train_batch_size=batch_size, seed=seed, ec_id=ec_id, id_ec=id_ec, emb=emb, dist_map=dist_map, ec_id_dict=ec_id_dict)

    return al_datamodule

######################ORIGINAL - NO CLEAN##################################


class StandardActiveLearningDataModule(ActiveLearningDataModule):
    def __init__(
            self,
            train_dataset: Dataset,
            query_dataset: Dataset = None,
            train_batch_size: int = 64,
            predict_batch_size: int = 256,
            seed: int = None,
            fill_train_loader_batch: bool = True,
            collator=None, 
            emb=None,
    ):
        super().__init__(train_dataset)
        self.train_dataset = train_dataset
        self.query_dataset = MyQueryDataset(dataset=train_dataset)
        self.train_batch_size = train_batch_size
        self.predict_batch_size = predict_batch_size
        self.fill_train_loader_batch = fill_train_loader_batch
        self.collator = collator
        self.emb=emb

        if query_dataset is None:
            rank_zero_warn('Using train_dataset for queries. Ensure that there are no augmentations used.')

        self.rng = setup_rng(seed)
        self.unlabeled_indices = list(range(len(self.query_dataset)))
        self.labeled_indices = []

    def get_dataloader(self, batch_size=6000, shuffle=True):
        params = {'batch_size': batch_size, 'shuffle': shuffle}
        loader = DataLoader(self.train_dataset, **params)
        return loader

    def train_mode(self):
        self.labeled_indices = self.unlabeled_indices

    def reset(self):
        self.unlabeled_indices = list(range(len(self.query_dataset)))
        self.labeled_indices = []

def create_standard_dataloader(id_ec, batch_size=6000, shuffle=True, return_dataset_only=False, path='./', emb_out_dir='/emb_data/', return_anchor=False, _format_esm=True):
    params = {'batch_size': batch_size, 'shuffle': shuffle}

    train_data = StandardDataset(id_ec, path=path, emb_out_dir=emb_out_dir, return_anchor=return_anchor, _format_esm=_format_esm)

    if return_dataset_only:
        return train_data

    train_loader = DataLoader(train_data, **params)
    
    return train_loader

def create_standard_active_learning_data_module(train_data_name, batch_size=128, shuffle=False, knn=30, precomputed_emb=False, path='./', seed=0, dtype=torch.float32, _format_esm=True, emb_dir='/emb_data/', cache_dir='/distance_map/', use_old_naming_convention=False, return_anchor=False):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    id_ec, ec_id_dict = get_ec_id_dict(path+'/'+train_data_name+'.csv')
    #ec_id = {key: list(ec_id_dict[key]) for key in ec_id_dict.keys()}

    if precomputed_emb:
        if use_old_naming_convention:
            ids = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'_ids.pkl', 'rb'))
            emb = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'_esm.pkl', 'rb')).to(device=device, dtype=dtype)
        else:
            ids = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'_ids.pkl', 'rb'))
            emb = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'_emb.pkl', 'rb')).to(device=device, dtype=dtype)

    else:
        ensure_dirs(path)
        csv_to_fasta(path+'/'+train_data_name+'.csv', path+'/'+train_data_name+'.fasta')
        retrieve_esm1b_embedding(train_data_name, path=path, emb_out_dir=emb_dir)

        ids, emb, ecs = load_embeddings(ec_id_dict, device, dtype, path=path, emb_out_dir=emb_dir, return_all=True, _format_esm=_format_esm)

        ensure_dirs(path+'/'+cache_dir)

        pickle.dump(ids, open(path+'/'+cache_dir+'/'+train_data_name+'_ids.pkl', 'wb'))
        pickle.dump(ecs, open(path+'/'+cache_dir+'/'+train_data_name+'_ecs.pkl', 'wb'))
        pickle.dump(emb, open(path+'/'+cache_dir+'/'+train_data_name+'_esm.pkl', 'wb'))

        if use_old_naming_convention:
            ids = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'_ids.pkl', 'rb'))
            emb = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'_esm.pkl', 'rb')).to(device=device, dtype=dtype)
        else:
            ids = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'_ids.pkl', 'rb'))
            emb = pickle.load(open(path+'/'+cache_dir+'/'+train_data_name+'_emb.pkl', 'rb')).to(device=device, dtype=dtype)

    emb = dict(zip(ids, emb))

    dataset = create_standard_dataloader(id_ec, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            return_dataset_only=True, 
            path=path, 
            return_anchor=return_anchor, 
            _format_esm=_format_esm,
            emb_out_dir=emb_dir)

    al_datamodule = StandardActiveLearningDataModule(dataset, 
            train_batch_size=batch_size, 
            seed=seed, 
            emb=emb)

    return al_datamodule
