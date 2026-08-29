from torch.utils.data import Dataset, DataLoader
import torch
import random

from clean_app.src.CLEAN.dataloader import mine_hard_negative, Triplet_dataset_with_mine_EC, MultiPosNeg_dataset_with_mine_EC, mine_negative, random_positive
from clean_app.src.CLEAN.utils import ensure_dirs, csv_to_fasta, retrieve_esm1b_embedding, mutate_single_seq_ECs, compute_emb_distance, get_ec_id_dict, mutate_incorrect_seq_ECs, format_esm
from clean_app.src.CLEAN.distance_map import get_dist_map

from dal_toolbox.active_learning import ActiveLearningDataModule, QueryDataset
from dal_toolbox.utils import setup_rng



class Heirarchical_dataset_with_mine_EC(Dataset):

    def __init__(self, id_ec, ec_id, ec_id_dict, mine_neg, path='./', emb_out_dir='/emb_data/', return_anchor=False, _format_esm=True):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.ec_id_dict = ec_id_dict
        self.mine_neg = mine_neg

        self.path = path
        self.emb_out_dir = emb_out_dir
        self.return_anchor = return_anchor
        self.format_esm = _format_esm
        self.query_dataset = False #can only get set in query dataset overload

        self.maxlen = 0
        self.full_list = []
        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

            if len(ec_id[ec]) > self.maxlen:
                self.maxlen = len(ec_id[ec])

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):

        if self.query_dataset: #need to pull from id not ec
            anchor = list(self.id_ec.keys())[index]
            anchor_ec = random.choice(self.id_ec[anchor])

        else:
            anchor_ec = self.full_list[index]
            anchor = random.choice(self.ec_id[anchor_ec])

        anchor_label = [int(i.replace('n', '1000')) for i in list(anchor_ec.split('.'))]
        
        if self.format_esm:
            a = format_esm(torch.load(self.path+'/'+self.emb_out_dir+'/'+anchor+'.pt')).unsqueeze(0)
        else:
            a = torch.load(self.path+'/'+self.emb_out_dir+'/'+anchor+'.pt').unsqueeze(0)

        if self.return_anchor:
            return torch.squeeze(torch.stack([a])), torch.tensor(anchor_label), (anchor, anchor_ec)

        return torch.squeeze(torch.stack([a])), torch.tensor(anchor_label)

    def update_ecs_and_ids(self, id_ec, ec_id, ec_id_dict, new_neg):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.ec_id_dict = ec_id_dict
        self.mine_neg = new_neg
        self.full_list = []

        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)


class Active_learning_triplet_dataset_with_mine_EC(Triplet_dataset_with_mine_EC):

    def __init__(self, id_ec, ec_id, ec_id_dict, mine_neg, train_id_ec=None, train_ec_id=None, train_mine_neg=None, ids_for_update=None, ecs_for_update=None, result_of_experiment=None, name='pool', path='./', emb_out_dir='/emb_data/', return_anchor=False, _format_esm=True):
        self.train_id_ec = train_id_ec
        self.train_ec_id = train_ec_id
        self.mine_neg_train = train_mine_neg
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.ec_id_dict = ec_id_dict
        self.full_list = []
        self.mine_neg = mine_neg
        self.ids_for_update = ids_for_update
        self.ecs_for_update = ecs_for_update
        self.result_of_experiment = result_of_experiment

        self.path = path
        self.emb_out_dir = emb_out_dir
        self.return_anchor=return_anchor
        self.query_dataset = False #only set if query dataset is activated in dataloader

        self.sorted_ids = sorted(list(self.id_ec.keys()))
        # Contrastive mining scope. When set via set_labeled_scope(), positives
        # and negatives come ONLY from train plus already-acquired pool points,
        # never from unlabelled pool entries.
        self.labeled_id_ec = None
        self.labeled_ec_id = None
        self.labeled_mine_neg = None


        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

        #need to get mutated sequences to be able to properly perform update
        if ids_for_update is not None:
            incorrect_fasta_file = mutate_incorrect_seq_ECs(ids_for_update, result_of_experiment, path=path, name=name, emb_out_dir=emb_out_dir)
            retrieve_esm1b_embedding(incorrect_fasta_file, path=path, emb_out_dir=emb_out_dir)

    def set_labeled_scope(self, labeled_ids, train_id_ec=None, train_ec_id=None):
        """Restrict contrastive mining to train + already-acquired pool points.

        Without this, __getitem__ mines positives and negatives from the pool's
        full id_ec/ec_id maps. Training anchors are correctly restricted to the
        labelled subset by the datamodule, but a negative drawn from an
        UNLABELLED pool entry uses that entry's EC label to establish it as a
        valid negative -- the label of data the model has not acquired.
        Hard-negative mining makes this worse than random: it selects the
        nearest wrong-EC neighbours, the most informative subset. The effect
        grows with pool size and would read as an active-learning gain.

        The deployed update path already avoids this via train_tuple; simulation
        was the odd one out. train_tuple alone is NOT sufficient, because
        random_positive wraps its train branch in try/except and falls back to
        pool mining for an anchor absent from train -- exactly the novel-EC
        case. Restricting the maps closes that path properly.

        Call after every update_annotations().
        """
        labeled = set(labeled_ids)
        id_ec, ec_id = {}, {}

        def _add(_id, ecs):
            if not ecs:
                return
            ecs = [ecs] if isinstance(ecs, str) else list(ecs)
            id_ec[_id] = ecs
            for ec in ecs:
                ec_id.setdefault(ec, set()).add(_id)

        if train_id_ec:
            for _id, ecs in train_id_ec.items():
                _add(_id, ecs)
        for _id in labeled:
            _add(_id, self.id_ec.get(_id))

        self.labeled_id_ec = id_ec
        self.labeled_ec_id = {k: sorted(v) for k, v in ec_id.items()}

        # Filter the mined-negative table to ECs inside the labelled scope,
        # renormalising sampling weights over what survives.
        filtered = {}
        for ec, entry in (self.mine_neg or {}).items():
            if ec not in ec_id:
                continue
            negs = entry["negative"] if isinstance(entry, dict) else entry
            weights = entry.get("weights") if isinstance(entry, dict) else None
            keep = [(n, (weights[i] if weights is not None else 1.0))
                    for i, n in enumerate(negs) if n in ec_id]
            if not keep:
                continue
            ns, ws = zip(*keep)
            total = float(sum(ws)) or 1.0
            filtered[ec] = {"negative": list(ns),
                            "weights": [w / total for w in ws]}
        self.labeled_mine_neg = filtered

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        if self.ids_for_update == None: #standard training
            if self.query_dataset: #need to pull from id not ec
                anchor = self.sorted_ids[index]
                anchor_ec = self.id_ec[anchor] #return entire list for query when its in active learning mode
            else:
                anchor_ec = self.full_list[index]
                anchor = random.choice(self.ec_id[anchor_ec])

            if self.labeled_id_ec is not None:
                # Mine only within train + acquired. random_positive already
                # falls back to a mutated self-variant when the anchor's EC has
                # exactly one member, which is the right behaviour for an EC
                # just acquired for the first time: it self-mutates until a
                # second real example arrives.
                scope_id_ec = self.labeled_id_ec
                scope_ec_id = self.labeled_ec_id
                scope_neg = self.labeled_mine_neg
                if anchor not in scope_id_ec:
                    # Never silently widen to the unlabelled pool.
                    scope_id_ec = {anchor: self.id_ec.get(anchor, [])}
                    scope_ec_id = {}
                    scope_neg = {}
            else:
                scope_id_ec = self.id_ec
                scope_ec_id = self.ec_id
                scope_neg = self.mine_neg

            pos = random_positive(anchor, scope_id_ec, scope_ec_id)
            neg = mine_negative(anchor, scope_id_ec, scope_ec_id, scope_neg)

        else: #trying specifically to target queried sequence ids from the previous round
            anchor = self.sorted_ids[index] #should always be a query dataset if we get here

            for _id, _ec, _result in zip(self.ids_for_update, self.ecs_for_update, self.result_of_experiment):
                if _id == anchor:
                    anchor_ec = _ec
                    result = _result
                    break

            if result: #result is correct continue with normal model update #FIXME -- should probably be train_ec_id but complicates how I currently update the ids
                pos = random_positive(anchor, self.id_ec, self.ec_id, train_tuple=(self.train_id_ec, self.train_ec_id))
                neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg, train_tuple=(self.train_id_ec, self.train_ec_id, self.mine_neg_train))
            
            else: #result is incorrect - set positive as negative and mutate sequence for new positive
                pos = anchor + '_' + str(random.randint(0, 9))
                neg = random_positive(anchor, self.id_ec, self.ec_id)

        a = torch.load(self.path+self.emb_out_dir+anchor+'.pt')
        p = torch.load(self.path+self.emb_out_dir+pos+'.pt')
        n = torch.load(self.path+self.emb_out_dir+neg+'.pt')
        
        if self.return_anchor:
            return format_esm(a), format_esm(p), format_esm(n), (anchor, anchor_ec)

        return format_esm(a), format_esm(p), format_esm(n)

    def update_ecs_and_ids(self, id_ec, ec_id, ec_id_dict, new_neg, train_id_ec=None, train_ec_id=None, ids_for_update=None, ecs_for_update=None, result_of_experiment=None):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.ec_id_dict = ec_id_dict
        self.mine_neg = new_neg
        self.train_id_ec = train_id_ec
        self.train_ec_id = train_ec_id
        self.ids_for_update = ids_for_update
        self.ecs_for_update = ecs_for_update
        self.result_of_experiment = result_of_experiment
        self.sorted_ids = sorted(list(self.id_ec.keys()))
        self.full_list = []

        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

class Active_learning_supconh_dataset_with_mine_EC(MultiPosNeg_dataset_with_mine_EC):

    def __init__(self, id_ec, ec_id, ec_id_dict, mine_neg, n_pos, n_neg, train_id_ec=None, train_ec_id=None, ids_for_update=None, ecs_for_update=None, result_of_experiment=None, name='pool', path='./', emb_out_dir='/emb_data/', return_anchor=False, _format_esm=True):
        self.train_id_ec = train_id_ec
        self.train_ec_id = train_ec_id
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.ec_id_dict = ec_id_dict
        self.full_list = []
        self.mine_neg = mine_neg
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.ids_for_update = ids_for_update
        self.ecs_for_update = ecs_for_update
        self.result_of_experiment = result_of_experiment

        self.path = path
        self.emb_out_dir = emb_out_dir
        self.return_anchor=return_anchor
        self.query_dataset = False #only set if query dataset is activated in dataloader

        self.sorted_ids = sorted(list(self.id_ec.keys()))

        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

        #mutate to properly perform train step in update mode
        if ids_for_update is not None:
            incorrect_fasta_file = mutate_incorrect_seq_ECs(ids_for_update, result_of_experiment, path=path, name=name, emb_out_dir=emb_out_dir)
            retrieve_esm1b_embedding(incorrect_fasta_file, path=path, emb_out_dir=emb_out_dir)

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):

        if self.ids_for_update == None: #standard training
            if self.query_dataset: #need to pull from id not ec
                anchor = self.sorted_ids[index]
                anchor_ec = self.id_ec[anchor] #return whole list when in init mode
            else:
                anchor_ec = self.full_list[index]
                anchor = random.choice(self.ec_id[anchor_ec])

            if self.format_esm:
                a = format_esm(torch.load(self.path+'/'+self.emb_out_dir+'/'+anchor+'.pt')).to(device).unsqueeze(0)
            else:
                a = torch.load(self.path+'/'+self.emb_out_dir+'/'+anchor+'.pt').to(device).unsqueeze(0)

            data = [a]

            for _ in range(self.n_pos):
                pos = random_positive(anchor, self.id_ec, self.ec_id)
                if self.format_esm:
                    p = format_esm(torch.load(self.path+'/'+self.emb_out_dir+'/'+pos+'.pt')).to(device).unsqueeze(0)
                else:
                    p = torch.load(self.path+'/'+self.emb_out_dir+'/'+pos+'.pt').to(device).unsqueeze(0)

                data.append(p)

            for _ in range(self.n_neg):
                neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)
                if self.format_esm:
                    n = format_esm(torch.load(self.path+'/'+self.emb_out_dir+'/'+neg+'.pt')).to(device).unsqueeze(0)
                else:
                    n = torch.load(self.path+'/'+self.emb_out_dir+'/'+neg+'.pt').to(device).unsqueeze(0)

                data.append(n)


        else: #trying specifically to target queried sequence ids from the previous round
            anchor = self.sorted_ids[index] #should be a query dataset if we get here
            
            for _id, _ec, _result in zip(self.ids_for_update, self.ecs_for_update, self.result_of_experiment):
                if _id == anchor:
                    anchor_ec = _ec
                    result = _result
                    break

            a = format_esm(torch.load(self.path+'/'+self.emb_out_dir+'/'+anchor+'.pt')).to(device).unsqueeze(0)
            data = [a]

            if result: #result is correct continue with normal model update
                for _ in range(self.n_pos):
                    pos = random_positive(anchor, self.id_ec, self.ec_id)
                    p = format_esm(torch.load(self.path+'/'+self.emb_out_dir+'/'+pos+'.pt')).to(device).unsqueeze(0)

                    data.append(p)
        
                for _ in range(self.n_neg):
                    neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)
                    n = format_esm(torch.load(self.path+'/'+self.emb_out_dir+'/'+neg+'.pt')).to(device).unsqueeze(0)
            
                    data.append(n)
            
            else: #result is incorrect - set positive as negative and mutate sequence for new positive
                for _ in range(self.n_pos):
                    pos = anchor + '_' + str(random.randint(0, 9))
                    p = format_esm(torch.load(self.path+'/'+self.emb_out_dir+'/'+pos+'.pt')).to(device).unsqueeze(0)

                    data.append(p)
        
                for _ in range(self.n_neg):
                    neg = random_positive(anchor, self.id_ec, self.ec_id)
                    n = format_esm(torch.load(self.path+'/'+self.emb_out_dir+'/'+neg+'.pt')).to(device).unsqueeze(0)
            
                    data.append(n)

        if self.return_anchor:
            return torch.cat(data), (anchor, anchor_ec)

        return torch.cat(data)

    def update_ecs_and_ids(self, id_ec, ec_id, ec_id_dict, new_neg, train_id_ec=None, train_ec_id=None, ids_for_update=None, result_of_experiment=None):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.ec_id_dict = ec_id_dict
        self.mine_neg = new_neg
        self.train_id_ec = train_id_ec
        self.train_ec_id = train_ec_id
        self.ids_for_update = ids_for_update
        self.ecs_for_update = ecs_for_update
        self.result_of_experiment = result_of_experiment
        self.sorted_ids = sorted(list(self.id_ec.keys()))
        self.full_list = []

        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

#not currently supported FIXME
class Active_learning_heirarchical_dataset_with_mine_EC(Heirarchical_dataset_with_mine_EC):

    def __init__(self, id_ec, ec_id, ec_id_dict, mine_neg, train_id_ec=None, train_ec_id=None, ids_for_update=None, result_of_experiment=None, name='pool', path='./', emb_out_dir='/emb_data/', return_anchor=False):
        self.train_id_ec = train_id_ec
        self.train_ec_id = train_ec_id
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.ec_id_dict = ec_id_dict
        self.full_list = []
        self.mine_neg = mine_neg
        self.ids_for_update = ids_for_update
        self.result_of_experiment = result_of_experiment

        self.path = path
        self.emb_out_dir = emb_out_dir
        self.return_anchor=return_anchor
        self.query_dataset = False #only set if query dataset is activated in dataloader

        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

        incorrect_fasta_file = mutate_incorrect_seq_ECs(ids_for_update, result_of_experiment, path=path, name=name, emb_out_dir=emb_out_dir)
        retrieve_esm1b_embedding(incorrect_fasta_file, path=path, emb_out_dir=emb_out_dir)

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        anchor_ec = self.full_list[index]

        if self.ids_for_update == None: #standard training
            anchor = random.choice(self.ec_id[anchor_ec])

        else: #trying specifically to target queried sequence ids from the previous round
            for _id, _result in zip(self.ids_for_update, self.result_of_experiment):
                if _id in self.ec_id[anchor_ec]:
                    anchor = _id
                    result = _result
                    break

            if not result: #result is not correct repredict EC
                anchor_ec = '1.1.1.1' #TODO: fix this repred of EC
            
        anchor_label = [int(i.replace('n', '1000')) for i in list(anchor_ec.split('.'))]
        a = format_esm(torch.load(self.path+'/'+self.emb_out_dir+'/'+anchor+'.pt')).unsqueeze(0)

        if self.return_anchor:
            return torch.squeeze(torch.stack([a])), torch.tensor(anchor_label), (anchor, anchor_ec)

        return torch.squeeze(torch.stack([a])), torch.tensor(anchor_label)

    def update_ecs_and_ids(self, id_ec, ec_id, ec_id_dict, new_neg, train_id_ec=None, train_ec_id=None, ids_for_update=None, result_of_experiment=None):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.ec_id_dict = ec_id_dict
        self.mine_neg = new_neg
        self.train_id_ec = train_id_ec
        self.train_ec_id = train_ec_id
        self.ids_for_update = ids_for_update
        self.result_of_experiment = result_of_experiment
        self.full_list = []

        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

class MyCLEANQueryDataset(QueryDataset):
    """A helper class which returns also the index along with the instances and targets."""
    # problem with dictionary output of dataset

    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset
        self.dataset.query_dataset = True #set query mode to maintain multiple types of compatibility with various training loops
        self.full_list = sorted(list(set(self.dataset.id_ec.keys())))

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        data = self.dataset.__getitem__(index)
        return data

#################ORIGINAL - NO CLEAN################################

class StandardDataset(Dataset):

    def __init__(self, id_ec, path='./', emb_out_dir='/emb_data/', return_anchor=False, _format_esm=True):
        self.path = path
        self.emb_out_dir = emb_out_dir
        self.return_anchor = return_anchor
        self.format_esm = _format_esm
        self.query_dataset = False #can only get set in query dataset overload
        self.label_encoder = None #will fill later

        self.full_list_ids = []
        self.full_list_labels = []
        self.id_ec = id_ec #need in active learning steps

        #parse ec_id to get the full lists for standard classifier
        for _id in id_ec.keys():
            for ec in id_ec[_id]:
                if '-' not in ec:
                    self.full_list_ids.append(_id)
                    self.full_list_labels.append(ec)

        assert(len(self.full_list_ids) == len(self.full_list_labels))

    def transform_labels(self):
        self.labels_transformed = self.label_encoder.transform(self.full_list_labels)

    def __len__(self):
        return len(self.full_list_ids)

    def __getitem__(self, index):

        anchor = self.full_list_ids[index]
        anchor_label = self.labels_transformed[index]

        if self.format_esm:
            a = format_esm(torch.load(self.path+'/'+self.emb_out_dir+'/'+anchor+'.pt'))
        else:
            a = torch.load(self.path+'/'+self.emb_out_dir+'/'+anchor+'.pt')

        if self.return_anchor:
            return a, torch.tensor(anchor_label), index, (anchor, anchor_ec)

        return a, torch.tensor(anchor_label), index



class MyQueryDataset(QueryDataset):
    """A helper class which returns also the index along with the instances and targets."""
    # problem with dictionary output of dataset

    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset

    def __getitem__(self, index):
        # TODO(dhuseljic): discuss with marek, index instead of target? maybe dictionary? leave it like that?
        return self.dataset.__getitem__(index)
