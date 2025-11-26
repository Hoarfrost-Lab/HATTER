import torch
import random
from .utils import format_esm
from tqdm import tqdm

def find_first_non_zero_distance(data):
    for index, (name, distance) in enumerate(data):
        if distance != 0:
            return index
    return None 

def mine_hard_negative(dist_map, knn=10):
    #print("The number of unique EC numbers: ", len(dist_map.keys()))
    ecs = list(dist_map.keys())
    negative = {}
    print("Mining hard negatives:")
    for _, target in tqdm(enumerate(ecs), total=len(ecs)):
        sorted_orders = sorted(dist_map[target].items(), key=lambda x: x[1], reverse=False)
        assert sorted_orders != None, "all clusters have zero distances!"
        neg_ecs_start_index = find_first_non_zero_distance(sorted_orders)
        closest_negatives = sorted_orders[neg_ecs_start_index:neg_ecs_start_index + knn]
        freq = [1/i[1] for i in closest_negatives]
        neg_ecs = [i[0] for i in closest_negatives]        
        normalized_freq = [i/sum(freq) for i in freq]
        negative[target] = {
            'weights': normalized_freq,
            'negative': neg_ecs
        }
    return negative


def mine_negative(anchor, id_ec, ec_id, mine_neg, return_ec_label=False, train_tuple=None):

    #insert helper code for active learning update
    run_normal = True
    if train_tuple is not None:
        train_id_ec, train_ec_id, mine_neg_train = train_tuple
        run_normal = False

        try:
            anchor_ec = train_id_ec[anchor]
            pos_ec = random.choice(anchor_ec)
            neg_ec = mine_neg_train[pos_ec]['negative']
            weights = mine_neg_train[pos_ec]['weights']
            
            result_ec = random.choices(neg_ec, weights=weights, k=1)[0]

            #can cause infinite loop with small data that dont have enough neg ec
            while result_ec in anchor_ec:
                result_ec = random.choices(neg_ec, weights=weights, k=1)[0]

            neg_id = random.choice(train_ec_id[result_ec])
        
        except:
            run_normal = True

    if run_normal:
        anchor_ec = id_ec[anchor]
        pos_ec = random.choice(anchor_ec)
        neg_ec = mine_neg[pos_ec]['negative']
        weights = mine_neg[pos_ec]['weights']

        #consider changing implementation in the future FIXME
        #filter list pre-emptively to prevent the looping below
        #neg_ec = [ec for ec in neg_ec if ec not in anchor_ec]
        #weights = [weight for weight, ec in zip(weights, neg_ec) if ec not in anchor_ec]
    
        #if not neg_ec: #not ideal case because it means we have very limited EC set to choose from
        #    neg_ec = [ec for ec in ec_id.keys() if ec not in anchor_ec]
        #    weights = None

        #if not neg_ec:
        #    return None #cannot get a negative for this id

        result_ec = random.choices(neg_ec, weights=weights, k=1)[0]
        #assert(result_ec not in anchor_ec)

        #can cause infinite loop with small data that dont have enough neg ec
        while result_ec in anchor_ec:
            result_ec = random.choices(neg_ec, weights=weights, k=1)[0]

        neg_id = random.choice(ec_id[result_ec])

    if return_ec_label and run_normal:
        return neg_id, list(set(id_ec[neg_id]))[0]

    return neg_id


def random_positive(_id, id_ec, ec_id, train_tuple=None):

    #insert helper code for active learning update
    run_normal = True
    if train_tuple is not None: 
        run_normal = False
        train_id_ec, train_ec_id = train_tuple
        try:
            pos_ec = random.choice(train_id_ec[_id])

            pos = _id
            if len(train_ec_id[pos_ec]) == 1:
                return pos + '_' + str(random.randint(0, 9))
            while pos == _id:
                pos = random.choice(train_ec_id[pos_ec])
        except:
            run_normal = True

    #original code
    if run_normal:
        pos_ec = random.choice(id_ec[_id])

        pos = _id
        if len(ec_id[pos_ec]) == 1:
            return pos + '_' + str(random.randint(0, 9))
        while pos == _id:
            pos = random.choice(ec_id[pos_ec])
    
    return pos


class Triplet_dataset_with_mine_EC(torch.utils.data.Dataset):

    def __init__(self, id_ec, ec_id, ec_id_dict, mine_neg, path='./', emb_out_dir='/emb_data/', return_anchor=False, _format_esm=True):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.ec_id_dict = ec_id_dict
        self.full_list = []
        self.mine_neg = mine_neg

        self.path = path
        self.emb_out_dir = emb_out_dir
        self.return_anchor=return_anchor
        self.format_esm = _format_esm
        self.query_dataset = False #only set in query overload
        self.sorted_ids = sorted(list(self.id_ec.keys()))

        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.query_dataset: #need to pull from id not ec
            anchor = self.sorted_ids[index]
            anchor_ec = random.choice(self.id_ec[anchor])
        else:
            anchor_ec = self.full_list[index]
            anchor = random.choice(self.ec_id[anchor_ec])
        
        pos = random_positive(anchor, self.id_ec, self.ec_id)
        neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)
        
        a = torch.load(self.path+'/'+self.emb_out_dir+'/'+anchor+'.pt')
        p = torch.load(self.path+'/'+self.emb_out_dir+'/'+pos+'.pt')
        n = torch.load(self.path+'/'+self.emb_out_dir+'/'+neg+'.pt')
        
        if self.return_anchor:
            if self.format_esm:
                return format_esm(a).to(device), format_esm(p).to(device), format_esm(n).to(device), (anchor, anchor_ec)
            else:
                return a, p, n, (anchor, anchor_ec)

        if self.format_esm:
            return format_esm(a).to(device), format_esm(p).to(device), format_esm(n).to(device)

        return a, p, n

    def update_ecs_and_ids(self, id_ec, ec_id, ec_id_dict, new_neg):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.ec_id_dict = ec_id_dict
        self.mine_neg = new_neg
        self.sorted_ids = sorted(list(self.id_ec.keys()))
        self.full_list = []

        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)


class MultiPosNeg_dataset_with_mine_EC(torch.utils.data.Dataset):

    def __init__(self, id_ec, ec_id, ec_id_dict, mine_neg, n_pos, n_neg, path='./', emb_out_dir='/emb_data/', return_anchor=False, _format_esm=True):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.ec_id_dict = ec_id_dict
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.mine_neg = mine_neg

        self.path = path
        self.emb_out_dir = emb_out_dir
        self.return_anchor = return_anchor
        self.format_esm = _format_esm
        self.query_dataset = False #only set in query overload

        self.sorted_ids = sorted(list(self.id_ec.keys()))
        self.full_list = []
        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.query_dataset: #need to pull from id not ec
            anchor = self.sorted_ids[index]
            anchor_ec = random.choice(self.id_ec[anchor])
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
        
        if self.return_anchor:
            return torch.cat(data).to(device), (anchor, anchor_ec)

        return torch.cat(data).to(device)

    def update_ecs_and_ids(self, id_ec, ec_id, ec_id_dict, new_neg):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.ec_id_dict = ec_id_dict
        self.mine_neg = new_neg
        self.sorted_ids = sorted(list(self.id_ec.keys()))
        self.full_list = []

        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

