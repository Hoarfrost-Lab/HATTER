import csv
import random
import os
import math
from re import L
import torch
import numpy as np
import subprocess
import pickle
from .distance_map import get_dist_map
from tqdm import tqdm

def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_ec_id_dict(csv_name: str) -> dict:
    csv_file = open(csv_name)
    csvreader = csv.reader(csv_file, delimiter='\t')
    id_ec = {}
    ec_id = {}

    for i, rows in enumerate(csvreader):
        if i > 0:
            id_ec[rows[0]] = rows[1].split(';')
            for ec in rows[1].split(';'):
                if ec not in ec_id.keys():
                    ec_id[ec] = set()
                    ec_id[ec].add(rows[0])
                else:
                    ec_id[ec].add(rows[0])
    return id_ec, ec_id

def get_ec_id_dict_non_prom(csv_name: str) -> dict:
    csv_file = open(csv_name)
    csvreader = csv.reader(csv_file, delimiter='\t')
    id_ec = {}
    ec_id = {}

    for i, rows in enumerate(csvreader):
        if i > 0:
            if len(rows[1].split(';')) == 1:
                id_ec[rows[0]] = rows[1].split(';')
                for ec in rows[1].split(';'):
                    if ec not in ec_id.keys():
                        ec_id[ec] = set()
                        ec_id[ec].add(rows[0])
                    else:
                        ec_id[ec].add(rows[0])
    return id_ec, ec_id


def format_esm(a):
    if type(a) == dict:
        a = a['mean_representations'][33]
    return a


def load_emb(lookup, path='./', emb_out_dir='/emb_data/', _format_esm=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if _format_esm:
        esm = format_esm(torch.load(path+'/'+emb_out_dir+'/'+lookup+'.pt')).to(device)
    else:
        esm = torch.load(path+'/'+emb_out_dir+'/'+lookup+'.pt').to(device)
    
    return esm.unsqueeze(0)


def load_embeddings(ec_id_dict, device, dtype, path='./', emb_out_dir='/emb_data/', return_all=False, _format_esm=True):
    '''
    Loading esm embedding in the sequence of EC numbers
    prepare for calculating cluster center by EC
    '''
    emb = []
    ids = []
    ecs = []
    
    for ec in tqdm(list(ec_id_dict.keys())):
        ids_for_query = list(ec_id_dict[ec])
        ids.extend(ids_for_query)
        ecs.extend([ec]*len(ids_for_query))

        emb_to_cat = [load_emb(_id, path=path, emb_out_dir=emb_out_dir, _format_esm=_format_esm) for _id in ids_for_query]
        emb = emb + emb_to_cat
    
    final_emb = torch.cat(emb).to(device=device, dtype=dtype)

    if return_all:
        return ids, final_emb, ecs
    
    return final_emb


def model_embedding_test(id_ec_test, model, device, dtype, path='./', emb_out_dir='/emb_data/', _format_esm=True):
    '''
    Instead of loading esm embedding in the sequence of EC numbers
    the test embedding is loaded in the sequence of queries
    then inferenced with model to get model embedding
    '''
    ids_for_query = list(id_ec_test.keys())
    emb_to_cat = [load_emb(_id, path=path, emb_out_dir=emb_out_dir, _format_esm=_format_esm) for _id in ids_for_query]
    emb = torch.cat(emb_to_cat).to(device=device, dtype=dtype)
    model_emb = model(emb)
    return model_emb

def model_embedding_test_ensemble(id_ec_test, device, dtype, path='./', emb_out_dir='/emb_data/', _format_esm=True):
    '''
    Instead of loading esm embedding in the sequence of EC numbers
    the test embedding is loaded in the sequence of queries
    '''
    ids_for_query = list(id_ec_test.keys())
    emb_to_cat = [load_emb(_id, path=path, emb_out_dir=emb_out_dir, _format_esm=_format_esm) for _id in ids_for_query]
    emb = torch.cat(emb_to_cat).to(device=device, dtype=dtype)
    return emb

def csv_to_fasta(csv_name, fasta_name):
    csvfile = open(csv_name, 'r')
    csvreader = csv.reader(csvfile, delimiter='\t')
    outfile = open(fasta_name, 'w')

    ids = set([])
    for i, rows in enumerate(csvreader):
        
        if rows[0] in ids: #dont add duplicate gene ids
            continue

        ids.add(rows[0])
        
        if i > 0:
            outfile.write('>' + rows[0] + '\n')
            outfile.write(rows[2] + '\n')
            
def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def retrieve_esm1b_embedding(name, path='./', emb_out_dir='/emb_data/'):
    esm_script = "esm/scripts/extract.py"
    esm_type = "esm1b_t33_650M_UR50S"
    esm_out = path+'/'+emb_out_dir
    fasta_name = path+"/" + name + ".fasta"
    command = ["python", esm_script, esm_type, 
              fasta_name, esm_out, "--include", "mean"]
    subprocess.run(command)
 
def compute_emb_distance(train_file, path='./', emb_out_dir='/emb_data/', cache_dir=None, ec_id_dict=None, device='cpu', dtype=torch.float32, _format_esm=True, use_old_naming_convention=False, dont_save=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if ec_id_dict is None:
        _, ec_id_dict = get_ec_id_dict(path +'/'+ train_file + '.csv')
    
    ids, emb, ecs = load_embeddings(ec_id_dict, device, dtype, path=path, emb_out_dir=emb_out_dir, return_all=True, _format_esm=_format_esm)
    dist = get_dist_map(ec_id_dict, emb, device, dtype)

    if dont_save:
        return ids, emb, dist

    if cache_dir is not None:
        ensure_dirs(path+'/'+cache_dir)
        dump_info(ids, ecs, dist, emb, path=path, cache_dir=cache_dir, train_file=train_file, use_old_naming_convention=use_old_naming_convention)

def dump_info(ids, ecs, dist, emb, path='./', cache_dir='/distance_map/', train_file='split100', use_old_naming_convention=False):
    ensure_dirs(path+'/'+cache_dir)

    if use_old_naming_convention:
        pickle.dump(ids, open(path+'/'+cache_dir+'/'+train_file+'_ids.pkl', 'wb'))
        pickle.dump(ecs, open(path+'/'+cache_dir+'/'+train_file+'_ecs.pkl', 'wb'))
        pickle.dump(dist, open(path+'/'+cache_dir+'/'+train_file+'.pkl', 'wb'))
        pickle.dump(emb, open(path+'/'+cache_dir+'/'+train_file+'_esm.pkl', 'wb'))
    else:
        pickle.dump(ids, open(path+'/'+cache_dir+'/'+train_file+'_ids.pkl', 'wb'))
        pickle.dump(ecs, open(path+'/'+cache_dir+'/'+train_file+'_ecs.pkl', 'wb'))
        pickle.dump(dist, open(path+'/'+cache_dir+'/'+train_file+'_dist.pkl', 'wb'))
        pickle.dump(emb, open(path+'/'+cache_dir+'/'+train_file+'_emb.pkl', 'wb'))

def prepare_infer_fasta(name, path='./', emb_out_dir='/emb_data/'):
    retrieve_esm1b_embedding(name, path=path, esm_out=path+emb_out_dir)
    csvfile = open(path+'/'+name+'.csv', 'w', newline='')
    csvwriter = csv.writer(csvfile, delimiter = '\t')
    csvwriter.writerow(['Entry', 'EC number', 'Sequence'])
    fastafile = open(path+'/'+name+'.fasta', 'r')
    for i in fastafile.readlines():
        if i[0] == '>':
            csvwriter.writerow([i.strip()[1:], ' ', ' '])
    
def mutate(seq: str, position: int) -> str:
    seql = seq[ : position]
    seqr = seq[position+1 : ]
    seq = seql + '*' + seqr
    return seq

def mask_sequences(single_id, csv_name, fasta_name, path='./', mask_token='<mask>') :
    csv_file = open(path+'/'+ csv_name + '.csv')
    csvreader = csv.reader(csv_file, delimiter = '\t')
    output_fasta = open(path+'/' + fasta_name + '.fasta','w')
    single_id = set(single_id)

    for i, rows in enumerate(csvreader):
        if rows[0] in single_id:
            for j in range(10):
                seq = rows[2].strip()
                mu, sigma = .10, .02 # mean and standard deviation
                s = np.random.normal(mu, sigma, 1)
                mut_rate = s[0]
                times = math.ceil(len(seq) * mut_rate)
                for k in range(times):
                    position = random.randint(1 , len(seq) - 1)
                    seq = mutate(seq, position)
                seq = seq.replace('*', mask_token)
                output_fasta.write('>' + rows[0] + '_' + str(j) + '\n')
                output_fasta.write(seq + '\n')

def mutate_single_seq_ECs(id_ec=None, ec_id=None, name='split100', path='./', emb_out_dir='/emb_data/', mask_token='<mask>'):
    
    if id_ec == None or ec_id == None:
        id_ec, ec_id =  get_ec_id_dict(path+'/'+name+'.csv')
    
    single_ec = set()
    for ec in ec_id.keys():
        if len(ec_id[ec]) == 1:
            single_ec.add(ec)
    single_id = set()
    for _id in id_ec.keys():
        for ec in id_ec[_id]:
            if ec in single_ec and not os.path.exists(path+'/'+emb_out_dir+_id+'_1.pt'):
                single_id.add(_id)
                break

    print("Number of EC numbers with only one sequences:",len(single_ec))
    print("Number of single-seq EC number sequences need to mutate: ",len(set(single_id)))
    print("Number of single-seq EC numbers already mutated: ", len(single_ec) - len(single_id))
    mask_sequences(single_id, name, name+'_single_seq_ECs', path=path, mask_token=mask_token)
    fasta_name = name+'_single_seq_ECs'
    return fasta_name

def mutate_incorrect_seq_ECs(ids, experimental_results, name='split100', path='./', emb_out_dir='/emb_data/', mask_token='<mask>'):

    incorrect_id = set()
    count = 0
    for _id, result in zip(ids, experimental_results):
        if not result: #incorrect result and sequence has not already been mutated
            count += 1
            if not os.path.exists(path+'/'+emb_out_dir+'/'+_id+'_1.pt'):
                incorrect_id.add(_id)

    print("Number of incorrect-seq EC number sequences need to mutate: ",len(set(incorrect_id)))
    print("Number of single-seq EC numbers already mutated: ", count - len(incorrect_id))
    mask_sequences(incorrect_id, name, name+'_incorrect_seq_ECs', path=path, mask_token=mask_token)
    fasta_name = name+'_incorrect_seq_ECs'
    return fasta_name

