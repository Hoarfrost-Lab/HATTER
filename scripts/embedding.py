import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, RobertaModel, EsmModel, GPT2Tokenizer, GPT2Model, AutoConfig, AutoModelForCausalLM
from transformers.models.bert.configuration_bert import BertConfig
from tqdm.auto import tqdm
import sys

from BioTokenizer import load_tokenizer

#not compatible with DNABERT
try:
    from evo import Evo
    from evo.scoring import prepare_batch
except:
    print('')

def get_tokenizer_and_encoder(embedding_type='esm1b'):

    #LookingGlass_v2.0 model and tokenizer
    if embedding_type == 'lookingglassv2':
        tokenizer = load_tokenizer("/home/ab18558/functional-prediction-agent/models/LOLBERT_V1.10.2", model_max_length=512, padding='max_length', truncation=True)
        model = RobertaModel.from_pretrained("/home/ab18558/functional-prediction-agent/models/LOLBERT_V1.10.2")
        model.eval()

    #ESM-2 model and tokenizer
    elif embedding_type == 'esm2':
        esm_type = "esm2_t33_650M_UR50D" #using the same as CLEAN
        tokenizer = AutoTokenizer.from_pretrained("facebook/{}".format(esm_type), model_max_length=1024, padding='max_length', truncation=True)
        model = EsmModel.from_pretrained("facebook/{}".format(esm_type))
        model.eval()

    #ESM-1b model and tokenizer
    elif embedding_type == 'esm1b':
        esm_type = "esm1b_t33_650M_UR50S" #using the same as CLEAN
        tokenizer = AutoTokenizer.from_pretrained("facebook/{}".format(esm_type), model_max_length=1024, padding='max_length', truncation=True)
        model = EsmModel.from_pretrained("facebook/{}".format(esm_type))
        model.eval()

    #EVO model and tokenizer
    elif embedding_type == 'evo':
        model_name = 'togethercomputer/evo-1-131k-base'
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_config.use_cache = False
        model = AutoModelForCausalLM.from_pretrained(model_name, config=model_config, trust_remote_code=True)#e, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = "X"
        model.eval()

    #ProtGPT2 model and tokenizer
    elif embedding_type == 'protgpt2':
        tokenizer = GPT2Tokenizer.from_pretrained("nferruz/ProtGPT2", model_max_length=1024, padding='max_length', truncation=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2Model.from_pretrained("nferruz/ProtGPT2")
        model.eval()

    else:
        raise ValueError(f"Invalid embedding type: {self.embedding_type}")

    return tokenizer, model

    
def generate_embeddings(embedding_type, sequences, tokenizer, path='./', emb_out_dir=None, seq_ids=[], encoder=None, device='cpu', batch_size=64, tokenize_only=False):
    if embedding_type == 'lookingglassv2':
        embs = generate_lookingglassv2_embeddings(sequences, tokenizer, encoder, device, batch_size=batch_size, tokenize_only=tokenize_only)
    elif embedding_type == 'esm2':
        embs = generate_esm_embeddings(sequences, tokenizer, encoder, device, batch_size=batch_size, tokenize_only=tokenize_only)
    elif embedding_type == 'esm1b':
        embs = generate_esm_embeddings(sequences, tokenizer, encoder, device, batch_size=batch_size, tokenize_only=tokenize_only)
    elif embedding_type == 'evo':
        embs = generate_evo_embeddings(sequences, tokenizer, encoder, device, batch_size=batch_size, tokenize_only=tokenize_only)
    elif embedding_type == 'protgpt2':
        embs = generate_protgpt2_embeddings(sequences, tokenizer, encoder, device, batch_size=batch_size, tokenize_only=tokenize_only)
    else:
        raise ValueError(f"Invalid embedding type: {self.embedding_type}")
    
    if emb_out_dir is not None: 
        for seq_id, emb in zip(seq_ids, embs):
            torch.save(emb, path+'/'+emb_out_dir+'/'+seq_id+'.pt')

    return embs

def generate_lookingglassv2_embeddings(sequences, tokenizer, model=None, device='cpu', batch_size=64, tokenize_only=False):
    """
    Generate embeddings for a given DNA sequence using the LookingGlass_v2.0 model.
        
    Args:
        dna_sequence (str): The DNA sequence to encode.
        
    Returns:
        torch.Tensor: The mean-pooled embeddings produced by the LookingGlass_v2.0 model.
    """

    with torch.no_grad():

        embeddings = []
        if tokenize_only or model is None:
            input_ids = []
            attention_mask = []

        for chunk in tqdm(range(len(sequences) // batch_size+1)):
            seqs = sequences[(batch_size * chunk) : (batch_size * (chunk+1))]

            if not seqs:
                break
            else:
                inputs = tokenizer(seqs, return_tensors='pt', padding=True, truncation=True).to(device)

            if tokenize_only or model is None:
                input_ids += inputs['input_ids']
                attention_mask += inputs['attention_mask']
            else:
                model.to(device)
                hidden_states = model(**inputs).last_hidden_state
                embedding_mean = torch.mean(hidden_states, dim=1).squeeze(0)
                embedding_mean /= inputs.attention_mask.sum(dim=1, keepdim=True).float()
                embeddings += embedding_mean
        
    if tokenize_only or model is None:
        embeddings = {'input_ids' : input_ids, 'attention_mask' : attention_mask}

    return embeddings

def generate_esm_embeddings(sequences, tokenizer, model=None, device='cpu', batch_size=64, tokenize_only=False):
    """
    Generate embeddings for a given DNA sequence using the ESM-2 3 Billion parameter model.
        
    Args:
        dna_sequence (str): The DNA sequence to encode.
        
    Returns:
        torch.Tensor: The mean-pooled embeddings produced by the ESM-2 model.
    """

    with torch.no_grad():

        embeddings = []
        if tokenize_only or model is None:
            input_ids = []
            attention_mask = []

        for chunk in tqdm(range(len(sequences) // batch_size + 1)):
            seqs = sequences[(batch_size * chunk) : (batch_size * (chunk+1))]

            if not seqs:
                break
            else:
                inputs = tokenizer(seqs, return_tensors='pt', padding='max_length', truncation=True, max_length=1024).to(device)

            if tokenize_only or model is None:
                    input_ids += inputs['input_ids']
                    attention_mask += inputs['attention_mask']
            else:
                model.to(device)
                hidden_states = model(**inputs).last_hidden_state
                embedding_mean = torch.mean(hidden_states, dim=1).squeeze(0)
                embedding_mean /= inputs.attention_mask.sum(dim=1, keepdim=True).float()
                embeddings += embedding_mean
      
    if tokenize_only or model is None:
        embeddings = {'input_ids' : input_ids, 'attention_mask' : attention_mask}

    return embeddings

def generate_evo_embeddings(sequences, tokenizer, model=None, device='cpu', batch_size=64, tokenize_only=False):
    """
    Generate embeddings for a given DNA sequence using the EVO model.
        
    Args:
        dna_sequence (str): The DNA sequence to encode.
        
    Returns:
        torch.Tensor: The mean-pooled embeddings produced by the EVO model.
    """

    with torch.no_grad():

        embeddings = []
        if tokenize_only or model is None:
            input_ids = []
            attention_mask = []

        for chunk in tqdm(range(len(sequences) // batch_size + 1)):
            seqs = sequences[(batch_size * chunk) : (batch_size * (chunk+1))]

            if not seqs:
                break
            else:
                inputs = tokenizer(seqs, return_tensors='pt', padding='max_length', truncation=True, max_length=1024).to(device)

            if tokenize_only or model is None:
                    input_ids += inputs['input_ids']
                    attention_mask += inputs['attention_mask']
            else:
                model.to(device)
                hidden_states = model(inputs['input_ids'])[0]
                embedding_mean = torch.mean(hidden_states, dim=1).squeeze(0)
                embedding_mean /= inputs.attention_mask.sum(dim=1, keepdim=True).float()
                embeddings += embedding_mean

    if tokenize_only or model is None:
        embeddings = {'input_ids' : input_ids, 'attention_mask' : attention_mask}

    return embeddings

def generate_protgpt2_embeddings(sequences, tokenizer, model=None, device='cpu', batch_size=64, tokenize_only=False):
    """
    Generate embeddings for a given DNA sequence using the ProtGPT-2 model.
        
    Args:
        dna_sequence (str): The DNA sequence to encode.
        
    Returns:
        torch.Tensor: The mean-pooled embeddings produced by the ProtGPT-2 model.
    """

    with torch.no_grad():

        embeddings = []
        if tokenize_only or model is None:
            input_ids = []
            attention_mask = []

        for chunk in tqdm(range(len(sequences) // batch_size + 1)):
            seqs = sequences[(batch_size * chunk) : (batch_size * (chunk+1))]

            if not seqs:
                break
            else:
                inputs = tokenizer(seqs, return_tensors='pt', padding='max_length', truncation=True, max_length=1024).to(device)

            if tokenize_only or model is None:
                    input_ids += inputs['input_ids']
                    attention_mask += inputs['attention_mask']
            else:
                model.to(device)
                hidden_states = model(**inputs).last_hidden_state
                embedding_mean = torch.mean(hidden_states, dim=1).squeeze(0)
                embedding_mean /= inputs.attention_mask.sum(dim=1, keepdim=True).float()
                embeddings += embedding_mean

    if tokenize_only or model is None:
        embeddings = {'input_ids' : input_ids, 'attention_mask' : attention_mask}

    return embeddings

