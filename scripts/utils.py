
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import json
import os, csv, sys, glob
from scipy.spatial.distance import jaccard, cosine, euclidean, sqeuclidean
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn import metrics
from collections import Counter
import faiss
import pickle

codon_table = {'TCA': 'S', 'AAT': 'N', 'TGG': 'W', 'GAT': 'D', 'GAA': 'E', 'TTC': 'F', 'CCG': 'P',
           'ACT': 'T', 'GGG': 'G', 'ACG': 'T', 'AGA': 'R', 'TTG': 'L', 'GTC': 'V', 'GCA': 'A',
           'TGA': '*', 'CGT': 'R', 'CAC': 'H', 'CTC': 'L', 'CGA': 'R', 'GCT': 'A', 'ATC': 'I',
           'ATA': 'I', 'TTT': 'F', 'TAA': '*', 'GTG': 'V', 'GCC': 'A', 'GAG': 'E', 'CAT': 'H',
           'AAG': 'K', 'AAA': 'K', 'GCG': 'A', 'TCC': 'S', 'GGC': 'G', 'TCT': 'S', 'CCT': 'P',
           'GTA': 'V', 'AGG': 'R', 'CCA': 'P', 'TAT': 'Y', 'ACC': 'T', 'TCG': 'S', 'ATG': 'M',
           'TTA': 'L', 'TGC': 'C', 'GTT': 'V', 'CTT': 'L', 'CAG': 'Q', 'CCC': 'P', 'ATT': 'I',
           'ACA': 'T', 'AAC': 'N', 'GGT': 'G', 'AGC': 'S', 'CGG': 'R', 'TAG': '*', 'CGC': 'R',
           'AGT': 'S', 'CTA': 'L', 'CAA': 'Q', 'CTG': 'L', 'GGA': 'G', 'TGT': 'C', 'TAC': 'Y',
           'GAC': 'D'}

def get_codons(seq):
    return ' '.join([seq[s:s+3] for s in range(0, len(seq), 3)])

def get_codon_list(sequences):
    return np.array([get_codons(seq) for seq in sequences], dtype=object)

def clean(df, col):
    return df[df[col].map(lambda d: len(str(d)) > 9 and len(str(d)) % 3 == 0 and set(str(d)) == set('ACGT'))]

def get_amino_acids(seq):
    return ''.join([codon_table[codon] if codon in codon_table.keys() else '' for codon in seq.split(' ')]) #drops non-existant codons but could also replace with 'X'

def convert_to_AA(df, col):
    codons = get_codon_list(df[col])
    AA = [get_amino_acids(seq) for seq in codons]
    return AA

#google AI assistant
def create_hierarchy_dict(labels, delimiter="/"):
    """
    Creates a hierarchical dictionary from a list of labels.

    Args:
        labels: A list of strings representing the labels.
        delimiter: The delimiter used to separate levels in the labels.

    Returns:
        A dictionary representing the hierarchy.
    """
    hierarchy = {}
    for label in labels:
        parts = label.split(delimiter)
        current_level = hierarchy
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                current_level[part] = ''
            else:
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]

    return hierarchy

# Example usage:
#labels = ["A/B/C", "A/B/D", "A/E", "F/G"]
#hierarchy_dict = create_hierarchy_dict(labels)
#print(hierarchy_dict)
# Expected output: {'A': {'B': {'C': None, 'D': None}, 'E': None}, 'F': {'G': None}}

def convert_string_to_list(string):
    return list(map(int, string.replace('n', '1000').split('.')))

def parse_infer_file(filename):
    ids = []
    ecs = []
    with open(filename, 'r') as f:
        lines = f.readlines()

        for line in lines:
            ID, _, EC = line.partition(',')

            ids.append(ID)

            ecl = []
            ECs = EC.split(',')
            for ec in ECs:
                true_ec = ec.partition('/')[0].partition(':')[-1]
                ecl.append(true_ec)

            ecs.append(ecl)

    df = pd.DataFrame()
    df['Entry'] = ids
    df['EC number'] = ecs

    return df

def read_list(filepath):
    with open(filepath, 'r') as f:
        txt = f.read()

    return list(ast.literal_eval(txt))

def save_predictions_to_csv(predictions, csv_file_path, output_csv_file_path, test_mode=False, nrows=50):
    # Load the existing test data
    if test_mode:
        df = pd.read_csv(csv_file_path, nrows=nrows)
    else:
        df = pd.read_csv(csv_file_path)

    # Add a new column for predictions
    df['Predicted_Label'] = predictions

    # Save the updated DataFrame to a new CSV
    df.to_csv(output_csv_file_path, index=False)

def save_metrics(metrics, file_path):
    """
    Save metrics to a JSON file.

    Args:
        metrics (dict): A dictionary of metrics to save.
        file_path (str): The path to the file where metrics should be saved.
    """
    with open(file_path, 'w') as f:
        json.dump(metrics, f)

def save_model(model, path):
    """
    Save the model state dictionary.

    Args:
        model (torch.nn.Module): The trained model.
        path (str): The path to save the model file.
    """
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def dump_seqs(csv_path, cache_dir='/distance_map/', id_list=None):
    data_name = csv_path.split('/')[-1].split('.')[0]
    data_path = csv_path.rpartition('/')[0]
    
    if id_list is None:
        id_list = np.array(pickle.load(open(data_path+'/'+cache_path+'/'+data_name+'_ids.pkl', 'rb')))
    
    df = pd.read_csv(csv_path, sep='\t', header=0)

    df = df.set_index('Entry')
    df = df.loc[id_list] #sorting index into same order as id_list
    seqs = df['Sequence']

    ensure_dirs(data_path+'/'+cache_dir)
    pickle.dump(seqs, open(data_path+'/'+cache_dir+'/'+data_name+'_seqs.pkl', 'wb'))

    return data_path+'/'+cache_dir+'/'+data_name+'_seqs.pkl'

def read_fasta(fasta_path):
    seq_ids = []
    seqs = []

    with open(fasta_path, 'r') as fastafile:
        for line in fastafile.readlines():
            if line[0] == '>':
                seq_ids.append(line.strip()[1:])
            else:
                seqs.append(line.strip())

    assert(len(seq_ids) == len(seqs))
    return seq_ids, seqs

def overlap_score(neighbor_ec: str, true_ec: str) -> tuple[int, int, int]:
    """Calculates the overlap score between two Enzyme Commission (EC) numbers.

    The overlap score is the number of levels (separated by periods) that match exactly
    between the two EC numbers. This function can be useful for comparing the specificity
    of EC numbers in enzyme classification.

    Args:
        neighbor_ec (str): The first EC number (e.g., "1.2.3.4").
        true_ec (str): The second EC number (e.g., "1.2.3.5").

    Returns:
        tuple[int, int, int]: A tuple containing:
            - overlap_score (int): The number of matching levels from the beginning.
            - alpha (int): The length (number of levels) of the first EC number.
            - beta (int): The length (number of levels) of the second EC number.

    Examples:
        >>> overlap_score("1.2.3.4", "1.2.3.5")
        (3, 4, 4)

         >>> overlap_score("1", "1.2")
        (1, 1, 2)

        >>> overlap_score("invalid_ec", "1.2.3")  # Raises ValueError
        ValueError: Invalid EC number format.
    """
    if not isinstance(neighbor_ec, str) or not '.' in neighbor_ec:
        raise ValueError("Invalid EC number format.")
    if not isinstance(true_ec, str) or not '.' in true_ec:
        raise ValueError("Invalid EC number format.")

    levels1 = neighbor_ec.split(".")
    levels2 = true_ec.split(".")
    alphaandbetabet = 0
    alpha = len(levels1)
    beta = len(levels2)
    for i in range(min(len(levels1), len(levels2))):
        if levels1[i] == levels2[i]:
            alphaandbetabet += 1
        else:
            break
    return alphaandbetabet, alpha, beta

def hiclass_score_metrics(true_ecs: list[str], neighboring_ecs: list[list[str]]) -> tuple[float, float, float]:
    """Calculates HiCLASS score metrics for a set of true EC numbers and their neighboring EC numbers.

    The HiCLASS (Hierarchical Classification) score metrics are used to evaluate the
    performance of classification algorithms in a hierarchical structure. In the context of
    enzyme classification, these metrics assess how well predicted EC numbers (neighboring_ecs)
    match the true EC numbers (true_ecs).

    Args:
        true_ecs (list[str]): A list of true Enzyme Commission (EC) numbers (e.g., ["1.2.3.4", "5.6.7.8"]).
        neighboring_ecs (list[list[str]]): A list of lists of neighboring EC numbers for each true EC number.
            Each inner list contains candidate EC numbers predicted for the corresponding true EC number.

    Returns:
        tuple[float, float, float]: A tuple containing the following HiCLASS score metrics:
            - hP (float): HiCLASS precision, representing the proportion of correctly classified EC numbers.
            - hR (float): HiCLASS recall, representing the completeness of classifications.
            - hF (float): HiCLASS F1-score, the harmonic mean of precision and recall.

    Raises:
        ValueError: If the lengths of `true_ecs` and `neighboring_ecs` do not match,
            indicating a mismatch between the number of true EC numbers and their corresponding neighboring EC numbers.

    Examples:
        >>> true_ecs = ["1.2.3.4", "5.6.7.8"]
        >>> neighboring_ecs = [["1.2.3.4", "1.2.3.5"], ["5.6.7.7", "5.6.7.9"]]
        >>> hP, hR, hF = hiclass_score_metrics(true_ecs, neighboring_ecs)
        >>> print(f"HiCLASS precision (hP): {hP:.4f}")
        >>> print(f"HiCLASS recall (hR): {hR:.4f}")
        >>> print(f"HiCLASS F1-score (hF): {hF:.4f}")
    """
    if len(true_ecs) != len(neighboring_ecs):
        raise ValueError("Length mismatch between true_ecs and neighboring_ecs lists.")

    intersection = 0
    alpha = 0
    beta = 0
    for true_ec , neighbor_ecs in zip(true_ecs, neighboring_ecs):
        max_score = 0
        for neighbor_ec in neighbor_ecs:
            #print(type(neighbor_ec), type(true_ec))
            alphaandbetabet, a, b = overlap_score(neighbor_ec, true_ec)
            max_score = max(max_score, alphaandbetabet)
        intersection += max_score
        alpha += a
        beta += b

    hP = intersection/alpha
    hR = intersection/beta

    if hR == 0 and hP == 0:
        hF = 0.0
    else:
        hF = 2*hP*hR/(hP+hR)

    return  hP, hR, hF

def level_accuracy(accuracies: dict[int, int], ec: str, retrieved_ec: str) -> dict[int, int]:
    """Calculates accuracy at different levels of the Enzyme Commission (EC) number hierarchy.

    This function updates a dictionary `accuracies` that tracks the number of correctly
    retrieved EC numbers at each level (key) in the hierarchy. It iterates through the
    levels in the `accuracies` dictionary and checks if the first `level` levels (inclusive)
    of the true EC number (`ec`) match the corresponding levels of a retrieved EC number (`retrieved_ec`).

    Args:
        accuracies (dict[int, int]): A dictionary where keys are levels (integers) and
            values are the number of correctly retrieved EC numbers at that level.
        ec (str): The true EC number (e.g., "1.2.3.4").
        retrieved_ec (str): A List of retrieved EC number to be compared against the true EC number.

    Returns:
        dict[int, int]: The updated `accuracies` dictionary with potentially incremented counts
            for correctly retrieved EC numbers at different levels.
    """
    for level, accuracy in accuracies.items():
        for r_ec in retrieved_ec:
            if ec.split('.')[:level] == r_ec.split('.')[:level]:
                accuracies[level] += 1
                break
    return accuracies


def top_k_retrieval(train_ec: list[str], test_ec: list[str], train_embeddings: torch.Tensor, test_embeddings: torch.Tensor) -> dict[int, tuple[dict[int, float], tuple[float, float, float]]]:
    """
    Performs top-k retrieval of EC numbers using Faiss and calculates accuracy and HiCLASS metrics.

    This function retrieves the top-k nearest neighbors (most similar EC numbers) for
    each EC number in the test set using Faiss for efficient nearest neighbor search.
    It then calculates accuracy at different levels of the EC number hierarchy and
    HiCLASS score metrics to evaluate the retrieval performance.

    Args:
        train_ec (list[str]): A list of true EC numbers in the training set.
        test_ec (list[str]): A list of true EC numbers for which to retrieve neighbors.
        train_embeddings (torch.Tensor): A tensor of embeddings for the training set EC numbers.
            The tensor is expected to have dimensions (num_train_ec, embedding_dim).
        test_embeddings (torch.Tensor): A tensor of embeddings for the test set EC numbers.
            The tensor is expected to have dimensions (num_test_ec, embedding_dim).

    Returns:
        dict[int, tuple[dict[int, float], tuple[float, float, float]]]: A dictionary where keys are
            k values (e.g., 1, 3, 5) and values are tuples containing:
                - accuracies (dict[int, float]): A dictionary where keys are levels (integers)
                    and values are the accuracy (proportion of correctly retrieved EC numbers)
                    at that level.
                - hiclass_metric (tuple[float, float, float]): A tuple containing HiCLASS score metrics
                    (precision, recall, F1-score) for the top-k retrieval results.
    """
    print(f"Data dimensions: train_ec={len(train_ec)}, test_ec={len(test_ec)}, "
            f"train_embeddings={train_embeddings.size()}, test_embeddings={test_embeddings.size()}")
    results = {}
    index = faiss.IndexFlatL2(train_embeddings.size(1))  # Use L2 distance for cosine similarity
    index.add(train_embeddings.cpu().detach().numpy())  # Add embeddings to the index
    K = [1,3,5]
    for k in K:
        distances, retrieval_indices = index.search(test_embeddings.cpu().detach().numpy(), k)
        retrieval_results = []
        for i, retrieved_idxs in enumerate(retrieval_indices):
            #         print(retrieved_idxs)
            retrieved_ec_numbers = [train_ec[idx] for idx in retrieved_idxs]
            retrieval_results.append(retrieved_ec_numbers)
        accuracies = {} # initialize ec level accuracy
        for level in range(4):
            accuracies.setdefault(level+1, 0)
        for i, retrieved_ec_numbers in enumerate(retrieval_results):
            accuracies = level_accuracy(accuracies, test_ec[i], retrieved_ec_numbers)
        for level, total_correct in accuracies.items():
            accuracies[level] /= len(test_ec)
        #calculate hiclass metrics
        hiclass_metric = hiclass_score_metrics(test_ec, retrieval_results)
        results[k] = accuracies, hiclass_metric
    return results


def get_embedding_stats(embeddings: torch.Tensor):
    """
    This function extracts class token embedding, minimum, maximum and average 
    for each sequence from the encoder outputs.
    Args:
        embeddings (torch.Tensor): Encoder outputs of shape (batch_size, sequence_length, embedding_dim)
    Returns:
        tuple: A tuple containing three tensors:
            - class_token_embeddings (torch.Tensor): Class token embeddings of shape (batch_size, embedding_dim)
            - min_embeddings (torch.Tensor): Minimum embedding per sequence of shape (batch_size, embedding_dim)
            - max_embeddings (torch.Tensor): Maximum embedding per sequence of shape (batch_size, embedding_dim)
            - avg_embeddings (torch.Tensor): Average embedding per sequence of shape (batch_size, embedding_dim)
    """
    class_token_embeddings = embeddings[:, 0, :]# Extract class token embeddings (assuming it's at index 0 for each sequence)
    min_embeddings, _ = torch.min(embeddings, dim=1)
    max_embeddings, _ = torch.max(embeddings, dim=1)
    avg_embeddings = torch.mean(embeddings, dim=-2) # torch.sum(attention_mask*embeddings, axis=-2)/torch.sum(attention_mask, axis=1)
    return class_token_embeddings, min_embeddings, max_embeddings, avg_embeddings


def compute_cosine_similarity_in_chunks(embeddings: torch.Tensor, chunk_size=1000) -> torch.Tensor:
    """Computes cosine similarity matrix for a large embedding tensor in chunks for memory efficiency.

    This function calculates the pairwise cosine similarity between all rows (vectors)
    in the input embedding tensor (`embeddings`). Due to memory constraints when dealing
    with large datasets, the computation is performed in chunks to avoid loading the
    entire matrix into memory at once.

    Args:
        embeddings (torch.Tensor): A 2D tensor of embeddings (num_embeddings x embedding_dim).
        chunk_size (int, optional): The size of each chunk to process. Defaults to 1000.

    Returns:
        torch.Tensor: A 2D tensor of cosine similarities between all embedding pairs (num_embeddings x num_embeddings).
    """
    n = embeddings.size(0)
    similarity_matrix = torch.zeros((n, n))
    for i in range(0, n, chunk_size):
        for j in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            end_j = min(j + chunk_size, n)
            chunk1 = embeddings[i:end_i]
            chunk2 = embeddings[j:end_j]
            similarity_chunk = torch.nn.functional.cosine_similarity(chunk1.unsqueeze(1), chunk2.unsqueeze(0), dim=2)
            similarity_matrix[i:end_i, j:end_j] = similarity_chunk.cpu()
    return similarity_matrix


def dist_matrix_chunks(embeddings: np.ndarray, chunk_size=1000) -> np.ndarray:
    """
    Computes pairwise cosine distance matrix for a large embedding array in chunks for memory efficiency.

    This function calculates the cosine distance between all pairs of rows (vectors)
    in the input embedding array (`embeddings`). Due to memory constraints when dealing
    with large datasets, the computation is performed in chunks to avoid loading the
    entire matrix into memory at once.

    Args:
        embeddings (np.ndarray): A 2D NumPy array of embeddings (num_embeddings x embedding_dim).
        chunk_size (int, optional): The size of each chunk to process. Defaults to 1000.

    Returns:
        np.ndarray: A 2D NumPy array of cosine distances between all embedding pairs (num_embeddings x num_embeddings).
    """

    n = embeddings.shape[0]
    distance_matrix = np.zeros((n, n))
    for i in range(0, n, chunk_size):
        for j in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            end_j = min(j + chunk_size, n)
            chunk1 = embeddings[i:end_i]
            chunk2 = embeddings[j:end_j]
            distance_chunk = cosine(embeddings[i].cpu().detach().numpy(), embeddings[j].cpu().detach().numpy())
            distance_matrix[i:end_i, j:end_j] = distance_chunk
    return distance_matrix

