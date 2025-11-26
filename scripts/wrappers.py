from dal_toolbox.metrics import entropy_from_logits, entropy_from_probas, ensemble_log_softmax, ensemble_entropy_from_logits
from dal_toolbox.models.utils.base import BaseModule
from dal_toolbox.active_learning import ActiveLearningDataModule
from dal_toolbox.active_learning.strategies import Query, UncertaintySampling, LeastConfidentSampling, EntropySampling, MarginSampling, BayesianEntropySampling, BayesianLeastConfidentSampling, BayesianMarginSampling, Badge, BALDSampling, BatchBALDSampling, TypiClust
from dal_toolbox.active_learning.strategies.typiclust import kmeans, get_nn, get_mean_nn_dist, calculate_typicality

from Bio import Align

import pickle
from torch.utils.data import DataLoader
import torch
from scipy.stats import entropy
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist
import numpy as np
import pandas as pd

class RandomSampling(UncertaintySampling):
    def get_utilities(self, logits):
        return torch.flatten(torch.from_numpy(np.random.rand(1, logits.shape[0])))

    @torch.no_grad()
    def query(
        self,
        *,
        model: BaseModule,
        al_datamodule: ActiveLearningDataModule,
        acq_size: int,
        return_utilities: bool = False,
        # forward_kwargs: dict = None, TODO
        **kwargs
    ):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(
            subset_size=self.subset_size)
        logits = model.get_logits(unlabeled_dataloader)  # , **forward_kwargs)

        scores = self.get_utilities(logits)
        top_scores, indices = scores.topk(acq_size)

        #print(scores, indices)
        actual_indices = [unlabeled_indices[i] for i in indices]

        if return_utilities:
            return actual_indices, scores

        return actual_indices

class QBC(Query):
    def __init__(self, learners_list, subset_size=None, random_seed=None):
        super().__init__(random_seed=random_seed)
        self.learners_list = learners_list
        self.subset_size = subset_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @torch.no_grad()
    def query(
        self,
        *,
        model: BaseModule,
        al_datamodule: ActiveLearningDataModule,
        acq_size: int,
        return_utilities: bool = False,
        # forward_kwargs: dict = None, TODO
        **kwargs
    ):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(
            subset_size=self.subset_size)
        logits = model.get_logits(unlabeled_dataloader)  # , **forward_kwargs)

        scores = self.get_utilities(logits)
        top_scores, indices = scores.topk(acq_size)

        actual_indices = [unlabeled_indices[i] for i in indices]
        
        if return_utilities:
            return actual_indices, scores

        return actual_indices

    def get_utilities(self, logits):
        scores_list = []
        for learner in self.learners_list:
            scores = learner.get_utilities(logits)
            scores_list.append(scores)

        scores_list = torch.stack(scores_list, dim=0)

        #take max disagreement of committee
        disagreement = self.calculate_disagreement(scores_list.detach().cpu().numpy(), logits.shape[0])
        return torch.from_numpy(disagreement).to(self.device)

    #based on modAL package's KL-max-disagreement
    def calculate_disagreement(self, scores_list, X_shape):
        p_consensus = np.mean(scores_list, axis=1)
        learner_KL_div = np.zeros(shape=(X_shape, len(self.learners_list)))

        #for learner_idx, _ in enumerate(self.learners_list):
        for i in range(X_shape):
            learner_KL_div[i, :] = entropy(np.transpose(scores_list[:, i]), qk=np.transpose(p_consensus))

        return np.max(learner_KL_div, axis=1)


class BioInspiredSampling(Query):
    def __init__(self, learner_list, seq_path, subset_size=None, random_seed=None, match=2, mismatch=-1, gap=-2, gap_ext=0):
        super().__init__(random_seed=random_seed)
        self.learners_list = learner_list #can be single or multiple learners
        self.seq_path = seq_path
        self.subset_size = subset_size
        self.aligner = Align.PairwiseAligner()
        self.aligner.match_score = match
        self.aligner.mismatch_score = mismatch
        self.aligner.open_gap_score = gap
        self.aligner.extend_gap_score = gap_ext
        self.aligner.mode = "local" #smith waterman
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def alignment_score(self, seqs):

        col = len(seqs)
        num_elements = int((col**2 - col) / 2)
        
        distance_matrix = [0]*num_elements #upper triangle only to save memory 
        distance_matrix = np.array(distance_matrix, dtype=np.int16)
        average_distance = [0]*len(seqs)
        
        offset = 0
        for i, seq1 in enumerate(seqs):
            offset = offset + i + 1 #to access correctly upper triangle
            for j, seq2 in enumerate(seqs):
                if i >= j: #ignore diagonal and lower triangle
                    continue

                distance_matrix[((i*col)+j)-offset] = int(self.aligner.score(seq1, seq2))
                
            average_distance[i] = sum(distance_matrix[((i*col)+j)-offset] for j in range(col) if j > i) / (col - 1)

        average_distance = np.array(average_distance)
        normalized_distance = (average_distance - np.min(average_distance))/np.ptp(average_distance)
        inverse_distance = 1.0 - normalized_distance #want furthest alignments on average

        return inverse_distance
    
    @torch.no_grad()
    def query(
        self,
        *,
        model: BaseModule,
        al_datamodule: ActiveLearningDataModule,
        acq_size: int,
        return_utilities: bool = False,
        # forward_kwargs: dict = None, TODO
        **kwargs
    ):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(
            subset_size=self.subset_size)
        logits = model.get_logits(unlabeled_dataloader)  # , **forward_kwargs)
        
        #FIXME: get seqs and test alignment code
        seqs_file = pickle.load(open(self.seq_path, 'rb'))
        seqs = [seqs_file[i] for i in unlabeled_indices]

        alignment_scores = self.alignment_score(seqs)
        
        scores = self.get_utilities(logits, alignment_scores=alignment_scores)
        top_scores, indices = scores.topk(acq_size)

        actual_indices = [unlabeled_indices[i] for i in indices]
        
        if return_utilities:
            return actual_indices, scores

        return actual_indices

    def get_utilities(self, logits, alignment_scores=None):
        scores_list = []
        for learner in self.learners_list:
            scores = learner.get_utilities(logits)
            scores_list.append(scores)

        scores_list.append(torch.tensor(alignment_scores).to(self.device)) #append bio-inspired alignment score to weight uncertainty
        scores_list = torch.stack(scores_list, dim=0)

        #take max disagreement of committee
        disagreement = self.calculate_disagreement(scores_list.detach().cpu().numpy(), logits.shape[0])
        return torch.from_numpy(disagreement).to(self.device)

    #based on modAL package's KL-max-disagreement
    def calculate_disagreement(self, scores_list, X_shape):
        p_consensus = np.mean(scores_list, axis=1)
        learner_KL_div = np.zeros(shape=(X_shape, len(self.learners_list)))

        #for learner_idx, _ in enumerate(self.learners_list):
        for i in range(X_shape):
            learner_KL_div[i, :] = entropy(np.transpose(scores_list[:, i]), qk=np.transpose(p_consensus))

        return np.max(learner_KL_div, axis=1)

class MyTypiClust(TypiClust):
    #adjusted parameters here
    MIN_CLUSTER_SIZE = 3
    MAX_NUM_CLUSTERS = 10000
    K_NN = 20

    def __init__(self, subset_size=None, random_seed=None):
        super().__init__(random_seed=random_seed)
        self.subset_size = subset_size

    @torch.no_grad()
    def query(self,
              *,
              model: BaseModule,
              al_datamodule: ActiveLearningDataModule,
              acq_size: int,
              return_utilities: bool = False,
              **kwargs):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(self.subset_size)
        labeled_dataloader, labeled_indices = al_datamodule.labeled_dataloader()

        num_clusters = min(len(labeled_indices) + acq_size, self.MAX_NUM_CLUSTERS)

        unlabeled_features = model.get_representations(unlabeled_dataloader)
        if len(labeled_indices) > 0:
            labeled_features = model.get_representations(labeled_dataloader)
        else:
            labeled_features = torch.Tensor([])

        features = torch.cat((labeled_features, unlabeled_features))
        clusters = kmeans(features, num_clusters=num_clusters)

        labels = clusters.copy()
        existing_indices = np.arange(len(labeled_indices))

        # counting cluster sizes and number of labeled samples per cluster
        cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
        cluster_labeled_counts = np.bincount(labels[existing_indices], minlength=len(cluster_ids))
        clusters_df = pd.DataFrame(
            {'cluster_id': cluster_ids, 'cluster_size': cluster_sizes, 'existing_count': cluster_labeled_counts,
             'neg_cluster_size': -1 * cluster_sizes})
        # drop too small clusters
        clusters_df = clusters_df[clusters_df.cluster_size > self.MIN_CLUSTER_SIZE]
        # sort clusters by lowest number of existing samples, and then by cluster sizes (large to small)
        clusters_df = clusters_df.sort_values(['existing_count', 'neg_cluster_size'])
        labels[existing_indices] = -1

        selected = []
        #typicality_scores = []
        for i in range(acq_size):
            cluster = clusters_df.iloc[i % len(clusters_df)].cluster_id
            indices = (labels == cluster).nonzero()[0]
            rel_feats = features[indices]
            # in case we have too small cluster, calculate density among half of the cluster
            typicality = calculate_typicality(rel_feats, min(self.K_NN, len(indices) // 2))
            #typicality_scores.append(typicality)
            idx = indices[typicality.argmax()]
            selected.append(idx)
            labels[idx] = -1

        selected = np.array(selected)
        actual_indices = [unlabeled_indices[i - len(labeled_indices)] for i in selected]

        if return_utilities:
            return actual_indices, torch.tensor(clusters)

        return actual_indices

class MyBadge(Query):
    def __init__(self, subset_size=None):
        super().__init__()
        self.subset_size = subset_size

    def query(self, *, model, al_datamodule, acq_size, return_utilities=False, **kwargs):
        unlabeled_dataloader, unlabeled_indices = al_datamodule.unlabeled_dataloader(subset_size=self.subset_size)

        grad_embedding = model.get_grad_representations(unlabeled_dataloader)
        chosen, cluster_probs = self.kmeans_plusplus(grad_embedding.numpy(), acq_size, rng=self.rng)

        actual_indices = [unlabeled_indices[idx] for idx in chosen]

        if return_utilities:
            return actual_indices, torch.from_numpy(cluster_probs)

        return actual_indices

    def kmeans_plusplus(self, X, n_clusters, rng):
        # Start with highest grad norm since it is the "most uncertain"
        grad_norm = np.linalg.norm(X, ord=2, axis=1)
        idx = np.argmax(grad_norm)

        #all_distances = pairwise_distances(X, X) #preomputing takes too much space

        indices = [idx]
        centers = [X[idx]]
        dist_mat = []
        total_p = np.zeros(len(X))
        for _ in range(1, n_clusters):
            # Compute the distance of the last center to all samples
            dist = np.sqrt(np.sum((X - centers[-1])**2, axis=-1))
            #dist = all_distances[indices[-1]]

            dist_mat.append(dist)
            # Get the distance of each sample to its closest center
            min_dist = np.min(dist_mat, axis=0)
            min_dist_squared = min_dist**2
            if np.all(min_dist_squared == 0):
                raise ValueError('All distances to the centers are zero!')
        
            # sample idx with probability proportional to the squared distance
            p = min_dist_squared / np.sum(min_dist_squared)
            total_p = total_p+p

            if np.any(p[indices] != 0):
                print('Already sampled centers have probability', p)

            idx = rng.choice(range(len(X)), p=p.squeeze())
            indices.append(idx)
            centers.append(X[idx])
        
        return indices, total_p / len(X)

def get_sampling_active_learner(query_strategy='uncertainty'):
    if query_strategy == 'uncertainty':
        return LeastConfidentSampling()
    elif query_strategy == 'entropy':
        return EntropySampling()
    elif query_strategy == 'margin':
        return MarginSampling()
    elif query_strategy == 'random':
        return RandomSampling()
    else:
        raise ValueError('Please specify a valid query strategy')

def get_badge_active_learner():
    return MyBadge(subset_size=10000)

def get_committee_active_learner(learners_list=[LeastConfidentSampling(), EntropySampling(), MarginSampling()]):
    return QBC(learners_list)

def get_bayesian_active_learner(query_strategy='uncertainty'):
    if query_strategy == 'uncertainty':
        return BayesianLeastConfidentSampling(subset_size=10000)
    elif query_strategy == 'entropy':
        return BayesianEntropySampling(subset_size=10000)
    elif query_strategy == 'margin':
        return BayesianMarginSampling(subset_size=10000)
    else:
        raise ValueError('Please specify a valid query strategy')

def get_bald_active_learner(batch=False): #currently ignoring batch option -- maybe in future release
    if batch:
        return BatchBALDSampling(subset_size=10000)
        
    return BALDSampling(subset_size=10000)

def get_clust_active_learner():
    return MyTypiClust(subset_size=10000)

def get_bio_active_learner(seq_path=None, learners_list=[LeastConfidentSampling(), EntropySampling(), MarginSampling()]):
    return BioInspiredSampling(learners_list, seq_path, subset_size=10000)

