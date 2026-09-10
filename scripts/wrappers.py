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

    def calculate_disagreement(self, scores_list, X_shape):
        """Committee disagreement per pool instance.

        scores_list is (L, N): one utility per committee member per instance.

        The previous implementation had three defects, none of which raised:
          1. `np.mean(scores_list, axis=1)` averaged each MEMBER across the
             whole pool, giving an (L,) pool-level constant rather than the
             (N,) per-instance consensus the cited modAL method needs. It also
             made a score depend on pool composition, which drifts every AL
             round.
          2. `entropy(pk, qk)` on two length-L vectors returns a SCALAR, which
             then broadcast across the whole `learner_KL_div[i, :]` row, so the
             per-member axis carried no information and the final
             `np.max(..., axis=1)` reduced over identical copies.
          3. Members return utilities on incommensurable scales (measured:
             LeastConfident 0-0.50, Entropy 0-0.69, Margin 0-1.00), and
             `scipy.stats.entropy` renormalises them, so the result tracked
             scale artefacts. Empirically this INVERTED the ordering: the
             committee selected the most confident instances.

        Fix: z-score each member across the pool to put them on a common scale,
        then disagreement is the spread of member opinions on each instance.
        Scale-invariant and genuinely per-instance. (Correcting only the axis
        also restores the right ordering, but leaves the largest-range member
        dominating the max.)
        """
        scores_list = np.asarray(scores_list, dtype=np.float64)

        mu = scores_list.mean(axis=1, keepdims=True)
        sd = scores_list.std(axis=1, keepdims=True) + 1e-12
        z = (scores_list - mu) / sd

        return z.std(axis=0)


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

    def calculate_disagreement(self, scores_list, X_shape):
        """Committee disagreement per pool instance.

        scores_list is (L, N): one utility per committee member per instance.

        The previous implementation had three defects, none of which raised:
          1. `np.mean(scores_list, axis=1)` averaged each MEMBER across the
             whole pool, giving an (L,) pool-level constant rather than the
             (N,) per-instance consensus the cited modAL method needs. It also
             made a score depend on pool composition, which drifts every AL
             round.
          2. `entropy(pk, qk)` on two length-L vectors returns a SCALAR, which
             then broadcast across the whole `learner_KL_div[i, :]` row, so the
             per-member axis carried no information and the final
             `np.max(..., axis=1)` reduced over identical copies.
          3. Members return utilities on incommensurable scales (measured:
             LeastConfident 0-0.50, Entropy 0-0.69, Margin 0-1.00), and
             `scipy.stats.entropy` renormalises them, so the result tracked
             scale artefacts. Empirically this INVERTED the ordering: the
             committee selected the most confident instances.

        Fix: z-score each member across the pool to put them on a common scale,
        then disagreement is the spread of member opinions on each instance.
        Scale-invariant and genuinely per-instance. (Correcting only the axis
        also restores the right ordering, but leaves the largest-range member
        dominating the max.)
        """
        scores_list = np.asarray(scores_list, dtype=np.float64)

        mu = scores_list.mean(axis=1, keepdims=True)
        sd = scores_list.std(axis=1, keepdims=True) + 1e-12
        z = (scores_list - mu) / sd

        return z.std(axis=0)

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

        # Cap the cluster count by the data being clustered, not only by the
        # labelled count. num_clusters grows with len(labeled_indices) and
        # ignores how many points there are, so once the candidate pool is small
        # -- late in a high-coverage sweep, or whenever the caller restricts
        # unlabeled_indices, as target-directed acquisition does -- it asks for
        # more clusters than points. The singleton clusters that result give
        # calculate_typicality k = len(indices)//2 = 0 and NearestNeighbors
        # raises "Found array with 0 sample(s)". Observed at round 105 of a
        # 436-round sweep.
        n_points = len(labeled_indices) + len(unlabeled_indices)
        num_clusters = min(len(labeled_indices) + acq_size,
                           self.MAX_NUM_CLUSTERS,
                           max(1, n_points // 5))

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
            # k must be >= 1: a one- or two-member cluster gives
            # len(indices)//2 == 0 and NearestNeighbors rejects an empty
            # neighbourhood. Guard here too, since a degenerate cluster can
            # survive the num_clusters cap.
            if len(indices) == 0:
                continue
            if len(indices) == 1:
                # singleton cluster: nothing to rank, the lone member is the
                # pick. Reached whenever the candidate set is restricted (the
                # target-EC shortlist shrinks to ~10 in late rounds).
                idx = indices[0]
                selected.append(idx)
                labels[idx] = -1
                continue
            # k neighbours need k + 1 points to fit against.
            k = max(1, min(self.K_NN, len(indices) // 2, len(indices) - 1))
            typicality = calculate_typicality(rel_feats, k)
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

# ---------------------------------------------------------------------------#
# Pool-size guard
#
# Every acquisition strategy carries at least one size hyperparameter fixed at
# construction: acq_size per round, subset_size for the strategies that
# subsample, num_clusters derived from the labelled count. None of them know how
# large the candidate pool actually is, and all three raise cryptically when it
# is smaller than they assume:
#
#   scores.topk(acq_size)                -> "selected index k out of range"
#   rng.choice(..., replace=False)       -> "Cannot take a larger sample than
#                                            population"
#   kmeans into more clusters than points -> singleton clusters, then
#                                            "Found array with 0 sample(s)" in
#                                            NearestNeighbors
#
# That is reachable in ordinary use -- late in a high-coverage sweep, or whenever
# the caller restricts the unlabelled set, as target-directed acquisition does by
# filtering to sequences predicted to be the target EC. Rather than patch each
# strategy, wrap them: one place, every strategy, and the failure becomes either
# a clamp or an explicit error depending on what the caller asked for.
# ---------------------------------------------------------------------------#

POOL_SIZE_POLICY = 'clamp'   # set from --pool_size_policy in driver.py


class PoolSizeGuard(Query):
    """Clamp a strategy's size hyperparameters to the pool actually available.

    policy='clamp' (default) shrinks the round and logs it: acquiring fewer
    sequences than requested is the honest outcome when fewer exist, and is what
    a deployment would do.

    policy='error' refuses instead, for callers who would rather stop than
    silently take a short round -- appropriate when the batch size is the
    experimental variable and a short round would confound it.
    """

    def __init__(self, inner, policy='clamp'):
        super().__init__(random_seed=getattr(inner, 'random_seed', None))
        if policy not in ('clamp', 'error'):
            raise ValueError(f"policy must be 'clamp' or 'error', got {policy!r}")
        self.inner = inner
        self.policy = policy

    def __getattr__(self, name):          # delegate anything we do not define
        return getattr(self.__dict__['inner'], name)

    def query(self, *, model, al_datamodule, acq_size, **kwargs):
        n_avail = len(al_datamodule.unlabeled_indices)
        if n_avail == 0:
            raise ValueError('No unlabelled instances remain to acquire from.')

        if acq_size > n_avail:
            if self.policy == 'error':
                raise ValueError(
                    f'acq_size={acq_size} exceeds the {n_avail} unlabelled instances '
                    f'available. Pass a smaller --n_instances, or use '
                    f'--pool_size_policy clamp to acquire what remains.')
            print(f'[pool-guard] acq_size {acq_size} > {n_avail} available; '
                  f'acquiring {n_avail}')
            acq_size = n_avail

        # subset_size is read inside the strategy, so it has to be clamped on the
        # object rather than passed through. Restored afterwards so a short round
        # does not permanently shrink the strategy.
        original = getattr(self.inner, 'subset_size', None)
        if original is not None and original > n_avail:
            print(f'[pool-guard] subset_size {original} > {n_avail} available; '
                  f'using {n_avail} for this round')
            self.inner.subset_size = n_avail
        try:
            return self.inner.query(model=model, al_datamodule=al_datamodule,
                                    acq_size=acq_size, **kwargs)
        finally:
            if original is not None:
                self.inner.subset_size = original


def get_sampling_active_learner(query_strategy='uncertainty'):
    if query_strategy == 'uncertainty':
        return PoolSizeGuard(LeastConfidentSampling(), policy=POOL_SIZE_POLICY)
    elif query_strategy == 'entropy':
        return PoolSizeGuard(EntropySampling(), policy=POOL_SIZE_POLICY)
    elif query_strategy == 'margin':
        return PoolSizeGuard(MarginSampling(), policy=POOL_SIZE_POLICY)
    elif query_strategy == 'random':
        return PoolSizeGuard(RandomSampling(), policy=POOL_SIZE_POLICY)
    else:
        raise ValueError('Please specify a valid query strategy')

def get_badge_active_learner():
    return PoolSizeGuard(MyBadge(subset_size=10000), policy=POOL_SIZE_POLICY)

def get_committee_active_learner(learners_list=[LeastConfidentSampling(), EntropySampling(), MarginSampling()]):
    return PoolSizeGuard(QBC(learners_list), policy=POOL_SIZE_POLICY)

def get_bayesian_active_learner(query_strategy='uncertainty'):
    if query_strategy == 'uncertainty':
        return PoolSizeGuard(BayesianLeastConfidentSampling(subset_size=10000), policy=POOL_SIZE_POLICY)
    elif query_strategy == 'entropy':
        return PoolSizeGuard(BayesianEntropySampling(subset_size=10000), policy=POOL_SIZE_POLICY)
    elif query_strategy == 'margin':
        return PoolSizeGuard(BayesianMarginSampling(subset_size=10000), policy=POOL_SIZE_POLICY)
    else:
        raise ValueError('Please specify a valid query strategy')

def get_bald_active_learner(batch=False): #currently ignoring batch option -- maybe in future release
    if batch:
        return PoolSizeGuard(BatchBALDSampling(subset_size=10000), policy=POOL_SIZE_POLICY)
        
    return PoolSizeGuard(BALDSampling(subset_size=10000), policy=POOL_SIZE_POLICY)

def get_clust_active_learner():
    return PoolSizeGuard(MyTypiClust(subset_size=10000), policy=POOL_SIZE_POLICY)

def get_bio_active_learner(seq_path=None, learners_list=[LeastConfidentSampling(), EntropySampling(), MarginSampling()]):
    return PoolSizeGuard(BioInspiredSampling(learners_list, seq_path, subset_size=10000), policy=POOL_SIZE_POLICY)

