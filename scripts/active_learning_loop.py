import numpy as np
import torch
import os
import random
import warnings
from tqdm import tqdm

from clean_app.src.CLEAN.infer import infer_maxsep, infer_pvalue
from clean_app.src.CLEAN.utils import (ensure_dirs, dump_info, get_ec_id_dict,
                                       mutate_incorrect_seq_ECs, retrieve_esm1b_embedding)

from train_loop import test_CLEAN_model, train_step_triplet, train_step_supconh, train_step_himulcone, validation_loop, save_end_of_training_metrics, reinit_CLEAN
from plots import plot_pca_by_uncertainty, plot_pca_by_class
from collections import defaultdict
from dataloader import reformat_emb, update_ec_id_dicts
from clean_app.src.CLEAN.distance_map import get_cluster_center
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler, Subset
from utils import save_metrics

def train_CLEAN_model_AL(model, criterion, optimizer, al_strat, train_datamodule, loss='triplet', eval_dataloader=None, test_data_list=[], num_epochs=100, batch_size=32, generate_plots=False, save_path='.', adaptive_rate=100, learning_rate=0.0001, checkpoint_and_eval=False, train_data_path='./', eval_data_path='./', train_filename='train', eval_filename='eval', save_recomputed_embeddings=False, maxsep=True, emb_dir='/emb_data/', cache_dir='/distance_map/', knn=30, shuffle=True, _format_esm=True, temp=0.1, n_pos=9, clip_norm=False, model_name='CLEAN', metrics_save_path='training_metrics.json'):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    dtype=torch.float32

    if maxsep:
        infer = infer_maxsep
    else:
        infer = infer_pvalue

    if loss == 'triplet':
        train_step = train_step_triplet
    elif loss == 'supconh':
        train_step = train_step_supconh
    elif loss == 'himulcone':
        train_step = train_step_himulcone

    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)

    epoch_losses = []
    val_losses = []
    val_aucs = []
    val_prcs = []
    val_accuracies = []
    val_balanced_accuracies = []
    val_precs = []
    val_recs = []
    val_f1s = []
    val_hps = []
    val_hrs = []
    val_hf1s = []

    if len(train_datamodule.unlabeled_indices) < batch_size:
        batch_size = max(len(train_datamodule.unlabeled_indices) / 3, 32) #use 1/3 the data per batch or 32 whichever is larger
        warnings.warn('Batch size decreased to {} due to data size.'.format(batch_size))

    min_val_loss = 100000000

    lowest_val_loss_model = None

    train_datamodule.random_init(n_samples=batch_size)
    
    for epoch in range(num_epochs):

        if epoch != 0:
            if len(train_datamodule.unlabeled_indices) < batch_size:
                train_datamodule.reset() #I think this is ok from a training context

            indices, scores = al_strat.query(model=model, al_datamodule=train_datamodule, acq_size=batch_size, return_utilities=True)
            train_datamodule.update_annotations(indices)

        #model.reset_states() # not sure why we should do this
        model.to(device)
        model.train()

        torch.cuda.empty_cache()
        epoch_loss = 0.0
        vtotal = 0
        vcorrect = 0
        num_batches = 0

        #train one epoch
        #____________________________________________________________________#
        for i, item in enumerate(train_datamodule.train_dataloader()):
            batch_loss = train_step(item, device, optimizer, model, criterion, clip_norm=clip_norm, temp=temp, n_pos=n_pos)
            epoch_loss += batch_loss

            num_batches += 1

        # store epoch loss and accuracy
        avg_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_epoch_loss:.4f}')
        #____________________________________________________________________#

        #validate one epoch
        #____________________________________________________________________#
        if checkpoint_and_eval and eval_dataloader!=None:
            avg_val_loss, min_val_loss, vpre, vrec, vf1, vauc, vprc, vacc, vbacc, vhp, vhr, vhf1, lowest_val_loss_model = validation_loop(model, 
                    eval_dataloader, 
                    min_val_loss, 
                    device, 
                    criterion, 
                    save_path, 
                    model_name, 
                    infer, 
                    train_filename, 
                    eval_filename, 
                    train_data_path, 
                    eval_data_path, 
                    emb_dir, 
                    reformat_emb(train_datamodule.emb, train_datamodule.ec_id_dict), 
                    _format_esm, 
                    loss, 
                    epoch,
                    active_learning_mode=True,
                    temp=temp,
                    n_pos=n_pos)

            val_losses.append(avg_val_loss)
            val_aucs.append(vauc)
            val_prcs.append(vprc)
            val_accuracies.append(vacc)
            val_balanced_accuracies.append(vbacc)
            val_precs.append(vpre)
            val_recs.append(vrec)
            val_f1s.append(vf1)
            val_hps.append(vhp)
            val_hrs.append(vhr)
            val_hf1s.append(vhf1)

            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.2f}, Validation Accuracy: {100*vacc:.2f}%')

        if len(train_datamodule.unlabeled_indices) < batch_size:
            train_datamodule.reset() #redo the unlabaled/labeled indices to complete number of epochs

        #____________________________________________________________________#

    #if save_recomputed_embeddings:
        #ensure_dirs(save_path+'/'+cache_dir+'/recomputed/')

        #FIXME - not a priority for simulation

    
    save_end_of_training_metrics(model, 
            save_path, 
            model_name, 
            epoch_losses, 
            val_losses, 
            val_aucs, 
            val_prcs, 
            val_accuracies, 
            val_balanced_accuracies, 
            val_precs, 
            val_recs, 
            val_f1s, 
            val_hps, 
            val_hrs, 
            val_hf1s, 
            generate_plots, 
            num_epochs, 
            checkpoint_and_eval, 
            eval_dataloader,
            metrics_save_path,
            active_learning_mode=True)

    if checkpoint_and_eval and eval_dataloader != None:
        return model, lowest_val_loss_model, (epoch_losses, val_losses, val_accuracies)
    else:
        return model, None, (epoch_losses)

_CENTROID_BATCH = 8192


def build_reference_set(train_datamodule, pool_datamodule, train_data_path,
                        train_filename, round_dir):
    """Reference set = train PLUS everything acquired so far.

    CLEAN predicts by nearest EC centroid, and those centroids are built from a
    reference set. By default that set is the train partition alone: the
    centroids MOVE every round (they are recomputed through the updated
    encoder), but their MEMBERSHIP never changes -- an acquired sequence never
    joins the set it is being predicted against.

    That is not what deployment looks like. Validating a sequence experimentally
    means adding it to your reference database, which gives a centroid nearer
    the shifted region immediately, with no encoder update at all.

    infer_maxsep rebuilds its own ec_id_dict by reading the reference CSV off
    disk, and get_cluster_center slices the embedding tensor in that dict's
    order -- so the CSV and the embedding tensor have to agree. The cleanest way
    to extend the set is therefore to write a combined CSV for the round and
    hand back a matching tensor.

    Returns (path, name, emb_tensor) to pass through to test_CLEAN_model.
    """
    full_list = pool_datamodule.query_dataset.full_list
    labeled = [full_list[i] for i in pool_datamodule.labeled_indices]

    src = os.path.join(train_data_path, train_filename + '.csv')
    rows, seen = [], set()
    with open(src) as fh:
        header = fh.readline()
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if parts and parts[0]:
                rows.append(parts[:3])
                seen.add(parts[0])

    emb_map = dict(train_datamodule.emb)
    for _id in labeled:
        if _id in seen:
            continue
        ecs = pool_datamodule.id_ec.get(_id)
        if not ecs:
            continue
        ec = ecs[0] if isinstance(ecs, (list, tuple)) else ecs
        if _id not in pool_datamodule.emb:
            continue
        rows.append([_id, ec, ''])
        emb_map[_id] = pool_datamodule.emb[_id]
        seen.add(_id)

    name = 'reference'
    out_csv = os.path.join(round_dir, name + '.csv')
    with open(out_csv, 'w') as fh:
        fh.write(header if header.strip() else 'Entry\tEC number\tSequence\n')
        for r in rows:
            fh.write('\t'.join((r + ['', '', ''])[:3]) + '\n')

    # Rebuild the dict exactly as infer_maxsep will, so the tensor order matches.
    _, ec_id_dict = get_ec_id_dict(out_csv)
    return round_dir, name, reformat_emb(emb_map, ec_id_dict)

def _replay_subset(train_datamodule, n_acquired, ratio, selection, seed):
    """Choose which training sequences to replay alongside the new ones.

    ratio is replayed-per-new. ratio=1.0 gives a 50:50 new:replay mix, 3.0 gives
    25:75, 0.33 gives 75:25. ratio=None replays the ENTIRE train partition,
    which is the original ft_integrated behaviour -- and at 144,364 train
    against a few hundred acquired that is roughly 451:1, so the new data is
    ~0.2% of each epoch and is effectively drowned out. The continual-learning
    literature works in the 50:50 to 25:75 band.

    selection:
      uniform_ec  spread the replay budget evenly over training ECs, so rare and
                  abundant functions are rehearsed alike. The natural default
                  here: uncertainty sampling already over-draws abundant ECs by
                  ~2.5-3.5x, so replaying proportionally would compound that
                  skew rather than counter it.
      random      uniform over sequences, i.e. proportional to EC abundance.
                  What the CL literature usually means by a replay buffer.
    """
    full = train_datamodule.train_dataset
    n_train = len(full)
    if ratio is None:
        return full

    n_replay = min(n_train, max(1, int(round(ratio * max(1, n_acquired)))))
    rng = random.Random(seed)

    if selection == 'random':
        idx = rng.sample(range(n_train), n_replay)
        return Subset(full, idx)

    # uniform_ec: round-robin over ECs, taking one member at a time, so the
    # budget spreads across as many distinct functions as it can reach.
    ec_id_dict = getattr(train_datamodule, 'ec_id_dict', None)
    if not ec_id_dict:
        idx = rng.sample(range(n_train), n_replay)
        return Subset(full, idx)

    order = getattr(train_datamodule.train_dataset, 'full_list', None)
    pos = {}
    if order is not None:
        # full_list holds ECs; positions must come from the dataset's own order
        cursor = 0
        for ec in ec_id_dict:
            for _id in ec_id_dict[ec]:
                pos.setdefault(ec, []).append(cursor)
                cursor += 1
    if not pos:
        idx = rng.sample(range(n_train), n_replay)
        return Subset(full, idx)

    ecs = list(pos.keys())
    rng.shuffle(ecs)
    for ec in ecs:
        rng.shuffle(pos[ec])
    picked, ring = [], 0
    while len(picked) < n_replay and ecs:
        ec = ecs[ring % len(ecs)]
        if pos[ec]:
            picked.append(pos[ec].pop())
        else:
            ecs.remove(ec)
            continue
        ring += 1
    return Subset(full, picked[:n_replay])



def seed_target(pool_datamodule, target_ec, n_seed, rng_seed=1234):
    """Pre-label a few known members of the hunted EC before round 0.

    WHY THIS IS REQUIRED, not a convenience. CLEAN predicts by nearest EC
    centroid, and build_ec_centroids only creates a centroid for an EC that has
    at least one LABELLED member. The target was deliberately removed from
    pretraining, so at round 0 it has no centroid, no column in the projection,
    and therefore cannot be the argmax for any pool sequence. The predicted-EC
    filter returns an empty shortlist every round, the loop silently falls back
    to generic acquisition, and the experiment measures nothing it intended.
    Measured over four runs before this was added: shortlist 0 in all 40 rounds,
    and positives found at chance (3-8 per 320 assays against 3.2 expected).

    The fix is also the more faithful scenario. Someone hunting a function has at
    least one known example -- that is how they know it exists. Seeding gives the
    model a reference point for the target while leaving the encoder ignorant of
    it, which is exactly the position of a scientist who has just characterised
    one enzyme and wants more.

    Seeds are marked acquired and recorded as positive assays, so they build the
    centroid and can anchor negative-assay triplets. They are logged separately
    so they are never counted as discoveries.
    """
    if not target_ec or n_seed <= 0:
        return []
    full = pool_datamodule.query_dataset.full_list
    cand = []
    for idx in list(pool_datamodule.unlabeled_indices):
        ecs = pool_datamodule.id_ec.get(full[idx], [])
        if target_ec in (ecs if isinstance(ecs, (list, tuple)) else [ecs]):
            cand.append(idx)
    if not cand:
        print(f'[seed] WARNING: no {target_ec} members in the pool to seed from')
        return []
    rnd = random.Random(rng_seed)
    chosen = rnd.sample(cand, min(n_seed, len(cand)))
    pool_datamodule.update_annotations(chosen)
    print(f'[seed] pre-labelled {len(chosen)} known members of {target_ec} '
          f'({len(cand)} available); these are NOT counted as discoveries')
    return chosen


def target_candidates(model, pool_datamodule, target_ec, device, min_depth=2):
    """Pool indices the model predicts to be the target EC, with hierarchical backoff.

    This is what makes the target experiment match its use case. A scientist
    hunting one function does not send the globally most uncertain sequences to
    the bench; they send sequences their model thinks ARE the target, and the
    assay returns yes or no. Acquisition then ranks within that shortlist.

    BACKOFF BY EC PREFIX. Exact matches on the full EC are preferred, but when
    none exist the shortlist widens to the same sub-subclass (2.7.7.- for a
    2.7.7.6 target) and then the same subclass (2.7.-.-). Enzymes sharing three
    EC levels catalyse closely related chemistry, so that is where the target's
    undiscovered homologs actually sit -- a far better prior than the topping-up
    from the whole pool this replaces, which silently turned the run into generic
    acquisition. Backoff stops at `min_depth` rather than continuing to the top
    level, because sharing only the first EC digit ("it is a transferase") says
    almost nothing.

    Prediction is argmax over the same negated squared distances to EC centroids
    that the acquisition functions score, so filter and ranking agree by
    construction.

    Returns (candidate_indices, n_candidates, depth_used). depth 4 is an exact
    match, 3 is the sub-subclass, 2 the subclass; None means nothing at or above
    min_depth, which with seeding in place indicates a real problem rather than
    a cold start.
    """
    ec_order = getattr(model, '_ec_order', None) or []
    if not ec_order or not pool_datamodule.unlabeled_indices:
        return [], 0, None

    # Use the datamodule's own unlabeled_dataloader and the model's get_logits --
    # exactly what UncertaintySampling.query does. A hand-rolled DataLoader over
    # query_dataset was unpacking batches wrong and produced a degenerate forward
    # pass: all 32,309 pool sequences predicted into class 1.1. Sharing the
    # acquisition path guarantees the filter and the ranking see the same tensor.
    loader, unl = pool_datamodule.unlabeled_dataloader()
    with torch.no_grad():
        logits = model.get_logits(loader)
    preds = logits.argmax(1).cpu().numpy()
    pred_ec = [ec_order[i] if i < len(ec_order) else '' for i in preds]
    from collections import Counter as _C
    _pref = _C(".".join(e.split(".")[:2]) for e in pred_ec if e).most_common(3)
    print(f'[target-dbg] ec_order={len(ec_order)} target_in_order={target_ec in ec_order} '
          f'n_unlabeled={len(unl)} top-prefixes={_pref}')

    tparts = target_ec.split('.')
    for depth in range(4, min_depth - 1, -1):
        want = '.'.join(tparts[:depth])
        cand = [idx for idx, pe in zip(unl, pred_ec)
                if pe and '.'.join(pe.split('.')[:depth]) == want]
        if cand:
            return cand, len(cand), depth
    return [], 0, None


def oracle_results(pool_datamodule, acquired_idx, target_ec):
    """Wet-lab oracle: for each acquired sequence, does it have the target function?

    Returns (ids, ecs_as_predicted, results). `ecs` is the TARGET label, not the
    true one, because that is what the experimenter was testing for -- the assay
    answers 'does this do X', not 'what does this do'. On a False the update path
    uses that label to push the sequence away from the target EC, which is
    exactly the information a negative assay carries.
    """
    full = pool_datamodule.query_dataset.full_list
    ids = [full[i] for i in acquired_idx]
    res = []
    for _id in ids:
        true = pool_datamodule.id_ec.get(_id, [])
        if not isinstance(true, (list, tuple)):
            true = [true]
        res.append(target_ec in true)
    return ids, [target_ec] * len(ids), res


def update_dataloader(pool_datamodule, train_datamodule, regime, newly_acquired,
                      replay_ratio=None, replay_selection='uniform_ec', seed=1234):
    """Dataloader for one round's update, per --update_regime.

    Mirrors dal_toolbox's train_dataloader (Subset + RandomSampler + collator)
    so batching and sampling stay identical across regimes; only the anchor set
    differs.

      scratch        cumulative acquired pool -- the published HATTER setup,
                     where the pool IS the training data and the model is built
                     from nothing.
      ft_new         ONLY the sequences acquired this round. Fine-tuning a
                     pretrained model on new labels alone, with no rehearsal of
                     anything previously seen.
      ft_integrated  the original train partition PLUS everything acquired.
                     Rehearsal, a standard domain-adaptation remedy for the
                     forgetting that ft_new invites.

    ft_new vs ft_integrated is the comparison the paper turns on.
    """
    dm = pool_datamodule
    if regime == 'ft_new':
        if not newly_acquired:
            return []
        dataset = Subset(dm.train_dataset, indices=list(newly_acquired))
    elif regime == 'ft_integrated':
        acquired = Subset(dm.train_dataset, indices=list(dm.labeled_indices))
        replay = _replay_subset(train_datamodule, len(acquired), replay_ratio,
                                replay_selection, seed)
        dataset = ConcatDataset([replay, acquired])
    else:
        return dm.train_dataloader()

    sampler = RandomSampler(dataset, num_samples=None)
    return DataLoader(dataset, batch_size=dm.train_batch_size,
                      sampler=sampler, collate_fn=dm.collator)

def apply_labeled_scope(pool_datamodule, train_datamodule):
    """Restrict the pool dataset's contrastive mining to train + acquired.

    No-op unless --mining_scope labeled is in effect (which sets
    _use_labeled_mining on the datamodule). Must be re-applied after every
    random_init / update_annotations, since the labelled set grows each round.
    """
    if not getattr(pool_datamodule, '_use_labeled_mining', False):
        return
    ds = getattr(pool_datamodule, 'train_dataset', None)
    if ds is None or not hasattr(ds, 'set_labeled_scope'):
        return
    full_list = pool_datamodule.query_dataset.full_list
    labeled_ids = [full_list[i] for i in pool_datamodule.labeled_indices]
    ds.set_labeled_scope(labeled_ids,
                         train_id_ec=getattr(train_datamodule, 'id_ec', None),
                         train_ec_id=getattr(train_datamodule, 'ec_id', None))

def build_ec_centroids(model, train_datamodule, pool_datamodule, device):
    """EC cluster centres over everything currently LABELLED.

    CLEAN predicts an EC by finding the nearest EC cluster centre, where a
    centre is the mean projected embedding of the labelled sequences carrying
    that EC. In a simulation the labelled set grows every round, so the centres
    must be rebuilt from train PLUS whatever has been acquired so far --
    otherwise a rare EC that started with two examples keeps a centre computed
    from those two even after AL has labelled five more of it, which is exactly
    the regime the rare-EC analysis cares about.

    Acquired ECs that were absent from the initial train set get a centre here
    once they are labelled, matching the real workflow: labelling a novel EC is
    what makes it predictable. The number of EC columns therefore grows across
    rounds. That is fine for acquisition, which scores per row.

    Only pool items in `labeled_indices` contribute. The rest of the pool has
    labels available (it is a simulation) but using them would be leakage.
    """
    # --- assemble the labelled set, reusing the cached stack where possible ----
    #
    # The raw ESM embeddings never change; only the projection does, because the
    # encoder is retrained each round. So the expensive parts -- gathering
    # thousands of per-sequence tensors, stacking them, and moving them to the
    # GPU -- are done ONCE and cached on the model. Each later round appends
    # only the newly acquired rows. This is what made the naive version slow
    # enough to skip: it rebuilt a (n_labelled, 1280) tensor from a Python dict
    # and re-copied it to the device on every acquisition.
    cache = getattr(model, '_centroid_cache', None)
    full_list = pool_datamodule.query_dataset.full_list
    labelled = list(pool_datamodule.labeled_indices)

    def _ecs_for(s_id, datamodule):
        ecs = datamodule.id_ec.get(s_id, [])
        return [ecs] if isinstance(ecs, str) else list(ecs)

    if cache is None:
        rows, row_ecs = [], []
        for ec, ids in train_datamodule.ec_id_dict.items():
            for s_id in ids:
                rows.append(train_datamodule.emb[s_id])
                row_ecs.append(ec)
        seen_pool = set()
        for idx in labelled:
            s_id = full_list[idx]
            for ec in _ecs_for(s_id, pool_datamodule):
                rows.append(pool_datamodule.emb[s_id])
                row_ecs.append(ec)
            seen_pool.add(idx)
        cache = {'raw': torch.stack(rows, dim=0).to(device),
                 'row_ecs': row_ecs,
                 'seen_pool': seen_pool}
        model._centroid_cache = cache
    else:
        new_rows, new_ecs = [], []
        for idx in labelled:
            if idx in cache['seen_pool']:
                continue
            s_id = full_list[idx]
            for ec in _ecs_for(s_id, pool_datamodule):
                new_rows.append(pool_datamodule.emb[s_id])
                new_ecs.append(ec)
            cache['seen_pool'].add(idx)
        if new_rows:
            cache['raw'] = torch.cat(
                [cache['raw'], torch.stack(new_rows, dim=0).to(device)], dim=0)
            cache['row_ecs'].extend(new_ecs)

    # --- project once, then mean-pool per EC with a scatter -------------------
    #
    # Batched so a large labelled set cannot OOM, and the per-EC mean is a
    # single index_add rather than a Python loop over ECs.
    raw, row_ecs = cache['raw'], cache['row_ecs']
    ecs = sorted(set(row_ecs))
    # column order for the projection, so an argmax can be mapped back to an EC
    model._ec_order = ecs
    ec_pos = {ec: i for i, ec in enumerate(ecs)}
    idx_t = torch.tensor([ec_pos[e] for e in row_ecs], device=device, dtype=torch.long)

    chunks = []
    with torch.no_grad():
        for start in range(0, raw.shape[0], _CENTROID_BATCH):
            chunks.append(model.model(raw[start:start + _CENTROID_BATCH]))
    projected = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]

    sums = torch.zeros(len(ecs), projected.shape[1],
                       device=device, dtype=projected.dtype)
    sums.index_add_(0, idx_t, projected)
    counts = torch.zeros(len(ecs), device=device, dtype=projected.dtype)
    counts.index_add_(0, idx_t, torch.ones_like(idx_t, dtype=projected.dtype))

    return sums / counts.unsqueeze(1).clamp_min(1)

RANDOM_SEED_FALLBACK = 1234


def run_CLEAN_active_learning_simulation(model, criterion, optimizer, al_strat, train_datamodule, pool_datamodule, eval_dataloader=None, n_instances=32, n_queries=3, generate_plots=False, save_path='.', adaptive_rate=100, learning_rate=0.0001, checkpoint_and_eval=False, train_data_path='./', eval_data_path='./', pool_data_path='./', train_filename='train', eval_filename='eval', pool_filename='./', pca=None, label_encoder=None, plot_tuple=None, save_recomputed_embeddings=False, maxsep=True, loss='triplet', model_name='CLEAN', metrics_save_path='training_metrics.json', emb_dir='/emb_data/', cache_dir='/distance_map/', clip_norm=False, temp=0.1, n_pos=9, _format_esm=False, test_data_list=[], update_regime='scratch', reference_set='train', replay_ratio=None, replay_selection='uniform_ec', eval_every=1, target_ec=None, n_seed_target=0):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    dtype=torch.float32

    if maxsep:
        infer = infer_maxsep
    else:
        infer = infer_pvalue

    if loss == 'triplet':
        train_step = train_step_triplet
    elif loss == 'supconh':
        train_step = train_step_supconh
    elif loss == 'himulcone':
        train_step = train_step_himulcone

    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)

    epoch_losses = []
    val_losses = []
    val_aucs = []
    val_prcs = []
    val_accuracies = []
    val_balanced_accuracies = []
    val_precs = []
    val_recs = []
    val_f1s = []
    val_hps = []
    val_hrs = []
    val_hf1s = []

    min_val_loss = 100000000

    lowest_val_loss_model = None

    if plot_tuple is not None and generate_plots:
        pool_ids_full, pool_embeddings_full, pool_ecs_full = plot_tuple

    pool_datamodule.random_init(n_samples=n_instances)
    indices = pool_datamodule.labeled_indices
    apply_labeled_scope(pool_datamodule, train_datamodule)
    newly_acquired = list(pool_datamodule.labeled_indices)

    break_early = False
    exit_round = False
    seeded_ids = []
    match_tally = {'exact': 0, 'sub-subclass': 0, 'subclass': 0, 'NONE': 0}
    if target_ec is not None and n_seed_target > 0:
        _sel = seed_target(pool_datamodule, target_ec, n_seed_target, RANDOM_SEED_FALLBACK)
        seeded_ids = [pool_datamodule.query_dataset.full_list[j] for j in _sel]
        apply_labeled_scope(pool_datamodule, train_datamodule)
        _sd = pool_datamodule.train_dataset
        _sd = _sd.dataset if hasattr(_sd, 'dataset') else _sd
        if hasattr(_sd, 'ids_for_update') and seeded_ids:
            _sd.target_ec = target_ec
            _sd.known_target_ids = list(seeded_ids)

    for i_cycle in range(n_queries+1):
        if break_early:
            exit_round = True

        if not os.path.exists(save_path+'/round_{}'.format(i_cycle)):
            os.makedirs(save_path+'/round_{}'.format(i_cycle))

        if i_cycle != 0:
            if generate_plots:
                seq_ids_before_query = [pool_datamodule.query_dataset.full_list[i] for i in pool_datamodule.unlabeled_indices] #need for plotting

            # Refresh EC cluster centres from the CURRENT model before scoring,
            # so acquisition sees a posterior over ECs rather than a softmax
            # over embedding dimensions. No-op unless --acquisition_space
            # distance was passed. Recomputed every round because the encoder
            # is retrained each cycle and the centres move with it.
            if getattr(model, '_use_distance_logits', False):
                model.set_ec_centroids(None)   # embed with the raw encoder
                model.set_ec_centroids(
                    build_ec_centroids(model, train_datamodule, pool_datamodule, device))

            _restore = None
            n_cand = None
            if target_ec is not None:
                cand, n_cand, _depth = target_candidates(model, pool_datamodule, target_ec, device)
                _lvl = {4: 'exact', 3: 'sub-subclass', 2: 'subclass'}.get(_depth, 'NONE')
                if cand:
                    # Rank within the shortlist even when it is smaller than the
                    # batch: dal_toolbox returns everything available rather than
                    # failing, so a short list simply means a short round. That
                    # is the honest behaviour -- padding from the wider pool is
                    # what silently turned this into generic acquisition before.
                    _restore = list(pool_datamodule.unlabeled_indices)
                    pool_datamodule.unlabeled_indices = cand
                    if len(cand) < n_instances:
                        print(f'[target] shortlist {len(cand)} < batch {n_instances} '
                              f'({_lvl}); acquiring the whole shortlist')
                else:
                    print(f'[target] NO candidates at or above the sub-subclass level for '
                          f'{target_ec}. With seeding this should not happen; the round '
                          f'falls back to generic acquisition and is flagged in oracle.json')
                match_tally[_lvl] = match_tally.get(_lvl, 0) + 1
                print(f'[target] round {i_cycle}: shortlist={n_cand} match={_lvl}'
                      f'  [tally exact={match_tally["exact"]} '
                      f'sub-subclass={match_tally["sub-subclass"]} '
                      f'subclass={match_tally["subclass"]} none={match_tally["NONE"]}]')
            indices, scores = al_strat.query(model=model, al_datamodule=pool_datamodule, acq_size=n_instances, return_utilities=True)
            if _restore is not None:
                pool_datamodule.unlabeled_indices = _restore
            newly_acquired = list(indices)
            scores = scores.cpu()
            if target_ec is not None:
                _ids, _ecs, _res = oracle_results(pool_datamodule, newly_acquired, target_ec)
                n_yes = sum(1 for r in _res if r)
                print(f'[target] round {i_cycle}: {n_yes}/{len(_res)} assayed POSITIVE for {target_ec}')
                save_metrics({'round': i_cycle, 'n_predicted': n_cand, 'ids': _ids,
                              'results': [bool(r) for r in _res], 'n_yes': n_yes,
                              'n_seeded': len(seeded_ids), 'match_depth': _depth,
                              'match_level': _lvl, 'degraded': bool(not cand)},
                             save_path + '/round_{}/oracle.json'.format(i_cycle))
                _ds = pool_datamodule.train_dataset
                _ds = _ds.dataset if hasattr(_ds, 'dataset') else _ds
                if hasattr(_ds, 'ids_for_update'):
                    _prev_i = list(getattr(_ds, 'ids_for_update', None) or [])
                    _prev_e = list(getattr(_ds, 'ecs_for_update', None) or [])
                    _prev_r = list(getattr(_ds, 'result_of_experiment', None) or [])
                    _ds.ids_for_update = _prev_i + _ids
                    _ds.ecs_for_update = _prev_e + _ecs
                    _ds.result_of_experiment = _prev_r + [bool(r) for r in _res]
                    # Confirmed members of the hunted EC. A negative assay
                    # anchors on these and pushes the rejected sequence away, so
                    # the update sharpens the boundary around the target rather
                    # than nudging one false positive. Empty until the first
                    # positive assay, which is also when a target centroid first
                    # exists and the predicted-EC filter starts working.
                    _ds.target_ec = target_ec
                    _ds.known_target_ids = list(seeded_ids) + [
                        i for i, r in zip(_ds.ids_for_update, _ds.result_of_experiment) if r]
                    print(f'[target] confirmed members of {target_ec} so far: '
                          f'{len(_ds.known_target_ids)}')
                    _neg_ids = [i for i, r in zip(_ids, _res) if not r]
                    if _neg_ids:
                        mutate_incorrect_seq_ECs(_neg_ids, [False] * len(_neg_ids),
                                                 name=pool_filename, path=pool_data_path,
                                                 emb_out_dir=emb_dir)
                        retrieve_esm1b_embedding(pool_filename + '_incorrect_seq_ECs',
                                                 path=pool_data_path, emb_out_dir=emb_dir)

            print(indices)
            print(scores)

            pool_datamodule.update_annotations(indices)
            apply_labeled_scope(pool_datamodule, train_datamodule)

        #model.reset_states()
        model.to(device)
        model.train()

        torch.cuda.empty_cache()
        epoch_loss = 0.0
        vtotal = 0
        vcorrect = 0
        num_batches = 0

        #train one epoch
        #____________________________________________________________________#
        for i, item in enumerate(update_dataloader(pool_datamodule, train_datamodule, update_regime, newly_acquired,
                                                  replay_ratio, replay_selection, RANDOM_SEED_FALLBACK)):
            batch_loss = train_step(item, device, optimizer, model, criterion, clip_norm=clip_norm, temp=temp, n_pos=n_pos)
            epoch_loss += batch_loss

            num_batches += 1

        # store epoch loss and accuracy
        avg_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss)

        print(f'Round [{i_cycle}/{n_queries}], Training Loss: {avg_epoch_loss:.4f}')
        #____________________________________________________________________#

        #not enough instances to do requested amount so reset the indices
        if len(pool_datamodule.unlabeled_indices) < n_instances:
            #pool_datamodule.reset() #dont think reset is the right functionality from experimental context
            n_instances = len(pool_datamodule.unlabeled_indices) #sample whatever is left
            break_early=True
            warnings.warn('Only Completing {} number of rounds due to runnning out of pool data points.'.format(i_cycle+1))

        #validate one epoch
        #____________________________________________________________________#
        if checkpoint_and_eval and eval_dataloader!=None:
            avg_val_loss, min_val_loss, vpre, vrec, vf1, vauc, vprc, vacc, vbacc, vhp, vhr, vhf1, lowest_val_loss_model = validation_loop(model, 
                    eval_dataloader, 
                    min_val_loss, 
                    device, 
                    criterion, 
                    save_path+'/round_{}/'.format(i_cycle), 
                    model_name, 
                    infer, 
                    train_filename, 
                    eval_filename, 
                    train_data_path, 
                    eval_data_path, 
                    emb_dir, 
                    reformat_emb(train_datamodule.emb, train_datamodule.ec_id_dict), 
                    _format_esm, 
                    loss, 
                    i_cycle,
                    active_learning_mode=True,
                    temp=temp,
                    n_pos=n_pos)

            val_losses.append(avg_val_loss)
            val_aucs.append(vauc)
            val_prcs.append(vprc)
            val_accuracies.append(vacc)
            val_balanced_accuracies.append(vbacc)
            val_precs.append(vpre)
            val_recs.append(vrec)
            val_f1s.append(vf1)
            val_hps.append(vhp)
            val_hrs.append(vhr)
            val_hf1s.append(vhf1)

            print(f'Round [{i_cycle}/{n_queries}], Validation Loss: {avg_val_loss:.2f}, Validation Accuracy: {100*vacc:.2f}%')

        #____________________________________________________________________#

        seq_ids = [pool_datamodule.query_dataset.full_list[i] for i in indices]
        ec_ids = [pool_datamodule.id_ec[_id] for _id in seq_ids]

        assert(len(ec_ids) == len(seq_ids))

        save_metrics(seq_ids, save_path+'/round_{}/gene_ids.txt'.format(i_cycle))
        save_metrics(ec_ids, save_path+'/round_{}/ec_ids.txt'.format(i_cycle))
        
        if generate_plots and i_cycle != 0: #can only do after scores are calculated
            current_pool = [list(pool_ids_full).index(_id) for _id in seq_ids_before_query]
            pool_embeddings_current = pool_embeddings_full[current_pool]
            pool_ids_current = pool_ids_full[current_pool]
            pool_ecs_current = pool_ecs_full[current_pool]

            to_plot = [list(pool_ids_current).index(_id) for _id in seq_ids]
            scores_dict = dict(zip(seq_ids_before_query, scores))
            mapped_id_scores = [scores_dict[id_index] for id_index in pool_ids_current]

            plot_pca_by_uncertainty(pca, 
                    pool_embeddings_current, 
                    pool_embeddings_current[to_plot], 
                    mapped_id_scores, 
                    instance=i_cycle, 
                    path=save_path+'/round_{}/'.format(i_cycle))

            plot_pca_by_class(pca, 
                    label_encoder, 
                    pool_embeddings_current, 
                    pool_ecs_current, 
                    pool_embeddings_current[to_plot], 
                    np.array(pool_ecs_current)[to_plot], 
                    instance=i_cycle, 
                    path=save_path+'/round_{}'.format(i_cycle))

        #update data lists to add annotations to train data
        #FIXME - likely bugs; not needed for simulation; not currently supported
        #train_datamodule, pool_datamodule = update_ec_id_dicts(train_datamodule, 
        #        pool_datamodule, 
        #        ec_ids, 
        #        seq_ids, 
        #        train_path=train_data_path, 
        #        pool_path=pool_data_path, 
        #        train_name=train_filename, 
        #        pool_name=pool_filename)

        #FIXME - likely bugs; not needed for simulation; not currently supported
        #if generate_plots or save_recomputed_embeddings:
            #update points for next round
        #    pool_ids_full = []
        #    pool_ecs_full = []
        #    for ec in list(pool_datamodule.ec_id_dict.keys()):
        #        ids_for_query = list(pool_datamodule.ec_id_dict[ec])
        #        pool_ids_full.extend(ids_for_query)
        #        pool_ecs_full.extend([ec]*len(ids_for_query))

        #    pool_embeddings_full = reformat_emb(pool_datamodule.emb, pool_datamodule.ec_id_dict).cpu()

        #FIXME - likely bugs; not needed for simulation; not currently supported
        #if save_recomputed_embeddings:
        #    ensure_dirs(save_path+'/round_{}/{}/recomputed/'.format(cache_dir, i_cycle))

        #    train_ids = []
        #    train_ecs = []
        #    for ec in list(train_datamodule.ec_id_dict.keys()):
        #        ids_for_query = list(train_datamodule.ec_id_dict[ec])
        #        train_ids.extend(ids_for_query)
        #        train_ecs.extend([ec]*len(ids_for_query))

        #    dump_info(train_ids, 
        #            train_ecs, 
        #            train_datamodule.dist_map, 
        #            reformat_emb(train_datamodule.emb, train_datamodule.ec_id_dict), 
        #            path=save_path, 
        #            extra_folder='/round_{}/{}/recomputed/'.format(cache_dir, i_cycle), 
        #            train_file=train_filename)
            
        #    dump_info(pool_ids_full, 
        #            pool_ecs_full, 
        #            pool_datamodule.dist_map, 
        #            pool_embeddings_full.to(device), 
        #            path=save_path, 
        #            extra_folder='/round_{}/{}/recomputed/'.format(cache_dir, i_cycle), 
        #            train_file=pool_filename)

        # Evaluation dominates round cost: it infers every test sequence and,
        # under train_plus_acquired, rebuilds the reference set too. Skipping it
        # on most rounds is what makes a full-pool sweep affordable. The LAST
        # round always evaluates, so the endpoint is never lost to the stride.
        _do_eval = (eval_every <= 1) or (i_cycle % eval_every == 0) or (i_cycle == n_queries)
        for (test_data_name, test_data_path, test_data) in (test_data_list if _do_eval else []):
            _ref_path, _ref_name = train_data_path, train_filename
            _ref_emb = reformat_emb(train_datamodule.emb, train_datamodule.ec_id_dict)
            if reference_set == 'train_plus_acquired':
                _ref_path, _ref_name, _ref_emb = build_reference_set(
                    train_datamodule, pool_datamodule, train_data_path,
                    train_filename, save_path + '/round_{}'.format(i_cycle))

            test_CLEAN_model(model=model, 
                         train_data_path=_ref_path, 
                         test_data_path=test_data_path, 
                         device=device, 
                         train_name=_ref_name, 
                         test_name=test_data_name, 
                         checkpoint_dir=save_path+'/round_{}'.format(i_cycle), 
                         metrics_save_path=test_data_name+'_metrics.json',
                         train_emb=_ref_emb,
                         emb_out_dir=emb_dir,
                         _format_esm=_format_esm,
                         maxsep=maxsep,
                         model_name=model_name)

        #exiting due to lack of points to complete round
        if exit_round:
            break
    if target_ec is not None:
        _tot = sum(match_tally.values()) or 1
        print(f'[target] SHORTLIST TIER SUMMARY for {target_ec} over {_tot} rounds:')
        for _k in ('exact', 'sub-subclass', 'subclass', 'NONE'):
            print(f'[target]   {_k:14s} {match_tally[_k]:4d} rounds  ({100*match_tally[_k]/_tot:5.1f}%)')
        if match_tally['exact'] == 0:
            print('[target]   WARNING: the model NEVER predicted the target exactly. '
                  'Treat these results as a homolog-neighbourhood search, not target recovery.')
        save_metrics({'target_ec': target_ec, 'rounds': _tot, 'n_seeded': len(seeded_ids),
                      'tier_rounds': match_tally,
                      'frac_exact': match_tally['exact'] / _tot,
                      'frac_degraded': match_tally['NONE'] / _tot},
                     save_path + '/target_summary.json')


    save_end_of_training_metrics(model, 
            save_path, 
            model_name, 
            epoch_losses, 
            val_losses, 
            val_aucs, 
            val_prcs, 
            val_accuracies, 
            val_balanced_accuracies, 
            val_precs, 
            val_recs, 
            val_f1s, 
            val_hps, 
            val_hrs, 
            val_hf1s, 
            generate_plots, 
            n_queries, 
            checkpoint_and_eval, 
            eval_dataloader,
            metrics_save_path,
            active_learning_mode=True)

    if checkpoint_and_eval and eval_dataloader != None:
        return model, lowest_val_loss_model, (epoch_losses, val_losses, val_accuracies)
    else:
        return model, None, (epoch_losses)


############################## ORIGINAL - NO CLEAN ###############################################

#rewrote some functions for standard nn
from train_loop import test_model, train_step_standard, validation_loop_standard, save_end_of_training_metrics_standard

def train_standard_model_AL(model, criterion, optimizer, al_strat, train_datamodule, eval_dataloader=None, test_data_list=[], num_epochs=100, batch_size=32, generate_plots=False, save_path='.', learning_rate=0.0001, checkpoint_and_eval=False, train_data_path='./', eval_data_path='./', train_filename='train', eval_filename='eval', emb_dir='/emb_data/', cache_dir='/distance_map/', knn=30, shuffle=True, _format_esm=True, clip_norm=False, model_name='standard', metrics_save_path='training_metrics.json'):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    dtype=torch.float32
    train_step = train_step_standard
    le = train_datamodule.train_dataset.label_encoder

    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)

    epoch_losses = []
    val_losses = []
    val_accuracies = []
    val_precs = []
    val_recs = []
    val_f1s = []
    val_hps = []
    val_hrs = []
    val_hf1s = []
    min_val_loss = 100000000
    lowest_val_loss_model = None

    if len(train_datamodule.unlabeled_indices) < batch_size:
        batch_size = max(len(train_datamodule.unlabeled_indices) / 3, 32) #use 1/3 the data per batch or 32 whichever is larger
        warnings.warn('Batch size decreased to {} due to data size.'.format(batch_size))

    train_datamodule.random_init(n_samples=batch_size)
    
    for epoch in range(num_epochs):

        if epoch != 0:
            if len(train_datamodule.unlabeled_indices) < batch_size:
                train_datamodule.reset() #I think this is ok from a training context

            indices, scores = al_strat.query(model=model, al_datamodule=train_datamodule, acq_size=batch_size, return_utilities=True)
            train_datamodule.update_annotations(indices)

        #model.reset_states() # not sure why we should do this
        model.to(device)
        model.train()

        torch.cuda.empty_cache()
        epoch_loss = 0.0
        vtotal = 0
        vcorrect = 0
        num_batches = 0

        #train one epoch
        #____________________________________________________________________#
        for i, item in enumerate(tqdm(train_datamodule.train_dataloader(), desc='train loop', leave=False)):
            batch_loss = train_step(item, device, optimizer, model, criterion, clip_norm=clip_norm)
            epoch_loss += batch_loss

            num_batches += 1

        # store epoch loss and accuracy
        avg_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_epoch_loss:.4f}')
        #____________________________________________________________________#

        #validate one epoch
        #____________________________________________________________________#
        if checkpoint_and_eval and eval_dataloader!=None:
            avg_val_loss, min_val_loss, vacc, vprec, vrec, vf1, vhp, vhr, vhf1, lowest_val_loss_model = validation_loop_standard(model, 
                    eval_dataloader, 
                    min_val_loss, 
                    device, 
                    criterion, 
                    save_path, 
                    model_name, 
                    le, 
                    train_filename, 
                    eval_filename, 
                    train_data_path, 
                    eval_data_path, 
                    emb_dir, 
                    _format_esm, 
                    epoch,
                    active_learning_mode=True)

            val_losses.append(avg_val_loss)
            val_accuracies.append(vacc)
            val_precs.append(vprec)
            val_recs.append(vrec)
            val_f1s.append(vf1)
            val_hps.append(vhp)
            val_hrs.append(vhr)
            val_hf1s.append(vhf1)

            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.2f}, Validation Accuracy: {100*vacc:.2f}%')

        if len(train_datamodule.unlabeled_indices) < batch_size:
            train_datamodule.reset() #redo the unlabaled/labeled indices to complete number of epochs

        #____________________________________________________________________#

    save_end_of_training_metrics_standard(model, 
            save_path, 
            model_name, 
            epoch_losses, 
            val_losses, 
            val_accuracies, 
            val_precs,
            val_recs,
            val_f1s,
            val_hps, 
            val_hrs, 
            val_hf1s, 
            generate_plots, 
            num_epochs, 
            checkpoint_and_eval, 
            eval_dataloader,
            metrics_save_path,
            active_learning_mode=True)

    if checkpoint_and_eval and eval_dataloader != None:
        return model, lowest_val_loss_model, (epoch_losses, val_losses, val_accuracies)
    else:
        return model, None, (epoch_losses)

def run_standard_active_learning_simulation(model, criterion, optimizer, al_strat, train_datamodule, pool_datamodule, eval_dataloader=None, n_instances=32, n_queries=3, generate_plots=False, save_path='.', learning_rate=0.0001, checkpoint_and_eval=False, train_data_path='./', eval_data_path='./', pool_data_path='./', train_filename='train', eval_filename='eval', pool_filename='./', pca=None, plot_tuple=None, model_name='standard', metrics_save_path='training_metrics.json', emb_dir='/emb_data/', cache_dir='/distance_map/', clip_norm=False, _format_esm=False, test_data_list=[], update_regime='scratch', reference_set='train', replay_ratio=None, replay_selection='uniform_ec'):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    dtype=torch.float32

    train_step = train_step_standard
    le = train_datamodule.train_dataset.label_encoder

    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)

    epoch_losses = []
    val_losses = []
    val_accuracies = []
    val_precs = []
    val_recs = []
    val_f1s = []
    val_hps = []
    val_hrs = []
    val_hf1s = []
    min_val_loss = 100000000
    lowest_val_loss_model = None

    if plot_tuple is not None and generate_plots:
        pool_ids_full, pool_embeddings_full, pool_ecs_full = plot_tuple

    pool_datamodule.random_init(n_samples=n_instances)
    indices = pool_datamodule.labeled_indices

    break_early = False
    exit_round = False
    for i_cycle in range(n_queries+1):
        if break_early:
            exit_round = True

        if not os.path.exists(save_path+'/round_{}'.format(i_cycle)):
            os.makedirs(save_path+'/round_{}'.format(i_cycle))

        if i_cycle != 0:
            if generate_plots:
                seq_ids_before_query = [pool_datamodule.query_dataset.dataset.full_list_ids[i] for i in pool_datamodule.unlabeled_indices] #need for plotting

            indices, scores = al_strat.query(model=model, al_datamodule=pool_datamodule, acq_size=n_instances, return_utilities=True)
            scores = scores.cpu()

            pool_datamodule.update_annotations(indices)

        #model.reset_states()
        model.to(device)
        model.train()

        torch.cuda.empty_cache()
        epoch_loss = 0.0
        vtotal = 0
        vcorrect = 0
        num_batches = 0

        #train one epoch
        #____________________________________________________________________#
        for i, item in enumerate(pool_datamodule.train_dataloader()):
            batch_loss = train_step(item, device, optimizer, model, criterion, clip_norm=clip_norm)
            epoch_loss += batch_loss

            num_batches += 1

        # store epoch loss and accuracy
        avg_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss)

        print(f'Round [{i_cycle}/{n_queries}], Training Loss: {avg_epoch_loss:.4f}')
        #____________________________________________________________________#

        #not enough instances to do requested amount so reset the indices
        if len(pool_datamodule.unlabeled_indices) < n_instances:
            #pool_datamodule.reset() #dont think reset is the right functionality from experimental context
            n_instances = len(pool_datamodule.unlabeled_indices) #sample whatever is left
            break_early=True
            warnings.warn('Only Completing {} number of rounds due to runnning out of pool data points.'.format(i_cycle+1))

        #validate one epoch
        #____________________________________________________________________#
        if checkpoint_and_eval and eval_dataloader!=None:
            avg_val_loss, min_val_loss, vacc, vprec, vrec, vf1, vhp, vhr, vhf1, lowest_val_loss_model = validation_loop_standard(model, 
                    eval_dataloader, 
                    min_val_loss, 
                    device, 
                    criterion, 
                    save_path+'/round_{}/'.format(i_cycle), 
                    model_name, 
                    le, 
                    train_filename, 
                    eval_filename, 
                    train_data_path, 
                    eval_data_path, 
                    emb_dir, 
                    _format_esm, 
                    i_cycle,
                    active_learning_mode=True)

            val_losses.append(avg_val_loss)
            val_accuracies.append(vacc)
            val_precs.append(vprec)
            val_recs.append(vrec)
            val_f1s.append(vf1)
            val_hps.append(vhp)
            val_hrs.append(vhr)
            val_hf1s.append(vhf1)

            print(f'Round [{i_cycle}/{n_queries}], Validation Loss: {avg_val_loss:.2f}, Validation Accuracy: {100*vacc:.2f}%')

        #____________________________________________________________________#

        seq_ids = [pool_datamodule.query_dataset.dataset.full_list_ids[i] for i in indices]
        ec_ids = [pool_datamodule.train_dataset.id_ec[_id] for _id in seq_ids]

        assert(len(ec_ids) == len(seq_ids))

        save_metrics(seq_ids, save_path+'/round_{}/gene_ids.txt'.format(i_cycle))
        save_metrics(ec_ids, save_path+'/round_{}/ec_ids.txt'.format(i_cycle))
        
        if generate_plots and i_cycle != 0 and al_strat.subset_size == None: #can only do after scores are calculated; NOTE: subset_size messes up indexing below. Need to fix before I can plot when subsets are taken.
            current_pool = [list(pool_ids_full).index(_id) for _id in seq_ids_before_query]
            pool_embeddings_current = pool_embeddings_full[current_pool]
            pool_ids_current = pool_ids_full[current_pool]
            pool_ecs_current = pool_ecs_full[current_pool]

            to_plot = [list(pool_ids_current).index(_id) for _id in seq_ids]
            scores_dict = dict(zip(seq_ids_before_query, scores))
            mapped_id_scores = [scores_dict[id_index] for id_index in pool_ids_current]

            plot_pca_by_uncertainty(pca, 
                    pool_embeddings_current, 
                    pool_embeddings_current[to_plot], 
                    mapped_id_scores, 
                    instance=i_cycle, 
                    path=save_path+'/round_{}/'.format(i_cycle))

            plot_pca_by_class(pca, 
                    le, 
                    pool_embeddings_current, 
                    pool_ecs_current, 
                    pool_embeddings_current[to_plot], 
                    np.array(pool_ecs_current)[to_plot], 
                    instance=i_cycle, 
                    path=save_path+'/round_{}'.format(i_cycle))

        for (test_data_name, test_data_path, test_data) in test_data_list:
            test_model(model, 
                    criterion, 
                    test_data, 
                    device, 
                    checkpoint_dir=save_path+'/round_{}'.format(i_cycle), 
                    metrics_save_path=test_data_name+'_metrics.json')

        #exiting due to lack of points to complete round
        if exit_round:
            break

    save_end_of_training_metrics_standard(model, 
            save_path, 
            model_name, 
            epoch_losses, 
            val_losses, 
            val_accuracies, 
            val_precs,
            val_recs, 
            val_f1s,
            val_hps, 
            val_hrs, 
            val_hf1s, 
            generate_plots, 
            n_queries, 
            checkpoint_and_eval, 
            eval_dataloader,
            metrics_save_path,
            active_learning_mode=True)

    if checkpoint_and_eval and eval_dataloader != None:
        return model, lowest_val_loss_model, (epoch_losses, val_losses, val_accuracies)
    else:
        return model, None, (epoch_losses)
