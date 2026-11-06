"""Grouped AL-PUB classification using the reused manual FFNN.

Network, manual backpropagation and Adam come from reused_network and
reused_project1. Publication IDs are not verified writer identities.

Student: Designed the classification architecture, mathematical objectives,
publication-grouped experiment, search and sampling choices, stopping protocol and
comparison/final-evaluation design.

LLM-assisted: Codex built the workflow on 2 November 2026, implementing the design,
orchestration, caching, checkpoints, error handling and result files.
"""
import argparse
import hashlib
import io
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.special import logsumexp
from scipy.stats import loguniform
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold, StratifiedShuffleSplit

from reused_network import (create_layers_batched, feed_forward_saver_batched,
                            backpropagation_batched, weight_penalty)
from reused_project1 import initialize_optimizer_state, optimizer_step

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/classification_part_e'
CACHE = ROOT / 'data/alpub_v2/processed_part_e'
SEED = 2026
NAMES = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta',
         'Theta', 'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu', 'Xi', 'Omicron',
         'Pi', 'Rho', 'LunateSigma', 'Tau', 'Upsilon', 'Phi', 'Chi',
         'Psi', 'Omega']
LETTERS = list('αβγδεζηθικλμνξοπρϲτυφχψω')


def write_json(path, value):
    """Save JSON atomically using a temporary file followed by replacement.

    LLM-assisted: Codex implemented this output/checkpoint helper.
    """
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def prepare(image):
    """Apply unchanged per-image AL-PUB pilot transforms.

    Student: Specified white compositing, grayscale, per-image autocontrast,
    aspect-preserving 26x26 resizing and white padding to 28x28.

    LLM-assisted: Codex retained the original AL-PUB preprocessing function from 27
    October 2026.
    """
    rgba = image.convert('RGBA')
    white = Image.new('RGBA', rgba.size, 'white')
    white.alpha_composite(rgba)
    gray = ImageOps.autocontrast(white.convert('L'))
    gray.thumbnail((26, 26), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (28, 28), 255)
    canvas.paste(gray, ((28 - gray.width) // 2, (28 - gray.height) // 2))
    return canvas


def prepare_data():
    """Cache only development pixels and save checked shared grouped folds.

    All test pixel decoding is deferred until the frozen final assessment.

    Student: Specified publication grouping, the stratified development sample and
    separate grouped fitting, monitoring and validation folds.

    LLM-assisted: Codex implemented manifest reconstruction, count validation, development
    caching, sampling, fold construction and output persistence.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    # rebuild split membership from the unchanged archive and saved
    # dataset checks when the saved manifest is missing.
    # Verify class counts against the saved split summary.
    manifest_path = OUT / 'reconstructed_manifest.csv'
    if not manifest_path.exists():
        audit = json.loads((ROOT / 'results/alpub_v2/audit.json').read_text())
        archive_path = ROOT / 'data/alpub_v2/ALPUB_v2.zip'
        assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == audit['archive_sha256']
        excluded = set(pd.read_csv(ROOT / 'results/alpub_v2/duplicates.csv').path)
        assert len(excluded) == 21
        holdout = set(audit['reserved_test_documents'])
        records = []
        with zipfile.ZipFile(archive_path) as archive:
            for path in sorted(n for n in archive.namelist() if n.lower().endswith('.jpg')):
                match = re.search(r'POxy\.v(\d+)\.n?(\d+)', Path(path).name)
                assert match, path
                group = f'POxy.n{int(match[2])}'
                split = ('excluded' if path in excluded else
                         'reserved_test' if group in holdout else 'development')
                records.append({'path':path, 'label':path.split('/')[2],
                                'document':group, 'status':'kept' if split != 'excluded' else 'conflicting_pixels',
                                'split':split})
        rebuilt = pd.DataFrame(records)
        original_counts = pd.read_csv(ROOT / 'results/alpub_v2/class_counts.csv').set_index('label')
        for split in ['development', 'reserved_test']:
            actual = rebuilt[rebuilt.split == split].groupby('label').size()
            assert (actual.reindex(NAMES).to_numpy() == original_counts.loc[NAMES,split].to_numpy()).all()
        rebuilt.to_csv(manifest_path,index=False)
    manifest = pd.read_csv(manifest_path)
    assert manifest.split.value_counts().to_dict() == {
        'development': 159053, 'reserved_test': 46723, 'excluded': 21}
    dev = manifest[manifest.split == 'development'].copy().reset_index(drop=True)
    test = manifest[manifest.split == 'reserved_test']
    assert not set(dev.document) & set(test.document)
    assert set(dev.status) == {'kept'}
    dev.to_csv(OUT / 'development_manifest.csv', index=False)
    write_json(OUT / 'class_mapping.json', dict(enumerate(NAMES)))
    dest = CACHE / 'development_uint8.npy'
    if not dest.exists():
        start = time.perf_counter()
        tmp = CACHE / 'development_building.npy'
        X = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.uint8,
                                     shape=(len(dev), 784))
        with zipfile.ZipFile(ROOT / 'data/alpub_v2/ALPUB_v2.zip') as archive:
            for i, row in enumerate(dev.itertuples()):
                with Image.open(io.BytesIO(archive.read(row.path))) as image:
                    X[i] = np.asarray(prepare(image), dtype=np.uint8).ravel()
                if (i + 1) % 25000 == 0:
                    print('Development preprocessing', i + 1, flush=True)
        X.flush()
        del X
        tmp.replace(dest)
        write_json(OUT / 'preprocessing_runtime.json', {
            'seconds': time.perf_counter() - start, 'images': len(dev)})
    y = np.array([NAMES.index(name) for name in dev.label], dtype=np.int64)
    groups = dev.document.to_numpy()
    # Use a stratified 24,000-image development sample to limit search cost.
    subset = StratifiedShuffleSplit(n_splits=1, train_size=24000, random_state=SEED)
    chosen, _ = next(subset.split(y, y))
    chosen = np.sort(chosen)
    np.save(OUT / 'search_indices.npy', chosen)
    dev.iloc[chosen].to_csv(OUT / 'search_manifest.csv', index=False)
    folds, counts = {}, []
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for fold, (local_train, local_val) in enumerate(outer.split(y[chosen], y[chosen], groups[chosen])):
        train, val = chosen[local_train], chosen[local_val]
        inner = StratifiedGroupKFold(5, shuffle=True, random_state=SEED + fold + 1)
        fit_local, monitor_local = next(inner.split(y[train], y[train], groups[train]))
        pieces = {'fit': train[fit_local], 'monitor': train[monitor_local],
                  'validation': val}
        sets = [set(groups[idx]) for idx in pieces.values()]
        assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
        assert np.array_equal(np.sort(np.concatenate(list(pieces.values()))),
                              chosen)
        for role, idx in pieces.items():
            assert set(y[idx]) == set(range(24)), f'Missing {role} class, fold {fold}'
            folds[f'{role}_{fold}'] = idx
            for label in range(24):
                counts.append({'fold': fold, 'role': role, 'label': NAMES[label],
                               'images': int(np.sum(y[idx] == label)),
                               'groups': len(set(groups[idx[y[idx] == label]]))})
    folds['y'] = y
    folds['search_indices'] = chosen
    np.savez_compressed(OUT / 'shared_folds.npz', **folds)
    pd.DataFrame(counts).to_csv(OUT / 'fold_class_coverage.csv', index=False)
    draws = list(ParameterSampler({
        'depth': [1, 2, 3], 'width': [32, 64, 128],
        'activation': ['sigmoid', 'relu', 'leaky_relu'],
        'gamma': loguniform(1e-4, 1e-2), 'penalty': ['none', 'l1', 'l2'],
        'positive_lmbd': loguniform(1e-7, 1e-2),
    }, n_iter=12, random_state=SEED))
    configs = []
    for i, d in enumerate(draws):
        configs.append({'id': f'trial_{i:02d}', 'hidden': [int(d['width'])] * int(d['depth']),
                        'activation': d['activation'], 'gamma': float(d['gamma']),
                        'penalty': d['penalty'],
                        'lmbd': float(d['positive_lmbd']) if d['penalty'] != 'none' else 0.0})
    write_json(OUT / 'sampled_configurations.json', configs)
    write_json(OUT / 'design.json', {
        'development': len(dev), 'reserved_test': len(test), 'excluded_blanks': 21,
        'seed': SEED, 'outer_folds': 5, 'inner_monitor_fraction': 'one inner grouped fold out of five',
        'trials': 12, 'search_images': 24000, 'batch_size': 128, 'patience': 10, 'max_epochs': 100,
        'final_budget_rule': 'ceil(median(best_epoch * fit_rows) / 159053), at least one full-development epoch',
        'min_delta': 0.0, 'selection': 'highest mean outer validation macro-F1, 24 fixed classes',
        'monitor': 'unpenalized mean cross-entropy', 'dtype': 'float64 computation, uint8 image cache',
        'preprocessing': 'white alpha composite, grayscale, per-image autocontrast, aspect-preserving thumbnail within 26x26, center white pad to 28x28, invert /255, flatten',
        'augmentation': None, 'penalty': 'weight-only full sums, L1 sign(0)=0',
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'network_sha256': hashlib.sha256((ROOT / 'Code/reused_network.py').read_bytes()).hexdigest()})
    print('Prepared shared folds with all 24 classes and disjoint publication groups', flush=True)


def load_development():
    """Load memory-mapped development images, labels and grouped split metadata.

    LLM-assisted: Codex implemented this cache-access helper.
    """
    return (np.load(CACHE / 'development_uint8.npy', mmap_mode='r'),
            np.load(OUT / 'shared_folds.npz'))


def inputs_at(X, indices):
    """Scale selected uint8 images to inverted float64 intensities.

    Student: Specified ink-as-one normalization, 1 - pixel/255, without fitted statistics.

    LLM-assisted: Codex adapted the pilot conversion to convert only requested rows and
    limit memory use.
    """
    values = np.asarray(X[indices], dtype=np.float64)
    values /= 255.0
    np.subtract(1.0, values, out=values)
    return values


def score(X, y, indices, layers, names, return_prob=False):
    """Return accuracy, macro-F1 and stable logit CE, optionally with probabilities.

    Student: Specified the evaluation metrics, sample-mean CE and fixed 24-class output
    order.

    LLM-assisted: Codex implemented chunked inference, metric calculation and optional
    probability collection.
    """
    probs, total = [], 0.0
    for start in range(0, len(indices), 2048):
        idx = indices[start:start + 2048]
        _, zs, p = feed_forward_saver_batched(inputs_at(X, idx), layers, names)
        total += float(np.sum(logsumexp(zs[-1], axis=1) - zs[-1][np.arange(len(idx)), y[idx]]))
        probs.append(p)
    p = np.concatenate(probs)
    pred = p.argmax(axis=1)
    result = {'cross_entropy': total / len(indices),
              'accuracy': float(accuracy_score(y[indices], pred)),
              'macro_f1': float(f1_score(y[indices], pred, labels=range(24),
                                       average='macro', zero_division=0))}
    return (result, p) if return_prob else result


def initial_layers(config, seed=SEED):
    """Build activation-matched layers using the shared initializer.

    Student: Specified the architecture, Xavier initialization for sigmoid/softmax and He
    initialization for ReLU-family hidden layers.

    LLM-assisted: Codex adapted the Part c layer creation and activation-name setup.
    """
    hidden = config['hidden']
    scheme = 'xavier' if config['activation'] == 'sigmoid' else 'he'
    layers = create_layers_batched(784, hidden + [24], seed=seed,
                                  initialization=[scheme] * len(hidden) + ['xavier'])
    return layers, [config['activation']] * len(hidden) + ['softmax']


def train_classifier(X, y, fit, monitor, config, epochs=100, patience=10,
                     seed=SEED, progress=False):
    """Reuse manual minibatch backprop/Adam with grouped CE early stopping.

    Outer validation data is never passed here. Without monitoring, use a
    fixed development-derived epoch budget and return the final weights.

    Student: Specified the manual-gradient/Adam algorithm, grouped CE monitoring, patience
    rule, best-weight restoration and fixed-epoch refit protocol.

    LLM-assisted: Codex adapted reused_network.train_network, retaining its equations,
    float64, shuffled rows and final partial batch; implemented stopping, state handling
    and training records.
    """
    layers, names = initial_layers(config, seed)
    penalty = None if config['penalty'] == 'none' else config['penalty']
    states = [(initialize_optimizer_state(W, 'adam'), initialize_optimizer_state(b, 'adam'))
              for W, b in layers]
    rng = np.random.default_rng(seed)
    onehot = np.eye(24)[y]
    best, best_ce, best_epoch, stale = None, np.inf, 0, 0
    history = []
    start_time = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(fit)
        epoch_start = time.perf_counter()
        batch_ce = 0.0
        for start in range(0, len(order), 128):
            idx = order[start:start + 128]
            grads, objective = backpropagation_batched(
                inputs_at(X, idx), layers, names, onehot[idx],
                'cross_entropy', return_loss=True, penalty=penalty,
                lmbd=config['lmbd'])
            # fail visibly instead of saving nonfinite runs as scores.
            if not np.isfinite(objective):
                raise FloatingPointError(f'Nonfinite objective at epoch {epoch}')
            batch_ce += len(idx) * (objective - weight_penalty(layers, penalty, config['lmbd']))
            for i, ((W, b), (dW, db)) in enumerate(zip(layers, grads)):
                sw, sb = states[i]
                W, sw = optimizer_step(W, dW, sw, method='adam', learning_rate=config['gamma'])
                b, sb = optimizer_step(b, db, sb, method='adam', learning_rate=config['gamma'])
                layers[i], states[i] = (W, b), (sw, sb)
        row = {'epoch': epoch, 'online_train_ce': batch_ce / len(fit),
               'seconds': time.perf_counter() - epoch_start}
        if monitor is not None:
            metrics = score(X, y, monitor, layers, names)
            row.update({f'monitor_{key}': value for key, value in metrics.items()})
            if metrics['cross_entropy'] < best_ce:
                best_ce, best_epoch, stale = metrics['cross_entropy'], epoch, 0
                best = [(W.copy(), b.copy()) for W, b in layers]
            else:
                stale += 1
        history.append(row)
        if progress and (epoch == 1 or epoch % 20 == 0):
            print(config['id'], 'epoch', epoch, row, flush=True)
        if monitor is not None and stale >= patience:
            break
    if monitor is not None:
        layers = best
    else:
        best_epoch = epoch
    info = {'epochs_run': epoch, 'best_epoch': best_epoch,
            'stopped_early': bool(monitor is not None and stale >= patience),
            'hit_ceiling': epoch == epochs, 'seconds': time.perf_counter() - start_time}
    return layers, names, pd.DataFrame(history), info




def save_weights(path, layers):
    """Save ordered network weight and bias arrays in NPZ format.

    LLM-assisted: Codex implemented this checkpoint serialization helper.
    """
    arrays = {}
    for i, (W, b) in enumerate(layers):
        arrays[f'W_{i}'], arrays[f'b_{i}'] = W, b
    np.savez_compressed(path, **arrays)


def load_weights(path):
    """Load ordered network weight and bias arrays from NPZ format.

    LLM-assisted: Codex implemented this checkpoint deserialization helper.
    """
    with np.load(path) as f:
        return [(f[f'W_{i}'], f[f'b_{i}']) for i in range(len(f.files)//2)]


def fit_fold(config, fold, family='manual'):
    """Fit one development-only job and checkpoint scores and best weights.

    Skip only complete jobs with the same saved config. Failed fits stop
    visibly and are not discarded.

    Student: Specified the grouped fitting/monitoring/validation experiment and metrics.

    LLM-assisted: Codex implemented fold orchestration, model dispatch, checkpoint reuse
    and output persistence.
    """
    folder = OUT / 'fits' / config['id'] / str(fold)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'scores.json').exists():
        saved = json.loads((folder / 'scores.json').read_text())
        assert saved['config'] == config and saved['family'] == family
        return saved
    X, folds = load_development()
    fit, monitor, val = [folds[f'{role}_{fold}'] for role in ['fit','monitor','validation']]
    design = json.loads((OUT / 'design.json').read_text())
    if family == 'library':
        layers, names, history, info = train_library(
            X, folds['y'], fit, monitor, config,
            epochs=design['max_epochs'], patience=design['patience'])
    else:
        layers, names, history, info = train_classifier(
            X, folds['y'], fit, monitor, config,
            epochs=design['max_epochs'], patience=design['patience'])
    history.to_csv(folder / 'learning_curve.csv', index=False)
    save_weights(folder / 'best_weights.npz', layers)
    metrics, prob = score(X, folds['y'], val, layers, names, return_prob=True)
    np.save(folder / 'validation_probabilities.npy', prob)
    train_metrics = score(X, folds['y'], fit, layers, names)
    monitor_metrics = score(X, folds['y'], monitor, layers, names)
    result = {'config':config, 'fold':fold, 'family':family, **info,
              **metrics, **{f'train_{k}':v for k,v in train_metrics.items()},
              **{f'monitor_{k}':v for k,v in monitor_metrics.items()},
              'fit_n':len(fit), 'monitor_n':len(monitor), 'validation_n':len(val)}
    write_json(folder / 'scores.json', result)
    print(config['id'], 'fold', fold, 'macro-F1', metrics['macro_f1'],
          'epochs', info['epochs_run'], 'best', info['best_epoch'],
          'seconds', round(info['seconds'],1), flush=True)
    return result


def summarize():
    """Rank complete five-fold settings by mean macro-F1.

    Missing/unfinished configurations stay out of the pick table and are
    not mistaken for completed five-fold scores.

    LLM-assisted: Codex implemented result aggregation, fold counts, descriptive
    statistics, ranking and summary-table output for the specified selection rule.
    """
    rows = []
    for path in sorted((OUT / 'fits').glob('*/*/scores.json')):
        result = json.loads(path.read_text())
        config = result.pop('config')
        rows.append({**result, **{k:v for k,v in config.items() if k != 'hidden'},
                     'hidden':json.dumps(config['hidden'])})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'fold_scores.csv', index=False)
    summary = df.groupby(['id','family'], as_index=False).agg(
        folds=('fold','count'), macro_f1=('macro_f1','mean'),
        macro_f1_fold_sd=('macro_f1','std'), accuracy=('accuracy','mean'),
        accuracy_fold_sd=('accuracy','std'),
        cross_entropy=('cross_entropy','mean'), cross_entropy_fold_sd=('cross_entropy','std'),
        train_macro_f1=('train_macro_f1','mean'), train_cross_entropy=('train_cross_entropy','mean'),
        best_epoch_median=('best_epoch','median'), max_epochs_run=('epochs_run','max'),
        total_seconds=('seconds','sum'), ceilings=('hit_ceiling','sum'))
    summary = summary[summary.folds == 5].sort_values(['macro_f1','id'],ascending=[False,True])
    summary.to_csv(OUT / 'search_summary.csv',index=False)
    return summary


def search():
    """Run the frozen random draws on the five shared folds.

    No test access.

    Student: Specified the 12-configuration, five-fold development-only search design.

    LLM-assisted: Codex implemented the resumable parallel fold-job runner and incremental
    result summaries.
    """
    assert (OUT / 'development_checks.json').exists()
    configs = json.loads((OUT / 'sampled_configurations.json').read_text())
    from concurrent.futures import ProcessPoolExecutor, as_completed
    # jobs write separate folders. Only this parent writes summaries.
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fit_fold, config, fold)
                   for config in configs for fold in range(5)]
        for future in as_completed(futures):
            future.result()
            summarize()


def train_library(X, y, fit, monitor, config, epochs=100, patience=10):
    """Fit sklearn MLP with the same grouped CE stopping protocol.

    Only sigmoid/ReLU and no penalty are eligible for exact architecture/
    activation/objective matching. sklearn keeps its own initialization, shuffle
    RNG and Adam epsilon placement. partial_fit avoids its built-in random
    image-level monitoring split. It is not used for manual-network checks.

    Student: Specified the matched architecture, activation, objective and grouped
    monitoring comparison.

    LLM-assisted: Codex implemented partial_fit training, optimizer-state retention,
    best-weight restoration, inference checks and result records.
    """
    from sklearn.neural_network import MLPClassifier
    assert config['activation'] in ['sigmoid','relu'] and config['penalty']=='none'
    model = MLPClassifier(hidden_layer_sizes=tuple(config['hidden']),
        activation='logistic' if config['activation']=='sigmoid' else 'relu',
        solver='adam', alpha=0, batch_size=128, learning_rate_init=config['gamma'],
        random_state=np.random.RandomState(SEED), shuffle=True, early_stopping=False)
    # sklearn uses float64 here as in the manual network.
    features = inputs_at(X,fit)
    best, best_ce, best_epoch, stale = None, np.inf, 0, 0
    history=[]
    start=time.perf_counter()
    names=[config['activation']]*len(config['hidden'])+['softmax']
    for epoch in range(1,epochs+1):
        tick=time.perf_counter()
        model.partial_fit(features,y[fit],classes=np.arange(24))
        layers=list(zip(model.coefs_,model.intercepts_))
        row={'epoch':epoch,'online_train_ce':float(model.loss_),
             'seconds':time.perf_counter()-tick}
        if monitor is not None:
            metrics=score(X,y,monitor,layers,names)
            row.update({f'monitor_{k}':v for k,v in metrics.items()})
            if metrics['cross_entropy']<best_ce:
                best_ce,best_epoch,stale=metrics['cross_entropy'],epoch,0
                best=[(W.copy(),b.copy()) for W,b in layers]
            else:stale+=1
        history.append(row)
        if monitor is not None and stale>=patience:break
    if monitor is None:
        best=[(W.copy(),b.copy()) for W,b in layers]
        best_epoch=epoch
    # verify the shared inference matches the fitted library model.
    prob=model.predict_proba(features[:17])
    _,_,ours=feed_forward_saver_batched(features[:17],layers,names)
    np.testing.assert_allclose(prob,ours,atol=1e-12,rtol=1e-12)
    return best,names,pd.DataFrame(history),{
        'epochs_run':epoch,'best_epoch':best_epoch,
        'stopped_early':bool(monitor is not None and stale>=patience),
        'hit_ceiling':epoch==epochs,'seconds':time.perf_counter()-start}


def comparisons():
    """Tune zero-hidden logistic rates and benchmark a matched library FFNN.

    Logistic regression is the same manual softmax network with no hidden
    layer. Its three-rate tuning budget is smaller than the nonlinear search.

    Student: Specified development-only comparator selection, the three-rate logistic
    tuning budget and exposure-derived full-development epoch budgets.

    LLM-assisted: Codex implemented the comparison jobs, summaries and frozen model/epoch
    selections.
    """
    for i, gamma in enumerate([.0001,.001,.01]):
        config={'id':f'logistic_{i}', 'hidden':[], 'activation':'sigmoid',
                'gamma':gamma,'penalty':'none','lmbd':0.0}
        for fold in range(5):fit_fold(config,fold,'logistic')
        summarize()
    summary=summarize()
    configs=json.loads((OUT/'sampled_configurations.json').read_text())
    eligible={c['id']:c for c in configs if c['activation'] in ['sigmoid','relu'] and c['penalty']=='none'}
    compatible=summary[(summary.family=='manual') & summary.id.isin(eligible)]
    assert len(compatible), 'No library-compatible sampled configuration'
    matched=eligible[compatible.iloc[0].id]
    write_json(OUT/'library_match.json',{'manual_id':matched['id'],
        'reason':'highest mean validation macro-F1 among sampled no-penalty sigmoid/ReLU settings',
        'differences':'sklearn initialization, shuffle RNG and Adam implementation, matching architecture, activation, rate, no penalty, precision, batch size, data and stopping protocol'})
    config={**matched,'id':'library_matched'}
    for fold in range(5):fit_fold(config,fold,'library')
    summary=summarize()
    manual_id=summary[summary.family=='manual'].iloc[0].id
    logistic_id=summary[summary.family=='logistic'].iloc[0].id
    choices={}
    for name,cid in [('manual',manual_id),('logistic',logistic_id),('library','library_matched')]:
        first=json.loads((OUT/'fits'/cid/'0'/'scores.json').read_text())
        best_epochs=[json.loads((OUT/'fits'/cid/str(f)/'scores.json').read_text())['best_epoch'] for f in range(5)]
        exposures=[json.loads((OUT/'fits'/cid/str(f)/'scores.json').read_text())['best_epoch'] *
                   json.loads((OUT/'fits'/cid/str(f)/'scores.json').read_text())['fit_n'] for f in range(5)]
        choices[name]={'config':first['config'],
                       'epochs':max(1,int(np.ceil(np.median(exposures)/159053))),
                       'fold_best_epochs':best_epochs,
                       'budget_rule':'match median development-only training-image exposures, rounded up to full-development epochs'}
    # resuming the same study must keep the original pre-test freeze.
    freeze_path = OUT / 'frozen_selection.json'
    if freeze_path.exists():
        assert json.loads(freeze_path.read_text()) == choices, 'Frozen choices changed'
    else:
        write_json(freeze_path, choices)
    print('Fixed picks and full-development epoch budgets before any test processing',choices,flush=True)


def final_assessment():
    """Refit frozen settings on all development rows, then score test once.

    This is the first reserved-test assessment in the Part e workflow.
    No downstream tuning is performed. Test-image decoding occurs only after
    config freeze.

    Student: Specified full-development refitting followed by reserved-test evaluation
    without downstream tuning.

    LLM-assisted: Codex implemented refits, deferred test preprocessing, model scoring and
    saved predictions and metrics.
    """
    assert (OUT/'frozen_selection.json').exists()
    choices=json.loads((OUT/'frozen_selection.json').read_text())
    X,folds=load_development()
    all_dev=np.arange(len(X))
    final=OUT/'final'
    final.mkdir(exist_ok=True)
    for name,pick in choices.items():
        prefix=final/name
        if prefix.with_suffix('.json').exists():continue
        trainer=train_library if name=='library' else train_classifier
        layers,names,hist,info=trainer(X,folds['y'],all_dev,None,pick['config'],epochs=pick['epochs'])
        save_weights(prefix.with_suffix('.npz'),layers)
        hist.to_csv(final/f'{name}_learning_curve.csv',index=False)
        train=score(X,folds['y'],all_dev,layers,names)
        write_json(prefix.with_suffix('.json'),{'pick':pick,'training':train,'training_rows':len(all_dev),**info})
        print('Full-development refit complete',name,info,flush=True)
    # Test decoding occurs only after every model and epoch budget is fixed.
    manifest=pd.read_csv(OUT/'reconstructed_manifest.csv')
    test=manifest[manifest.split=='reserved_test'].reset_index(drop=True)
    dest=CACHE/'reserved_test_uint8.npy'
    if not dest.exists():
        building=CACHE/'test_building.npy'
        features=np.lib.format.open_memmap(building,mode='w+',dtype=np.uint8,shape=(len(test),784))
        with zipfile.ZipFile(ROOT/'data/alpub_v2/ALPUB_v2.zip') as archive:
            for i,row in enumerate(test.itertuples()):
                with Image.open(io.BytesIO(archive.read(row.path))) as image:
                    features[i]=np.asarray(prepare(image),dtype=np.uint8).ravel()
        features.flush()
        del features
        building.replace(dest)
    Xt=np.load(dest,mmap_mode='r')
    yt=np.array([NAMES.index(label) for label in test.label])
    from sklearn.metrics import confusion_matrix, classification_report
    results=[]
    for name,pick in choices.items():
        layers=load_weights(final/f'{name}.npz')
        names=[pick['config']['activation']]*len(pick['config']['hidden'])+['softmax']
        metrics,prob=score(Xt,yt,np.arange(len(Xt)),layers,names,return_prob=True)
        np.save(final/f'{name}_test_probabilities.npy',prob)
        np.savetxt(final/f'{name}_confusion.csv',confusion_matrix(yt,prob.argmax(1),labels=range(24)),fmt='%d',delimiter=',')
        report=classification_report(yt,prob.argmax(1),labels=range(24),target_names=NAMES,output_dict=True,zero_division=0)
        pd.DataFrame(report).T.to_csv(final/f'{name}_per_class.csv')
        results.append({'model':name,**metrics})
    pd.DataFrame(results).to_csv(OUT/'test_scores.csv',index=False)
    print(pd.DataFrame(results).to_string(index=False),flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'search', 'comparisons', 'final'])
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare_data()

    elif args.stage == 'search':
        search()
    elif args.stage == 'comparisons':
        comparisons()
    elif args.stage == 'final':
        final_assessment()
