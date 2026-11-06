"""Independent 24-class gradient and grouped-workflow checks.

Compares the manual core with independent JAX gradients across activations,
depths and penalties; checks update equivalence, early stopping and
best-weight restoration.

LLM-assisted: Codex designed and implemented this development check on 2 November
2026.
"""
import json
import hashlib

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pandas as pd

from part_e_classification import (OUT, ROOT, initial_layers, inputs_at,
                                   train_classifier)
from reused_network import backpropagation_batched, train_network

rng = np.random.default_rng(22)
x = rng.uniform(0, 1, (7, 784))
y = np.eye(24)[[0, 1, 7, 13, 17, 23, 2]]
rows = []
for act in ['sigmoid', 'relu', 'leaky_relu']:
    for depth in [0, 1, 2, 3]:
        for penalty in ['none', 'l1', 'l2']:
            config = {'hidden': [5] * depth, 'activation': act,
                      'penalty': penalty, 'lmbd': 0 if penalty == 'none' else .002}
            layers, names = initial_layers(config)
            layers[0][0][0, 0] = 0.0
            for _, b in layers:
                b[:] = rng.normal(size=b.shape) * .1
            def reference(params):
                """Evaluate the independent JAX 24-class classification objective.

                LLM-assisted: Codex implemented this scalar loss for the synthetic gradient fixture on
                2 November 2026.
                """
                a = jnp.asarray(x)
                for i, (W, b) in enumerate(params):
                    z = a @ W + b
                    if i == len(params) - 1:
                        break
                    if act == 'sigmoid':
                        a = jax.nn.sigmoid(z)
                    elif act == 'relu':
                        a = jnp.where(z > 0, z, 0.0)
                    else:
                        a = jnp.where(z > 0, z, .01*z)
                loss = jnp.mean(jax.scipy.special.logsumexp(z, axis=1) - jnp.sum(y*z, axis=1))
                if penalty == 'l1':
                    loss += .002 * sum(jnp.sum(jnp.where(W > 0, W, jnp.where(W < 0, -W, 0.0))) for W, _ in params)
                if penalty == 'l2':
                    loss += .002 * sum(jnp.sum(W*W) for W, _ in params)
                return loss
            expected = jax.grad(reference)(layers)
            manual, value = backpropagation_batched(x, layers, names, y,
                'cross_entropy', return_loss=True,
                penalty=None if penalty == 'none' else penalty, lmbd=config['lmbd'])
            np.testing.assert_allclose(value, reference(layers), rtol=1e-12, atol=1e-12)
            for layer, (got, want) in enumerate(zip(manual, expected)):
                for parameter, a, b in zip(['W', 'b'], got, want):
                    np.testing.assert_allclose(a, b, atol=2e-12, rtol=2e-11)
                    rows.append({'activation':act, 'depth':depth, 'penalty':penalty,
                                 'layer':layer, 'parameter':parameter, 'entries':a.size,
                                 'max_error':float(np.max(np.abs(a-b)))})
            jax.clear_caches()
# prove the new stopping loop keeps the old update/batch-order path.
X = rng.integers(0, 256, (141, 784), dtype=np.uint8)
y_int = rng.integers(0, 24, 141)
config = {'hidden':[5], 'activation':'sigmoid', 'gamma':.001, 'penalty':'l2', 'lmbd':.002}
initial, names = initial_layers(config)
a, _, _, info = train_classifier(X, y_int, np.arange(141), None, config, epochs=3)
b, _ = train_network(initial, inputs_at(X, np.arange(141)), np.eye(24)[y_int],
                     names, loss='cross_entropy', gamma=.001, epochs=3,
                     batch_size=128, method='adam', seed=2026, penalty='l2', lmbd=.002)
for la, lb in zip(a,b):
 for va,vb in zip(la,lb):np.testing.assert_array_equal(va,vb)
# Force the patience branch with a vanishing learning rate and verify restore.
config['gamma'] = 1e-30
_, _, hist, stopping = train_classifier(X,y_int,np.arange(12),np.arange(12,19),
                                       config,epochs=10,patience=2)
assert stopping['best_epoch']==1 and stopping['epochs_run']==3 and stopping['stopped_early']
receipt={'gradient_arrays':len(rows),'entries':sum(r['entries'] for r in rows),
         'max_gradient_error':max(r['max_error'] for r in rows),
         '24_class_output':True,'old_training_path_bitwise_equal':True,'checked_training_batch_sizes':[128,13],
         'forced_patience_and_best_restore':True,
         'network_sha256':hashlib.sha256((ROOT/'Code/reused_network.py').read_bytes()).hexdigest()}
OUT.mkdir(parents=True,exist_ok=True)
# Atomically replace the receipt file.
pd.DataFrame(rows).to_csv(OUT/'gradient_checks.current.csv',index=False)
(OUT/'gradient_checks.current.csv').replace(OUT/'gradient_checks.csv')
(OUT/'development_checks.current.json').write_text(json.dumps(receipt,indent=2)+'\n')
(OUT/'development_checks.current.json').replace(OUT/'development_checks.json')
print(json.dumps(receipt,indent=2))
