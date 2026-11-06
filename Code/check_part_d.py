"""Check weight-only L1/L2 objectives, manual gradients and zero convention.

Checks penalty values and gradients, bias exclusion, validation and the L1
zero convention. JAX is not used for FFNN training.

LLM-assisted: Codex designed and implemented these synthetic checks and independent
JAX reference objectives on 2 November 2026.
"""
from pathlib import Path
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from reused_network import (
    create_layers_batched, backpropagation_batched, network_loss,
    weight_penalty, train_network, NeuralNetRegressor,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/runge_part_d'
jax.config.update('jax_enable_x64', True)


def reference_objective(layers, X, y, hidden, loss, penalty, strength):
    """Evaluate the independent JAX MSE/CE objective with L1/L2 weight penalties.

    LLM-assisted: Codex implemented this scalar reference objective for the surrounding
    checks.
    """
    a = jnp.asarray(X)
    for W, b in layers[:-1]:
        z = a @ W + b
        if hidden == 'sigmoid':
            a = jax.nn.sigmoid(z)
        elif hidden == 'relu':
            a = jax.nn.relu(z)
        else:
            a = jnp.where(z > 0, z, .01 * z)
    W, b = layers[-1]
    z = a @ W + b
    if loss == 'mse':
        result = jnp.mean((z - y) ** 2)
    else:
        result = -jnp.mean(jnp.sum(y * jax.nn.log_softmax(z), axis=1))
    for W, _ in layers:
        if penalty == 'l1':
            # The absolute-value function uses subgradient zero at zero.
            absolute = jnp.where(W > 0, W, jnp.where(W < 0, -W, 0.0))
            result += strength * jnp.sum(absolute)
        elif penalty == 'l2':
            result += strength * jnp.sum(W ** 2)
    return result


rng = np.random.default_rng(42)
X = rng.normal(size=(7, 2))
records = []
for hidden in ['sigmoid', 'relu', 'leaky_relu']:
    for sizes in [[3], [5, 3], [5, 4, 3]]:
        initial = create_layers_batched(2, sizes, initialization='xavier')
        # Nonzero biases detect an unintended bias penalty.
        layers = [(W.copy(), rng.normal(size=b.shape)) for W, b in initial]
        for W, _ in layers:
            W.flat[0] = 0.0
        for loss in ['mse', 'cross_entropy']:
            y = rng.normal(size=(7, 3)) if loss == 'mse' else np.eye(3)[np.arange(7) % 3]
            names = [hidden] * (len(sizes) - 1) + ['linear' if loss == 'mse' else 'softmax']
            plain = backpropagation_batched(X, layers, names, y, loss)
            for penalty in ['l1', 'l2']:
                for strength in [0.0, .002]:
                    actual, value = backpropagation_batched(
                        X, layers, names, y, loss, return_loss=True,
                        penalty=penalty, lmbd=strength,
                    )
                    expected = jax.grad(reference_objective)(layers, X, y, hidden, loss, penalty, strength)
                    np.testing.assert_allclose(
                        value, reference_objective(layers, X, y, hidden, loss, penalty, strength),
                        atol=1e-12, rtol=1e-12,
                    )
                    np.testing.assert_allclose(
                        value, network_loss(X, layers, names, y, loss, penalty=penalty, lmbd=strength),
                        atol=1e-12,
                    )
                    for layer, (pair, ref_pair, plain_pair) in enumerate(zip(actual, expected, plain)):
                        W, _ = layers[layer]
                        extra = strength * np.sign(W) if penalty == 'l1' else 2 * strength * W
                        np.testing.assert_allclose(pair[0], plain_pair[0] + extra, atol=1e-13)
                        np.testing.assert_array_equal(pair[1], plain_pair[1])
                        for parameter, gradient, reference in zip(['W', 'b'], pair, ref_pair):
                            np.testing.assert_allclose(gradient, reference, atol=1e-11, rtol=1e-9)
                            records.append({
                                'hidden': hidden, 'sizes': str(sizes), 'loss': loss,
                                'penalty': penalty, 'lambda': strength, 'layer': layer,
                                'parameter': parameter, 'entries': gradient.size,
                                'max_abs_error': float(np.max(np.abs(gradient - reference))),
                            })
                    shifted_biases = [(W, b + 100) for W, b in layers]
                    assert weight_penalty(layers, penalty, strength) == weight_penalty(shifted_biases, penalty, strength)

for penalty in ['l1', 'l2']:
    initial = create_layers_batched(2, [5, 1], initialization='xavier')
    y = rng.normal(size=(len(X), 1))
    old_layers, old_history = train_network(initial, X, y, ['sigmoid', 'linear'], epochs=5, method='adam', batch_size=3)
    zero_layers, zero_history = train_network(initial, X, y, ['sigmoid', 'linear'], epochs=5, method='adam', batch_size=3, penalty=penalty, lmbd=0)
    for pair, zero_pair in zip(old_layers, zero_layers):
        for array, zero_array in zip(pair, zero_pair):
            np.testing.assert_array_equal(array, zero_array)
    np.testing.assert_array_equal(old_history, zero_history)
    model = NeuralNetRegressor(hidden=(5,), penalty=penalty, lmbd=.001, epochs=5, method='adam')
    model.fit(X, y.ravel())
    assert np.isfinite(model.predict(X)).all()
    _, history = train_network(initial, X, y, ['sigmoid', 'linear'], epochs=5, method='adam', batch_size=3, penalty=penalty, lmbd=.001)
    assert np.isfinite(history).all()
for penalty, strength in [(None, .1), ('other', 0), ('l1', -1), ('l2', np.inf)]:
    try:
        weight_penalty(initial, penalty, strength)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid penalty settings were accepted')
frame = pd.DataFrame(records)
frame.to_csv(OUT / 'gradient_checks.csv', index=False)
receipt = {
    'checks': ['all W/b gradients and full objectives agree with independent float64 JAX',
               'L1 zero subgradient is zero and squared-L2 factor is 2 lambda W',
               'nonzero biases have exactly unchanged gradients and do not affect penalty value',
               'zero strength gives bitwise identical parameters and epoch histories',
               'all three hidden activations, zero/one/two hidden layers, MSE and softmax CE checked',
               'finite L1/L2 pilots, smaller final batch and sklearn wrapper fits',
               'unsupported, negative, nonfinite and silently dropped penalties rejected'],
    'gradient_arrays_checked': len(frame), 'gradient_entries_checked': int(frame.entries.sum()),
    'max_gradient_absolute_error': float(frame.max_abs_error.max()),
    'implementation_sha256': hashlib.sha256((ROOT / 'Code/reused_network.py').read_bytes()).hexdigest(),
}
(OUT / 'development_checks.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
