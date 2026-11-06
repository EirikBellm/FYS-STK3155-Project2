"""Check Part c activations, initialization and manual gradients.

Checks activation conventions, seeded initialization and manual regression
gradients for sigmoid/ReLU/Leaky ReLU. JAX is not used for network training.

LLM-assisted: Codex designed and implemented these synthetic checks and independent
JAX reference objectives on 2 November 2026.
"""
from pathlib import Path
import json
import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from reused_network import (
    LEAKY_RELU_SLOPE, activation, activation_derivative,
    create_layers_batched, backpropagation_batched, train_network,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/runge_part_c'
OUT.mkdir(exist_ok=True)
jax.config.update('jax_enable_x64', True)


def reference_loss(layers, X, y, hidden):
    """Evaluate an independent JAX MSE objective.

    LLM-assisted: Codex implemented this reference objective for the surrounding gradient
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
            a = jnp.where(z > 0, z, LEAKY_RELU_SLOPE * z)
    W, b = layers[-1]
    return jnp.mean((a @ W + b - y) ** 2)


X = np.array([[-1.1], [-.7], [-.2], [.15], [.5], [.85], [1.2]])
y = np.random.default_rng(17).normal(size=(7, 1))
records = []
for hidden in ['sigmoid', 'relu', 'leaky_relu']:
    for depth in [1, 2]:
        for width in [25, 50, 100]:
            sizes = [width] * depth + [1]
            schemes = ['xavier' if hidden == 'sigmoid' else 'he'] * depth
            schemes += ['xavier']
            layers = create_layers_batched(
                1, sizes, seed=2026, initialization=schemes
            )
            rng = np.random.default_rng(2026)
            n_in = 1
            for (W, b), n_out, scheme in zip(layers, sizes, schemes):
                variance = 2 / (n_in + n_out) if scheme == 'xavier' else 2 / n_in
                expected = rng.normal(0, np.sqrt(variance), size=(n_in, n_out))
                np.testing.assert_array_equal(W, expected)
                np.testing.assert_array_equal(b, np.zeros(n_out))
                n_in = n_out
            perturb = np.random.default_rng(27)
            shifted = [
                (W + perturb.normal(0, .02, W.shape),
                 b + perturb.normal(0, .02, b.shape))
                for W, b in layers
            ]
            names = [hidden] * depth + ['linear']
            for point, parameters in [('initial', layers), ('perturbed', shifted)]:
                actual = backpropagation_batched(X, parameters, names, y)
                expected = jax.grad(reference_loss)(parameters, X, y, hidden)
                for layer, (pair, reference_pair) in enumerate(zip(actual, expected)):
                    for parameter, gradient, reference in zip(['W', 'b'], pair, reference_pair):
                        np.testing.assert_allclose(
                            gradient, reference, atol=1e-10, rtol=1e-9
                        )
                        records.append({
                            'activation': hidden, 'depth': depth, 'width': width,
                            'point': point, 'layer': layer, 'parameter': parameter,
                            'entries': gradient.size,
                            'max_abs_error': float(np.max(np.abs(gradient - reference))),
                        })

z = np.array([-2., -.5, 0., .5, 2.])
np.testing.assert_array_equal(
    activation(z, 'leaky_relu'), np.array([-.02, -.005, 0., .5, 2.])
)
np.testing.assert_array_equal(
    activation_derivative(z, 'leaky_relu'), np.array([.01, .01, .01, 1., 1.])
)
assert activation_derivative(np.array([0.]), 'relu')[0] == 0
for hidden in ['sigmoid', 'relu', 'leaky_relu']:
    schemes = ['xavier' if hidden == 'sigmoid' else 'he', 'xavier']
    initial = create_layers_batched(1, [25, 1], initialization=schemes)
    _, history = train_network(
        initial, X, y, [hidden, 'linear'], method='adam', gamma=.001,
        epochs=5, batch_size=3, seed=2026,
    )
    assert np.isfinite(history).all()

frame = pd.DataFrame(records)
frame.to_csv(OUT / 'gradient_checks.csv', index=False)
receipt = {
    'checks': [
        'all 18 activation/architecture combinations checked at initial and perturbed parameters',
        'all layer weight/bias gradients agree with independent float64 JAX',
        'exact seeded layerwise Xavier/He draws and zero biases',
        'Leaky ReLU values, slope and zero convention checked',
        'existing ReLU zero convention preserved',
        'finite Adam pilots with a smaller final batch for all three activations',
    ],
    'gradient_arrays_checked': len(frame),
    'gradient_entries_checked': int(frame.entries.sum()),
    'max_gradient_absolute_error': float(frame.max_abs_error.max()),
    'leaky_relu_slope': LEAKY_RELU_SLOPE,
    'implementation_sha256': hashlib.sha256(
        (ROOT / 'Code/reused_network.py').read_bytes()
    ).hexdigest(),
}
(OUT / 'development_checks.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
