"""Join prior manual-backpropagation code for Project 2.

Sources: Week 41 (cells 37/39), Week 42 (cells 18, 21, 59, 63, 64),
Week 43 (cell 14), Week 44 (cell 5), and Project 1.

Conventions: float64, samples in rows, W shape (inputs, outputs), zero biases,
normal weights with SD sqrt(1/n_in) by default, optional Xavier/He normal,
full MSE averaged over all output entries, and CE averaged over samples.
Weight-only L1/L2 penalties are full lambda-weighted sums with no extra
sample/layer normalization. Biases are not penalized. L1 uses sign(0)=0.
Leaky ReLU uses a fixed negative slope of 0.01 and that slope at zero.

Student: Specified the network algorithms, mathematical formulas, architecture,
initialization, losses, regularization conventions and training design.

LLM-assisted: Codex integrated and adapted the coursework and Project 1
implementations on 27 October 2026; added activation, initialization, fused-loss and
regularization support on 2 November 2026, including interfaces, caching and
validation.
"""
import numpy as np
from scipy.special import expit, logsumexp
from sklearn.base import BaseEstimator, RegressorMixin

from reused_project1 import initialize_optimizer_state, optimizer_step

# Codex, 2 November 2026: fixed Part c setting, not a tuned parameter.
LEAKY_RELU_SLOPE = 0.01

def create_layers_batched(network_input_size, layer_output_sizes, seed=2026,
                          initialization='fan_in'):
    """Build width-scaled normal weights and zero biases.

    Xavier and He follow course notes section 8.10, Eqs. (8.73) and (8.74).
    A string applies the same initializer to every layer; fan-in remains the default.

    Student: Specified the fan-in, Xavier and He formulas and per-layer initialization
    choices.

    LLM-assisted: Codex adapted Week 43 cell 14 to float64 arrays; added Xavier, He,
    per-layer choices and seeded draws on 2 November 2026.
    """
    rng = np.random.default_rng(seed)
    # allow hidden-layer He with Xavier for the linear output layer.
    if isinstance(initialization, str):
        schemes = [initialization] * len(layer_output_sizes)
    else:
        schemes = list(initialization)
        if len(schemes) != len(layer_output_sizes):
            raise ValueError('Provide one initialization choice per layer')
    layers = []
    n_in = network_input_size
    for n_out, scheme in zip(layer_output_sizes, schemes):
        # NumPy takes standard deviation, not variance.
        if scheme == 'xavier':
            scale = np.sqrt(2.0 / (n_in + n_out))
        elif scheme == 'he':
            scale = np.sqrt(2.0 / n_in)
        elif scheme == 'fan_in':
            scale = np.sqrt(1.0 / n_in)
        else:
            raise ValueError(f'Unsupported initialization: {scheme}')
        W = rng.normal(loc=0.0, scale=scale, size=(n_in, n_out))
        layers.append((W, np.zeros(n_out)))
        n_in = n_out
    return layers

def activation(z, name):
    """Apply a supported elementwise activation or row softmax.

    Student: Specified the sigmoid, ReLU, linear, softmax and fixed-slope Leaky ReLU
    functions.

    LLM-assisted: Codex integrated Weeks 42–44 activations, used stable SciPy expit and
    softmax, and added Leaky ReLU for Part c.
    """
    if name == 'sigmoid':
        return expit(z)
    if name == 'relu':
        return np.maximum(z, 0)
    if name == 'leaky_relu':
        return np.where(z > 0, z, LEAKY_RELU_SLOPE * z)
    if name == 'linear':
        return z
    if name == 'softmax':
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    # unsupported work must fail clearly rather than silently fall back.
    raise ValueError(f'Unsupported activation: {name}')

def activation_derivative(z, name, output=None):
    """Evaluate elementwise derivatives, with ReLU derivative zero at zero.

    Leaky ReLU has derivative 0.01 at zero; ReLU has derivative zero.

    Student: Specified the activation derivatives and ReLU/Leaky ReLU conventions at zero.

    LLM-assisted: Codex adapted Week 42 cells 2/21, added linear and Leaky ReLU branches,
    and added cached sigmoid reuse on 2 November 2026.
    """
    if name == 'sigmoid':
        a = activation(z, name) if output is None else output
        return a * (1 - a)
    if name == 'relu':
        return (z > 0).astype(float)
    if name == 'leaky_relu':
        return np.where(z > 0, 1.0, LEAKY_RELU_SLOPE)
    if name == 'linear':
        return np.ones_like(z)
    raise ValueError(f'No elementwise derivative for {name}')

def feed_forward_saver_batched(inputs, layers, activation_names):
    """Save each layer input and preactivation for manual backpropagation.

    Student: Specified the layerwise forward equations and intermediate values needed for
    backpropagation.

    LLM-assisted: Codex adapted Week 42 cell 59 and the Week 41 row-batch pattern to named
    activations, saved intermediates and shape checks.
    """
    # guard against truncated zip traversal or one-dimensional inputs.
    if len(layers) != len(activation_names) or len(layers) == 0:
        raise ValueError('Provide one activation per layer and at least one layer')
    a = np.asarray(inputs, dtype=float)
    if a.ndim != 2:
        raise ValueError('inputs must have shape (samples, features)')
    layer_inputs, zs = [], []
    for (W, b), name in zip(layers, activation_names):
        layer_inputs.append(a)
        z = a @ W + b
        a = activation(z, name)
        zs.append(z)
    return layer_inputs, zs, a

def feed_forward_batched(inputs, layers, activation_names):
    """Return the final activations.

    Student: Specified the network forward computation.

    LLM-assisted: Codex adapted the Week 42 wrapper to delegate to the shared saved
    forward pass.
    """
    return feed_forward_saver_batched(inputs, layers, activation_names)[2]

def _loss_from_forward(predict, zs, targets, loss, output_activation):
    """Evaluate the reused loss from an existing forward pass.

    Uses full MSE or stable logits-based sample-mean cross-entropy.
    The mathematical objectives are specified by the student in network_loss.

    LLM-assisted: Codex extracted network_loss's unchanged loss formulas on 2 November
    2026 to reuse one forward pass; implemented the shared dispatch and shape checks.
    """
    targets = np.asarray(targets, dtype=float)
    # keep the existing target-shape guard in both calling paths.
    if targets.shape != predict.shape:
        raise ValueError('targets must have exactly the network output shape')
    if loss == 'mse':
        return float(np.mean((predict - targets) ** 2))
    if loss == 'cross_entropy' and output_activation == 'softmax':
        log_p = zs[-1] - logsumexp(zs[-1], axis=1, keepdims=True)
        return float(-np.mean(np.sum(targets * log_p, axis=1)))
    raise ValueError('Use mse, or cross_entropy with a softmax output')


def weight_penalty(layers, penalty=None, lmbd=0.0):
    """Return a full lambda-weighted L1 or squared-L2 weight sum.

    Student: Specified full-sum L1/L2 regularization of weights only, with biases
    excluded, on 2 November 2026.

    LLM-assisted: Codex adapted the Project 1 convention and Week 44/course section 8.12
    examples to network arrays and implemented argument validation.
    """
    # reject unsupported or silently dropped regularization settings.
    if penalty not in [None, 'l1', 'l2']:
        raise ValueError('penalty must be None, l1 or l2')
    if not np.isfinite(lmbd) or lmbd < 0:
        raise ValueError('lmbd must be finite and nonnegative')
    if penalty is None and lmbd != 0:
        raise ValueError('Provide l1 or l2 when lmbd is nonzero')
    if lmbd == 0:
        return 0.0
    if penalty == 'l1':
        return float(lmbd * sum(np.sum(np.abs(W)) for W, _ in layers))
    return float(lmbd * sum(np.sum(W ** 2) for W, _ in layers))


def network_loss(inputs, layers, activation_names, targets, loss='mse',
                 penalty=None, lmbd=0.0):
    """Evaluate the loss plus an optional weight-only L1/L2 penalty.

    Defaults return the unpenalized metric used for model selection.

    Student: Specified full MSE, sample-mean cross-entropy and the optional weight-only
    regularized objective.

    LLM-assisted: Codex combined Week 42 cells 59/63 with stable logit-based CE and the
    optional Part d penalty.
    """
    _, zs, predict = feed_forward_saver_batched(inputs, layers, activation_names)
    value = _loss_from_forward(
        predict, zs, targets, loss, activation_names[-1]
    )
    return value + weight_penalty(layers, penalty, lmbd)

def backpropagation_batched(inputs, layers, activation_names, targets,
                            loss='mse', return_loss=False,
                            penalty=None, lmbd=0.0):
    """Return manual gradients for every weight and bias array.

    CE uses the matched softmax-output error, not an elementwise softmax
    derivative. Penalty gradients are not propagated through earlier layers.
    Bias gradients are unchanged. The default return contains gradients only.

    Student: Specified the manual backpropagation equations, matched softmax-CE output
    gradient and direct weight-only penalty gradients.

    LLM-assisted: Codex integrated Week 42 cells 59/63; added loss dispatch, validation,
    cached sigmoid derivatives and optional fused loss return on 2 November 2026.
    """
    layer_inputs, zs, predict = feed_forward_saver_batched(
        inputs, layers, activation_names
    )
    targets = np.asarray(targets, dtype=float)
    # check shape and CE targets before using the matched-pair formula.
    if targets.shape != predict.shape:
        raise ValueError('targets must have exactly the network output shape')
    penalty_value = weight_penalty(layers, penalty, lmbd)
    value = None
    if return_loss:
        value = _loss_from_forward(
            predict, zs, targets, loss, activation_names[-1]
        ) + penalty_value
    if loss == 'mse':
        delta = 2 * (predict - targets) / predict.size
        if activation_names[-1] != 'linear':
            delta *= activation_derivative(
                zs[-1], activation_names[-1], output=predict
            )
    elif loss == 'cross_entropy' and activation_names[-1] == 'softmax':
        if np.any(targets < 0) or not np.allclose(targets.sum(axis=1), 1):
            raise ValueError('CE target rows must be nonnegative and sum to one')
        m = len(inputs)
        delta = (predict - targets) / m
    else:
        raise ValueError('Use mse, or cross_entropy with a softmax output')
    grads = [None] * len(layers)
    for i in range(len(layers) - 1, -1, -1):
        dW = layer_inputs[i].T @ delta
        # the full penalty has no extra division by batch size.
        if lmbd > 0:
            W, _ = layers[i]
            if penalty == 'l1':
                dW = dW + lmbd * np.sign(W)
            elif penalty == 'l2':
                dW = dW + 2 * lmbd * W
        grads[i] = (dW, np.sum(delta, axis=0))

        if i > 0:
            W, _ = layers[i]
            delta = (delta @ W.T) * activation_derivative(
                zs[i - 1], activation_names[i - 1], output=layer_inputs[i]
            )
    return (grads, value) if return_loss else grads

def train_network(initial_layers, inputs, targets, activation_names,
                  loss='mse', gamma=0.01, epochs=100, batch_size=32,
                  method='gd', seed=2026, penalty=None, lmbd=0.0):
    """Train with manual gradients and reused Project 1 updates.

    Pass batch_size=len(inputs) for full-batch GD. Final batches may be smaller.
    The fused loss and gradients share one forward pass and cached sigmoid
    outputs. Equations, seeded batch order and Project 1 updates are unchanged.
    History is the training objective J, including a nonzero penalty; score
    unpenalized MSE separately with network_loss defaults.

    Student: Specified minibatch training, optimizer choices, shuffled batch order,
    training budgets and optional weight-only penalties.

    LLM-assisted: Codex adapted Week 42 cell 64 / Week 43 cell 14, maintained separate
    Project 1 optimizer states for each W/b array, and fused loss/gradients on 2 November
    2026.
    """
    # clear failures for unsupported settings and nonfinite training.
    if method not in ['gd', 'momentum', 'adagrad', 'rmsprop', 'adam']:
        raise ValueError(f'Unsupported optimizer: {method}')
    if batch_size < 1 or epochs < 1 or gamma <= 0 or len(inputs) == 0:
        raise ValueError('Positive batch_size, epochs, gamma and sample count needed')
    weight_penalty(initial_layers, penalty, lmbd)
    layers = [(W.copy(), b.copy()) for W, b in initial_layers]
    # Recycled helpers: Project 1 initialize_optimizer_state / optimizer_step.
    # New joining code below applies their array interface separately to W and b.
    states = [(initialize_optimizer_state(W, method),
               initialize_optimizer_state(b, method)) for W, b in layers]
    rng = np.random.default_rng(seed)
    losses = []
    for epoch in range(epochs):

        indices = rng.permutation(len(inputs))
        for start in range(0, len(inputs), batch_size):
            batch = indices[start:start + batch_size]
            # one forward pass supplies the loss and manual gradients.
            grads, batch_loss = backpropagation_batched(
                inputs[batch], layers, activation_names, targets[batch],
                loss, return_loss=True, penalty=penalty, lmbd=lmbd,
            )
            if not np.isfinite(batch_loss):
                raise FloatingPointError(
                    f'Nonfinite batch loss at epoch {epoch + 1}'
                )
            for i, ((W, b), (dW, db)) in enumerate(zip(layers, grads)):
                state_W, state_b = states[i]
                W, state_W = optimizer_step(
                    W, dW, state_W, method=method, learning_rate=gamma
                )
                b, state_b = optimizer_step(
                    b, db, state_b, method=method, learning_rate=gamma
                )
                layers[i] = (W, b)
                states[i] = (state_W, state_b)
        value = network_loss(
            inputs, layers, activation_names, targets, loss,
            penalty=penalty, lmbd=lmbd,
        )
        if not np.isfinite(value):
            raise FloatingPointError(f'Nonfinite training loss at epoch {epoch + 1}')
        losses.append(value)
    return layers, np.asarray(losses)


class NeuralNetRegressor(RegressorMixin, BaseEstimator):
    """Fold-safe sklearn interface around the reused manual network.

    Student: Specified manual-gradient regression, full MSE, initialization and
    weight-only penalty options.

    LLM-assisted: Codex adapted Week 44 cell 5 to the shared manual network, replacing JAX
    training, unit-normal weights and half-MSE.
    """

    def __init__(self, hidden=(50,), activation='sigmoid', gamma=0.03,
                 lmbd=0.0, epochs=1000, batch_size=10, seed=2026, method='gd',
                 penalty='l2', initialization='fan_in'):
        """Store estimator settings without fitting.

        LLM-assisted: Codex implemented cloneable sklearn hyperparameter storage from the
        shared network settings.
        """
        self.hidden = hidden
        self.activation = activation
        self.gamma = gamma
        self.lmbd = lmbd
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.method = method
        self.penalty = penalty
        self.initialization = initialization

    def fit(self, X, y):
        """Fit regression using the shared manual network.

        Student: Specified the regression architecture, training settings and weight-only
        penalty objective.

        LLM-assisted: Codex adapted Week 44 estimator fitting, target reshaping and the Part d
        penalty arguments.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(len(X), 1)
        self.n_features_in_ = X.shape[1]
        self.activation_names_ = [self.activation] * len(self.hidden) + ['linear']
        initial = create_layers_batched(
            X.shape[1], list(self.hidden) + [1], self.seed,
            initialization=self.initialization,
        )
        self.layers_, self.loss_curve_ = train_network(
            initial, X, y, self.activation_names_, gamma=self.gamma,
            epochs=self.epochs, batch_size=self.batch_size,
            method=self.method, seed=self.seed,
            penalty=self.penalty, lmbd=self.lmbd,
        )
        self.n_iter_ = self.epochs
        return self

    def predict(self, X):
        """Return one regression prediction per input row.

        LLM-assisted: Codex adapted the Week 44 prediction wrapper to call the shared forward
        pass and flatten its output.
        """
        return feed_forward_batched(
            X, self.layers_, self.activation_names_
        ).ravel()


def binary_cross_entropy(predict, targets):
    """Clipped BCE from Week 43 cell 14.

    This diagnostic probability loss is not used by the softmax trainer.

    Student: Specified the binary cross-entropy formula in the earlier coursework.

    LLM-assisted: Codex extracted the earlier function unchanged in logic.
    """
    p = np.clip(predict, 1e-12, 1 - 1e-12)
    return -np.mean(targets * np.log(p) + (1 - targets) * np.log(1 - p))
