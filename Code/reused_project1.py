"""Project 1 functions reused unchanged.

Source: Project 1. Original LLM contribution credits are retained.
No old experiment runner is imported or executed.

LLM-assisted: Codex extracted these functions on 27 October 2026.
"""
import numpy as np
import jax
import jax.numpy as jnp
from sklearn.model_selection import KFold

jax.config.update("jax_enable_x64", True)



def runge_function(x):
    """Evaluate the Runge function.

    Returns
    -------
    ndarray
        Runge-function values.

    LLM-assisted
    ------------
    Tool/date: ChatGPT 5.6 Sol, browser, September 2026
    Contribution: Generated the initial implementation from my specified
        inputs, outputs, internal logic, and Runge formula
        f(x) = 1 / (1 + 25x^2).
    Changes: None.
    """
    return 1.0 / (1.0 + 25.0 * x**2)



def generate_data(n_points, noise_std=0.1, seed=None):
    """Generate noisy Runge-function data.

    Returns
    -------
    tuple of ndarray
        Input values, noisy targets, and noise-free targets.

    LLM-assisted
    ------------
    Tool/date: ChatGPT 5.6 Sol, browser, September 2026
    Contribution: Generated the initial implementation from my specified
        inputs, outputs, internal logic, and Runge-data and additive
        Gaussian-noise formulas.
    Changes: None.
    """
    rng = np.random.default_rng(seed)

    # Fixed, evenly spaced inputs keep locations comparable across experiments.
    x = np.linspace(-1, 1, n_points)
    y_true = runge_function(x)

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=n_points
    )

    y = y_true + noise

    return x, y, y_true



def create_design_matrix(x, degree):
    """Construct polynomial features x, x^2, ..., x^degree.

    No intercept column is included.

    Returns
    -------
    ndarray
        Polynomial design matrix without an intercept column.

    LLM-assisted
    ------------
    Tool/date: ChatGPT 5.6 Sol, browser, September 2026
    Contribution: Generated the initial implementation from my specified
        inputs, outputs, internal logic, and polynomial-basis formula
        x, x^2, ..., x^degree.
    Changes: None.
    """
    # Centering y handles the intercept, so no constant column is included.
    x = np.asarray(x).ravel()

    return np.column_stack(
        [x**p for p in range(1, degree + 1)]
    )



def scale_data(X_train, X_test, y_train, y_test=None):
    """Center and scale X using training statistics and center y.

    Returns the training statistics needed to transform test
    data and restore predictions to the original y scale.

    Returns
    -------
    tuple
        Scaled features, centered targets, and training-set statistics.

    LLM-assisted
    ------------
    Tool/date: ChatGPT 5.6 Sol, browser, September 2026
    Contribution: Generated the initial implementation from my specified
        inputs, outputs, internal logic, and standardization and target-centering
        formulas.
    Changes: None.
    """

    # Fit preprocessing on the training set only to avoid test-data leakage.
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)

    # LLM-assisted: ChatGPT suggested this zero-variance guard. I reviewed and
    # verified it against constant input columns.
    X_std[X_std == 0] = 1.0

    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std

    # Centering y removes the need for an explicit intercept during fitting.
    y_mean = np.mean(y_train)

    y_train_centered = y_train - y_mean

    # LLM-assisted: ChatGPT added optional test-target handling. I reviewed and
    # tested both branches.
    if y_test is not None:
        y_test_centered = y_test - y_mean
    else:
        y_test_centered = None

    return (
        X_train_scaled,
        X_test_scaled,
        y_train_centered,
        y_test_centered,
        X_mean,
        X_std,
        y_mean
    )



def mse(y_true, y_pred):
    """Mean squared error.

    Returns
    -------
    float
        Mean squared error.

    LLM-assisted
    ------------
    Tool/date: ChatGPT 5.6 Sol, browser, September 2026
    Contribution: Generated the initial implementation from my specified
        inputs, outputs, internal logic, and mean squared error formula.
    Changes: None.
    """
    return np.mean((y_true - y_pred)**2)



def r2_score(y_true, y_pred):
    """Coefficient of determination R^2.

    Returns
    -------
    float
        Coefficient of determination.

    LLM-assisted
    ------------
    Tool/date: ChatGPT 5.6 Sol, browser, September 2026
    Contribution: Generated the initial implementation from my specified
        inputs, outputs, internal logic, and R^2 = 1 - SS_res / SS_tot formula.
    Changes: None.
    """
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((y_true - np.mean(y_true))**2)

    return 1.0 - numerator / denominator



def fit_regression(X, y, lambda_=0.0):
    """Fit OLS or Ridge regression.

    lambda_ = 0  -> OLS
    lambda_ > 0  -> Ridge

    Returns
    -------
    ndarray
        Estimated OLS or Ridge coefficients.

    LLM-assisted
    ------------
    Tool/date: ChatGPT 5.6 Sol, browser, September 2026
    Contribution: Generated the initial implementation from my specified
        inputs, outputs, internal logic, and OLS pseudoinverse and Ridge
        normal-equation formulas.
    Changes: I requested Ridge support in Part B. OpenAI Codex implemented
        lambda_ while preserving Part A OLS as lambda_=0, and I reviewed it.
    """

    # The pseudoinverse remains defined for rank-deficient design matrices.
    if lambda_ == 0:
        theta = np.linalg.pinv(X) @ y

    else:
        n_parameters = X.shape[1]
        n_samples = X.shape[0]

        # Mean squared loss gives the n * lambda Ridge convention.
        theta = np.linalg.solve(
            X.T @ X + n_samples * lambda_ * np.eye(n_parameters),
            X.T @ y
        )

    return theta



def predict(X, theta, y_mean=0.0):
    """Predict and restore the centered target mean.

    Returns
    -------
    ndarray
        Predictions restored to the original target scale.

    LLM-assisted
    ------------
    Tool/date: ChatGPT 5.6 Sol, browser, September 2026
    Contribution: Generated the initial implementation from my specified
        inputs, outputs, internal logic, and prediction formula
        y_hat = X theta + y_mean.
    Changes: None.
    """
    return X @ theta + y_mean



def make_cv_splits(n_samples, k=5, seed=None):
    """Create one reusable set of shuffled cross-validation folds.

    Returns
    -------
    list of tuple
        Training and validation index arrays for every fold.

    LLM-assisted
    ------------
    Tool/date: ChatGPT 5.6 Sol, browser, September 2026
    Contribution: Generated the initial implementation from my specified
        inputs, outputs, fold-construction logic, and common-fold comparison
        rule.
    Changes: None.
    """
    # Reuse the same folds across candidate models for a fair comparison.
    splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
    indices = np.arange(n_samples)
    return [(train.copy(), validation.copy()) for train, validation in splitter.split(indices)]



def cross_validation_scores(
    x,
    y,
    degree,
    folds,
    model="ols",
    lambda_=0.0,
    lasso_options=None
):
    """Evaluate OLS, Ridge, or Lasso on an explicitly supplied set of folds.

    Returns
    -------
    tuple
        Validation MSE values and fold-level convergence diagnostics.

    LLM-assisted
    ------------
    Tool/date: ChatGPT 5.6 Sol, browser, September 2026
    Contribution: Generated the initial implementation from my specified
        inputs, outputs, leakage-free preprocessing, model formulas, and
        common-fold comparison logic.
    Changes: I requested Lasso support and fold-level convergence diagnostics
        for Part I. OpenAI Codex implemented them, and I reviewed them.
    """
    options = {} if lasso_options is None else dict(lasso_options)
    scores = []
    diagnostics = []

    for train_idx, validation_idx in folds:
        X_train = create_design_matrix(x[train_idx], degree)
        X_validation = create_design_matrix(x[validation_idx], degree)
        y_train = y[train_idx]
        y_validation = y[validation_idx]

        # Fit preprocessing inside each training fold to avoid leakage.
        (
            X_train,
            X_validation,
            y_train_centered,
            _,
            _,
            _,
            y_mean
        ) = scale_data(X_train, X_validation, y_train, y_validation)

        if model == "ols":
            theta = fit_regression(X_train, y_train_centered, lambda_=0.0)
            iterations, converged = 1, True
        elif model == "ridge":
            theta = fit_regression(X_train, y_train_centered, lambda_=lambda_)
            iterations, converged = 1, True
        elif model == "lasso":
            defaults = {
                "regularization": "lasso",
                "gradient_source": "automatic",
                "method": "adam",
                "learning_rate": 1e-2,
                "tolerance": 1e-7,
                "max_iterations": 3000,
                "seed": 42,
            }
            defaults.update(options)
            theta, iterations, _, converged = optimize(
                X_train,
                y_train_centered,
                lambda_=lambda_,
                **defaults
            )
        # LLM-assisted: ChatGPT added input validation. I reviewed and tested it.
        else:
            raise ValueError("model must be 'ols', 'ridge', or 'lasso'")

        prediction = predict(X_validation, theta, y_mean)
        scores.append(mse(y_validation, prediction))
        diagnostics.append({"iterations": iterations, "converged": converged})

    return np.asarray(scores), diagnostics



def cost_function(
    theta,
    X,
    y,
    lambda_=0.0,
    regularization=None
):
    """Regression cost with optional Ridge or Lasso regularization.

    regularization:
        None
        "ridge"
        "lasso"

    Returns
    -------
    jax.Array
        Regression cost including any requested penalty.

    LLM-assisted
    ------------
    Tool/date: ChatGPT 5.6 Sol, browser, September 2026
    Contribution: Generated the initial implementation from my specified
        inputs, outputs, selection logic, and OLS, Ridge, and Lasso cost formulas.
    Changes: I requested the Part G Lasso objective. OpenAI Codex implemented
        it while preserving Part E OLS/Ridge behaviour, and I reviewed it.
    """

    # Mean squared loss
    residual = X @ theta - y

    cost = jnp.mean(residual**2)

    if regularization is None:
        return cost

    elif regularization == "ridge":
        return cost + lambda_ * jnp.sum(theta**2)

    elif regularization == "lasso":
        return cost + lambda_ * jnp.sum(jnp.abs(theta))

    else:
        raise ValueError(
            "regularization must be None, 'ridge', or 'lasso'"
        )



def analytical_gradient(
    theta,
    X,
    y,
    lambda_=0.0,
    regularization=None
):
    """Analytical gradient for OLS and Ridge.

    Returns
    -------
    ndarray
        Analytical OLS or Ridge gradient.

    LLM-assisted
    ------------
    Tool/date: ChatGPT 5.6 Sol, browser, September 2026
    Contribution: Generated the initial implementation from my specified
        inputs, outputs, internal logic, and analytical OLS and Ridge gradient
        formulas.
    Changes: None.
    """

    residual = X @ theta - y

    n_samples = X.shape[0]
    # Differentiating the mean squared loss gives the factor 2 / n.
    gradient = (2.0 / n_samples) * X.T @ residual

    if regularization is None:
        return gradient

    elif regularization == "ridge":
        return gradient + 2.0 * lambda_ * theta

    # LLM-assisted: ChatGPT added this unsupported-case check. I reviewed and
    # tested it.
    else:
        raise ValueError(
            "Analytical gradient implemented only for OLS/Ridge."
        )


jax_gradient = jax.grad(cost_function, argnums=0)



def automatic_gradient(
    theta,
    X,
    y,
    lambda_=0.0,
    regularization=None
):
    """Calculate gradient using JAX automatic differentiation.

    Returns
    -------
    jax.Array
        Gradient calculated by JAX automatic differentiation.

    LLM-assisted
    ------------
    Tool/date: ChatGPT 5.6 Sol, browser, September 2026
    Contribution: Generated the initial implementation from my specified
        inputs, outputs, mathematical objective, and JAX automatic-differentiation
        logic.
    Changes: I requested JAX support for the Part G Lasso objective. ChatGPT
        Codex implemented it without a separate Lasso gradient, and I reviewed it.
    """

    theta = jnp.asarray(theta)
    X = jnp.asarray(X)
    y = jnp.asarray(y)

    return jax_gradient(
        theta,
        X,
        y,
        lambda_,
        regularization
    )



def check_gradient(
    X,
    y,
    lambda_=0.0,
    regularization=None,
    n_tests=5,
    seed=None
):
    """Compare analytical and JAX gradients at random theta values.

    Returns an array of relative errors from the gradient comparisons.

    LLM-assisted: ChatGPT 5.6 Sol, browser, September 2026: implemented the randomized
    gradient-check helper.
    """

    rng = np.random.default_rng(seed)

    errors = []

    # Test away from the optimum, where both gradients could be near zero.
    for _ in range(n_tests):

        theta = rng.normal(size=X.shape[1])

        g_analytical = analytical_gradient(
            theta,
            X,
            y,
            lambda_=lambda_,
            regularization=regularization
        )

        g_jax = np.asarray(
            automatic_gradient(
                theta,
                X,
                y,
                lambda_=lambda_,
                regularization=regularization
            )
        )

        err = relative_error(
            g_analytical,
            g_jax
        )

        errors.append(err)

    return np.array(errors)



def initialize_optimizer_state(theta, method):
    """Initialize state variables required by an optimizer.

    Returns
    -------
    dict
        State variables required by the selected optimizer.

    LLM-assisted
    ------------
    Tool/date: ChatGPT 5.6 Sol, browser, September 2026
    Contribution: Generated the initial implementation from my specified
        inputs, outputs, optimizer logic, and required mathematical state variables.
    Changes: I requested optimizer state for Part F. OpenAI Codex implemented
        state for Momentum, AdaGrad, RMSProp, and Adam, and I reviewed it.
    """

    state = {}

    if method == "momentum":
        state["velocity"] = np.zeros_like(theta)

    elif method == "adagrad":
        state["G"] = np.zeros_like(theta)

    elif method == "rmsprop":
        state["G"] = np.zeros_like(theta)

    elif method == "adam":
        state["m"] = np.zeros_like(theta)
        state["v"] = np.zeros_like(theta)
        state["t"] = 0

    return state



def optimizer_step(
    theta,
    gradient,
    state,
    method="gd",
    learning_rate=0.01,
    momentum=0.9,
    rho=0.9,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8
):
    """Perform one optimizer update.

    Returns
    -------
    tuple
        Updated coefficients and optimizer state.

    LLM-assisted
    ------------
    Tool/date: ChatGPT 5.6 Sol, browser, September 2026
    Contribution: Generated the initial implementation from my specified
        inputs, outputs, internal logic, and GD, momentum, AdaGrad, RMSProp, and
        Adam update formulas.
    Changes: I requested the Part F optimizer expansion. OpenAI Codex added
        Momentum, AdaGrad, RMSProp, and Adam branches, and I reviewed them.
    """

    if method == "gd":

        theta = theta - learning_rate * gradient


    elif method == "momentum":

        # Momentum carries part of the previous update direction forward.
        state["velocity"] = (
            momentum * state["velocity"]
            - learning_rate * gradient
        )

        theta = theta + state["velocity"]


    elif method == "adagrad":

        # AdaGrad scales parameters by accumulated squared gradients.
        state["G"] += gradient**2


        theta = theta - (
            learning_rate
            * gradient
            / (np.sqrt(state["G"]) + epsilon)
        )


    elif method == "rmsprop":

        # RMSProp uses exponential averaging to avoid AdaGrad's continual decay.
        state["G"] = (
            rho * state["G"]
            + (1 - rho) * gradient**2
        )


        theta = theta - (
            learning_rate
            * gradient
            / (np.sqrt(state["G"]) + epsilon)
        )


    elif method == "adam":

        # Adam combines first and second moments with bias correction.
        state["t"] += 1

        state["m"] = (
            beta1 * state["m"]
            + (1 - beta1) * gradient
        )

        state["v"] = (
            beta2 * state["v"]
            + (1 - beta2) * gradient**2
        )

        m_hat = (
            state["m"]
            / (1 - beta1**state["t"])
        )

        v_hat = (
            state["v"]
            / (1 - beta2**state["t"])
        )


        theta = theta - (
            learning_rate
            * m_hat
            / (np.sqrt(v_hat) + epsilon)
        )


    # LLM-assisted: ChatGPT added input validation. I reviewed and tested it.
    else:
        raise ValueError(f"Unknown optimizer: {method}")

    return theta, state



def optimize(
    X,
    y,
    lambda_=0.0,
    regularization=None,
    gradient_source="automatic",
    method="gd",
    learning_rate=0.001,
    tolerance=1e-8,
    max_iterations=10000,
    seed=None
):
    """Full-batch gradient optimization using JAX gradients.

    Returns
    -------
    tuple
        Coefficients, iteration count, cost history, and convergence status.

    LLM-assisted
    ------------
    Tool/date: ChatGPT 5.6 Sol, browser, September 2026
    Contribution: Generated the initial implementation from my specified
        inputs, outputs, optimization formulas, and full-batch convergence and
        history logic.
    Changes: I requested five optimizer choices, Part G Lasso support, and an
        explicit convergence result. OpenAI Codex implemented them, and I
        reviewed them.
    """

    rng = np.random.default_rng(seed)

    theta = rng.normal(
        scale=0.1,
        size=X.shape[1]
    )

    state = initialize_optimizer_state(
        theta,
        method
    )

    history = []
    converged = False

    # Full-batch optimization uses all training observations in every update.
    for iteration in range(max_iterations):

        if gradient_source == "automatic":
            gradient = np.asarray(
                automatic_gradient(
                    theta,
                    X,
                    y,
                    lambda_=lambda_,
                    regularization=regularization
                )
            )
        elif gradient_source == "analytical":
            gradient = analytical_gradient(
                theta,
                X,
                y,
                lambda_=lambda_,
                regularization=regularization
            )
        # LLM-assisted: ChatGPT added input validation. I reviewed and tested it.
        else:
            raise ValueError("gradient_source must be 'automatic' or 'analytical'")

        theta_new, state = optimizer_step(
            theta,
            gradient,
            state,
            method=method,
            learning_rate=learning_rate
        )

        history.append(
            float(
                cost_function(
                    jnp.asarray(theta_new),
                    jnp.asarray(X),
                    jnp.asarray(y),
                    lambda_,
                    regularization
                )
            )
        )

        # Stop when successive parameter vectors differ by less than tolerance.
        if np.linalg.norm(theta_new - theta) < tolerance:
            theta = theta_new
            converged = True
            break

        theta = theta_new

    return theta, iteration + 1, np.array(history), converged



def relative_error(a, b):
    """Relative L2 error between two vectors.

    Returns
    -------
    float
        Relative L2 error.

    LLM-assisted
    ------------
    Tool/date: ChatGPT 5.6 Sol, browser, September 2026
    Contribution: Generated the initial implementation from my specified
        inputs, outputs, internal logic, and relative L2-error formula
        ||a - b|| / ||a||.
    Changes: None.
    """
    return np.linalg.norm(a - b) / np.linalg.norm(a)
