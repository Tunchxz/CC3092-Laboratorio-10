"""LIME para datos tabulares con regresión ridge ponderada en forma cerrada."""

from itertools import combinations

import numpy as np


def sample_perturbations(x, X_train, n_samples, rng):
    """Generar vectores binarios ``z'`` y sus entradas ``z``.

    Cada componente de ``z'`` vale 1 con probabilidad 0.5. Si vale 1, la
    variable conserva el valor de ``x``; si vale 0, toma un valor al azar de
    la columna correspondiente de ``X_train``.

    Args:
        x: Instancia de forma ``(d,)``.
        X_train: Datos de entrenamiento de forma ``(n, d)``.
        n_samples: Número de muestras ``N``.
        rng: Generador ``numpy.random.Generator``.

    Returns:
        Tupla ``(Z_bin, Z)`` de forma ``(N, d)`` cada una.
    """
    x = np.asarray(x, dtype=float)
    X_train = np.asarray(X_train, dtype=float)
    d = x.size
    Z_bin = (rng.random((n_samples, d)) < 0.5).astype(int)
    rows = rng.integers(0, X_train.shape[0], size=(n_samples, d))
    replacement = X_train[rows, np.arange(d)]
    return Z_bin, np.where(Z_bin == 1, x, replacement)


def kernel_weights(Z_bin, sigma_rel):
    """Calcular ``pi = exp(-D^2 / sigma^2)`` con ``D^2`` igual al número de ausentes.

    Usa ``sigma = sigma_rel * sqrt(d')``.
    """
    d = Z_bin.shape[1]
    sigma = sigma_rel * np.sqrt(d)
    D2 = d - Z_bin.sum(axis=1)
    return np.exp(-D2 / sigma ** 2)


def weighted_ridge(Z_bin, y, weights, lam):
    """Resolver ``w = (Z'^T Pi Z' + lambda I)^{-1} Z'^T Pi y`` con intercepto.

    Se agrega una columna de unos para el intercepto, que no se penaliza.

    Args:
        Z_bin: Matriz binaria de forma ``(N, k)``.
        y: Salidas del modelo de forma ``(N,)``.
        weights: Pesos ``pi`` de forma ``(N,)``.
        lam: Regularización ``lambda``.

    Returns:
        Tupla ``(intercept, coef, fitted)`` con el intercepto, los
        coeficientes de forma ``(k,)`` y las predicciones ``g`` de forma
        ``(N,)``.
    """
    design = np.column_stack([np.ones(len(y)), Z_bin])
    penalty = lam * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    gram = design.T @ (weights[:, None] * design) + penalty
    w = np.linalg.solve(gram, design.T @ (weights * y))
    return w[0], w[1:], design @ w


def weighted_r2(y, fitted, weights):
    """Calcular la fidelidad ponderada ``R_w^2`` del ajuste local."""
    y_bar = np.sum(weights * y) / np.sum(weights)
    sse = np.sum(weights * (y - fitted) ** 2)
    sst = np.sum(weights * (y - y_bar) ** 2)
    return 1.0 - sse / sst


def effective_sample_size(weights):
    """Calcular ``N_ef = (sum pi)^2 / sum pi^2``."""
    return np.sum(weights) ** 2 / np.sum(weights ** 2)


def lime_explain(f, x, X_train, n_samples=5000, k=5, lam=1.0, sigma_rel=0.75, rng=None):
    """Explicar una predicción con LIME tabular.

    Ajusta primero la regresión ponderada sobre todas las variables, conserva
    las ``k`` de mayor ``|w|`` y vuelve a ajustar solo con ellas.

    Args:
        f: Función del modelo que recibe ``(N, d)`` y devuelve ``(N,)``.
        x: Instancia de forma ``(d,)``.
        X_train: Datos de entrenamiento de forma ``(n, d)``.
        n_samples: Número de muestras ``N``.
        k: Número de variables que se conservan.
        lam: Regularización ``lambda``.
        sigma_rel: Ancho relativo del kernel.
        rng: Generador ``numpy.random.Generator``.

    Returns:
        Diccionario con ``coef_full`` (forma ``(d,)``), ``features``
        (índices de las ``k`` variables), ``coef`` y ``intercept`` del
        reajuste, ``r2_w``, ``n_eff`` y ``weights``.
    """
    rng = np.random.default_rng() if rng is None else rng
    Z_bin, Z = sample_perturbations(x, X_train, n_samples, rng)
    y = f(Z)
    weights = kernel_weights(Z_bin, sigma_rel)
    _, coef_full, _ = weighted_ridge(Z_bin, y, weights, lam)
    features = np.argsort(-np.abs(coef_full), kind="stable")[:k]
    intercept, coef, fitted = weighted_ridge(Z_bin[:, features], y, weights, lam)
    return {
        "coef_full": coef_full,
        "features": features,
        "coef": coef,
        "intercept": intercept,
        "r2_w": weighted_r2(y, fitted, weights),
        "n_eff": effective_sample_size(weights),
        "weights": weights,
    }


def jaccard(a, b):
    """Calcular el índice de Jaccard ``|A n B| / |A u B|`` de dos conjuntos."""
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def mean_pairwise_jaccard(sets):
    """Calcular el Jaccard promedio entre todos los pares de una lista de conjuntos."""
    return float(np.mean([jaccard(a, b) for a, b in combinations(sets, 2)]))
