"""Explicaciones contrafactuales por muestreo aleatorio de candidatos."""

import numpy as np


def mad_scale(X_train):
    """Calcular la desviación absoluta mediana de cada columna.

    Si la mediana de las desviaciones es cero, usa la desviación absoluta
    media.

    Args:
        X_train: Datos de entrenamiento de forma ``(n, d)``.

    Returns:
        Arreglo de forma ``(d,)``.
    """
    X_train = np.asarray(X_train, dtype=float)
    mad = np.median(np.abs(X_train - np.median(X_train, axis=0)), axis=0)
    mean_ad = np.mean(np.abs(X_train - X_train.mean(axis=0)), axis=0)
    return np.where(mad > 0, mad, mean_ad)


def distance(x, candidates, numeric_idx, categorical_idx, scale):
    """Calcular la distancia de cada candidato a la instancia original.

    ``d(x, x') = sum_num |x_j - x'_j| / MAD_j + sum_cat 1[x_j != x'_j]``.

    Args:
        x: Instancia original de forma ``(d,)``.
        candidates: Candidatos de forma ``(m, d)``.
        numeric_idx: Índices de las variables numéricas.
        categorical_idx: Índices de las variables categóricas.
        scale: Escala ``MAD_j`` de forma ``(d,)``.

    Returns:
        Arreglo de forma ``(m,)``.
    """
    diff = np.abs(candidates - x)
    numeric = (diff[:, numeric_idx] / scale[numeric_idx]).sum(axis=1)
    categorical = (diff[:, categorical_idx] > 0).sum(axis=1)
    return numeric + categorical


def generate_candidates(x, X_train, actionable_idx, n_candidates, max_changes, rng):
    """Generar candidatos que cambian entre 1 y ``max_changes`` variables accionables.

    Cada candidato elige un subconjunto aleatorio de variables accionables y
    les asigna valores tomados al azar de los observados en ``X_train``.

    Args:
        x: Instancia original de forma ``(d,)``.
        X_train: Datos de entrenamiento de forma ``(n, d)``.
        actionable_idx: Índices de las variables que se pueden cambiar.
        n_candidates: Número de candidatos.
        max_changes: Máximo de variables cambiadas por candidato.
        rng: Generador ``numpy.random.Generator``.

    Returns:
        Arreglo de candidatos de forma ``(n_candidates, d)``.
    """
    x = np.asarray(x, dtype=float)
    X_train = np.asarray(X_train, dtype=float)
    actionable_idx = np.asarray(actionable_idx)
    candidates = np.tile(x, (n_candidates, 1))
    upper = min(max_changes, actionable_idx.size)
    for c in range(n_candidates):
        n_changes = rng.integers(1, upper + 1)
        chosen = rng.choice(actionable_idx, size=n_changes, replace=False)
        rows = rng.integers(0, X_train.shape[0], size=n_changes)
        candidates[c, chosen] = X_train[rows, chosen]
    return candidates


def select_diverse(x, candidates, dist, valid, n_select=3):
    """Elegir contrafactuales válidos de menor distancia con cambios distintos.

    Toma el candidato válido de menor distancia y luego, en cada paso, el de
    menor distancia cuyo conjunto de variables cambiadas no se haya elegido.

    Args:
        x: Instancia original de forma ``(d,)``.
        candidates: Candidatos de forma ``(m, d)``.
        dist: Distancias de forma ``(m,)``.
        valid: Máscara booleana de candidatos válidos.
        n_select: Número de contrafactuales a elegir.

    Returns:
        Lista de índices de ``candidates``.
    """
    chosen, seen = [], set()
    for idx in np.argsort(dist, kind="stable"):
        if not valid[idx]:
            continue
        changed = frozenset(np.flatnonzero(candidates[idx] != x))
        if changed in seen:
            continue
        chosen.append(int(idx))
        seen.add(changed)
        if len(chosen) == n_select:
            break
    return chosen
