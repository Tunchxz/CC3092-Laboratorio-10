"""Valores de Shapley exactos por enumeración de coaliciones."""

from math import factorial

import numpy as np


def coalition_values(f, x, background):
    """Calcular ``v(S)`` para las ``2^n`` coaliciones de variables.

    ``v(S)`` es el promedio de ``f`` sobre el fondo cuando las variables de
    ``S`` toman el valor de ``x`` y las demás conservan el valor del fondo.

    Args:
        f: Función del modelo que recibe ``(m, n)`` y devuelve ``(m,)``.
        x: Instancia de forma ``(n,)``.
        background: Fondo de forma ``(b, n)``.

    Returns:
        Arreglo de forma ``(2^n,)``. La posición ``mask`` corresponde a la
        coalición cuyos bits encendidos indican las variables presentes.
    """
    x = np.asarray(x, dtype=float)
    background = np.asarray(background, dtype=float)
    n_features = x.size
    n_bg = background.shape[0]
    masks = np.arange(2 ** n_features)
    present = (masks[:, None] >> np.arange(n_features)) & 1
    rows = np.repeat(background[None], masks.size, axis=0)
    rows = np.where(present[:, None, :] == 1, x, rows)
    outputs = f(rows.reshape(-1, n_features)).reshape(masks.size, n_bg)
    return outputs.mean(axis=1)


def shapley_from_values(values, n_features):
    """Aplicar la fórmula de Shapley con pesos factoriales.

    ``phi_i = sum_{S sin i} |S|! (n - |S| - 1)! / n! * [v(S U {i}) - v(S)]``.

    Args:
        values: Arreglo de :func:`coalition_values`.
        n_features: Número de variables ``n``.

    Returns:
        Arreglo ``phi`` de forma ``(n,)``.
    """
    n_fact = factorial(n_features)
    phi = np.zeros(n_features)
    for i in range(n_features):
        bit = 1 << i
        for mask in range(2 ** n_features):
            if mask & bit:
                continue
            size = bin(mask).count("1")
            weight = factorial(size) * factorial(n_features - size - 1) / n_fact
            phi[i] += weight * (values[mask | bit] - values[mask])
    return phi
