"""Envoltorios de ``shap.TreeExplainer`` sobre la salida en log-odds."""

import numpy as np
import shap


def tree_shap(model, X, background=None):
    """Calcular valores SHAP de un modelo de árboles en log-odds.

    Con ``background`` usa la versión intervencional sobre ese fondo. Sin
    ``background`` usa la versión dependiente de la trayectoria del árbol
    (``tree_path_dependent``).

    Args:
        model: Modelo de árboles compatible con ``shap.TreeExplainer``.
        X: Instancias a explicar (``DataFrame``).
        background: Fondo opcional (``DataFrame``).

    Returns:
        Tupla ``(explanation, base_value)`` con el ``shap.Explanation`` y el
        valor base ``phi_0``.
    """
    if background is None:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent",
                                       model_output="raw")
    else:
        explainer = shap.TreeExplainer(model, data=background,
                                       feature_perturbation="interventional",
                                       model_output="raw")
    explanation = explainer(X)
    return explanation, float(np.ravel(explainer.expected_value)[0])
