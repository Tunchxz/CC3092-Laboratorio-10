"""Carga, codificación y modelo de gradient boosting del conjunto German Credit."""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

SEED = 42
N_ESTIMATORS = 200
MAX_DEPTH = 3


def load_credit():
    """Descargar German Credit y codificar la clase y las variables categóricas.

    La clase ``good`` se codifica como 1 y ``bad`` como 0. Las variables
    categóricas se codifican con ``OrdinalEncoder`` y se tratan como
    numéricas.

    Returns:
        Tupla ``(X, y, encoder, categorical, numeric)`` con las variables
        codificadas como ``float``, la clase binaria, el codificador ajustado
        y las listas de nombres de columnas categóricas y numéricas.
    """
    datos = fetch_openml("credit-g", version=1, as_frame=True)
    X = datos.data.copy()
    y = (datos.target == "good").astype(int).rename("good")
    categorical = list(X.select_dtypes(include=["category", "object"]).columns)
    numeric = [c for c in X.columns if c not in categorical]
    encoder = OrdinalEncoder()
    X[categorical] = encoder.fit_transform(X[categorical])
    return X.astype(float), y, encoder, categorical, numeric


def split(X, y, seed=SEED):
    """Dividir en 80% de entrenamiento y 20% de prueba de forma estratificada.

    Returns:
        Tupla ``(X_train, X_test, y_train, y_test)``.
    """
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)


def train_gbm(X, y, seed=SEED):
    """Entrenar un ``GradientBoostingClassifier`` con 200 árboles de profundidad 3."""
    model = GradientBoostingClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                                       random_state=seed)
    return model.fit(X, y)


def log_odds_fn(model, columns):
    """Crear una función que devuelve la salida del modelo en log-odds.

    Args:
        model: Clasificador con ``decision_function``.
        columns: Nombres de las columnas con que se entrenó el modelo.

    Returns:
        Función que recibe un arreglo o ``DataFrame`` de forma ``(n, d)`` y
        devuelve un arreglo de forma ``(n,)``.
    """
    def f(X):
        return model.decision_function(pd.DataFrame(np.asarray(X), columns=columns))
    return f


def proba_fn(model, columns):
    """Crear una función que devuelve la probabilidad de la clase ``good``.

    Args:
        model: Clasificador con ``predict_proba``.
        columns: Nombres de las columnas con que se entrenó el modelo.

    Returns:
        Función que recibe un arreglo o ``DataFrame`` de forma ``(n, d)`` y
        devuelve un arreglo de forma ``(n,)``.
    """
    def f(X):
        return model.predict_proba(pd.DataFrame(np.asarray(X), columns=columns))[:, 1]
    return f


def decode(value, column, encoder, categorical):
    """Convertir el código ordinal de una variable categórica a su categoría.

    Las variables numéricas se devuelven sin cambios.
    """
    if column not in categorical:
        return value
    return encoder.categories_[categorical.index(column)][int(round(value))]
