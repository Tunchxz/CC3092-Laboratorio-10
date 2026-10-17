"""Validación de mapas de saliencia: deletion, insertion y aleatorización en cascada."""

import copy

import numpy as np
import torch
from scipy.stats import spearmanr
from torchvision.transforms.functional import gaussian_blur

from .gradcam import grad_cam
from .images import gray_like, predict_proba

CASCADE_STAGES = ("fc", "layer4", "layer3", "layer2", "layer1", "conv1")


def blurred_like(x, sigma=10.0):
    """Desenfocar una imagen normalizada con un filtro gaussiano.

    Args:
        x: Imagen normalizada de forma ``(1, 3, H, W)``.
        sigma: Desviación estándar del filtro en píxeles.

    Returns:
        Tensor desenfocado con la forma de ``x``.
    """
    kernel = 2 * int(np.ceil(3 * sigma)) + 1
    return gaussian_blur(x, kernel_size=[kernel, kernel], sigma=[sigma, sigma])


def pixel_order(heatmap):
    """Ordenar los índices de píxel de mayor a menor valor del mapa."""
    return np.argsort(-np.asarray(heatmap).ravel(), kind="stable")


def perturbation_curve(model, x, order, class_idx, mode, reference, steps=20):
    """Calcular la curva de deletion o de insertion de una imagen.

    En ``deletion`` se parte de ``x`` y en el paso ``k`` se reemplaza por
    ``reference`` la fracción ``k / steps`` de los píxeles indicados primero
    en ``order``. En ``insertion`` se parte de ``reference`` y se copian esos
    píxeles desde ``x``.

    Args:
        model: Clasificador de PyTorch.
        x: Imagen normalizada de forma ``(1, 3, H, W)``.
        order: Índices de píxel aplanados, del más al menos importante.
        class_idx: Clase cuya probabilidad se registra.
        mode: ``"deletion"`` o ``"insertion"``.
        reference: Imagen de reemplazo con la forma de ``x``.
        steps: Número de pasos ``P``.

    Returns:
        Tupla ``(curve, auc)``. ``curve`` tiene ``steps + 1`` probabilidades,
        de ``k = 0`` a ``k = steps``, y ``auc`` es el promedio de ``c_1`` a
        ``c_P``.
    """
    if mode not in ("deletion", "insertion"):
        raise ValueError("mode debe ser 'deletion' o 'insertion'")
    _, _, height, width = x.shape
    n_pixels = height * width
    start, end = (x, reference) if mode == "deletion" else (reference, x)
    rank = np.empty(n_pixels, dtype=np.int64)
    rank[np.asarray(order)] = np.arange(n_pixels)
    rank = torch.from_numpy(rank).view(1, 1, height, width)
    batch = []
    for k in range(steps + 1):
        n_changed = int(round(k / steps * n_pixels))
        mask = (rank < n_changed).to(x.dtype)
        batch.append(start * (1 - mask) + end * mask)
    probs = predict_proba(model, torch.cat(batch))[:, class_idx].numpy()
    return probs, float(probs[1:].mean())


def faithfulness_scores(model, x, heatmap, class_idx, rng, mode, reference,
                        n_random=10, steps=20):
    """Comparar la curva del mapa contra órdenes aleatorios de los píxeles.

    Args:
        model: Clasificador de PyTorch.
        x: Imagen normalizada de forma ``(1, 3, H, W)``.
        heatmap: Mapa de forma ``(H, W)`` que define el orden de los píxeles.
        class_idx: Clase cuya probabilidad se registra.
        rng: Generador ``numpy.random.Generator`` para los órdenes aleatorios.
        mode: ``"deletion"`` o ``"insertion"``.
        reference: Imagen de reemplazo con la forma de ``x``.
        n_random: Número de órdenes aleatorios.
        steps: Número de pasos ``P``.

    Returns:
        Diccionario con ``curve`` y ``auc`` del mapa, ``random_curves`` de
        forma ``(n_random, steps + 1)`` y ``random_aucs`` de forma
        ``(n_random,)``.
    """
    curve, auc = perturbation_curve(model, x, pixel_order(heatmap), class_idx,
                                    mode, reference, steps)
    random_curves, random_aucs = [], []
    n_pixels = x.shape[-1] * x.shape[-2]
    for _ in range(n_random):
        rc, ra = perturbation_curve(model, x, rng.permutation(n_pixels), class_idx,
                                    mode, reference, steps)
        random_curves.append(rc)
        random_aucs.append(ra)
    return {
        "curve": curve,
        "auc": auc,
        "random_curves": np.array(random_curves),
        "random_aucs": np.array(random_aucs),
    }


def deletion_insertion(model, x, heatmap, class_idx, rng, n_random=10, steps=20,
                       blur_sigma=10.0):
    """Calcular deletion con gris medio e insertion desde una versión desenfocada.

    Args:
        model: Clasificador de PyTorch.
        x: Imagen normalizada de forma ``(1, 3, H, W)``.
        heatmap: Mapa de forma ``(H, W)``.
        class_idx: Clase cuya probabilidad se registra.
        rng: Generador ``numpy.random.Generator``.
        n_random: Número de órdenes aleatorios.
        steps: Número de pasos ``P``.
        blur_sigma: Desviación estándar del desenfoque de insertion.

    Returns:
        Diccionario ``{"deletion": ..., "insertion": ...}`` con la salida de
        :func:`faithfulness_scores` para cada métrica.
    """
    return {
        "deletion": faithfulness_scores(model, x, heatmap, class_idx, rng, "deletion",
                                        gray_like(x), n_random, steps),
        "insertion": faithfulness_scores(model, x, heatmap, class_idx, rng, "insertion",
                                         blurred_like(x, blur_sigma), n_random, steps),
    }


def reset_module(module):
    """Reinicializar los parámetros de un módulo con la inicialización de PyTorch.

    Llama a ``reset_parameters`` en cada submódulo que lo define (capas
    convolucionales, lineales y de normalización por lotes).
    """
    for submodule in module.modules():
        if hasattr(submodule, "reset_parameters"):
            submodule.reset_parameters()


def cascading_randomization(model, images, classes, seed=42, layer_name="layer4"):
    """Aplicar la prueba de aleatorización de parámetros en cascada.

    Copia el modelo y reinicializa de forma acumulativa los módulos de
    :data:`CASCADE_STAGES`. En cada etapa calcula el mapa de Grad-CAM de
    ``layer_name`` (forma ``(h, w)``, con ReLU y sin ampliar) para la clase
    indicada de cada imagen y su correlación de Spearman con el mapa del
    modelo original.

    Args:
        model: Clasificador original, no se modifica.
        images: Lista de imágenes normalizadas de forma ``(1, 3, H, W)``.
        classes: Lista con la clase a explicar en cada imagen.
        seed: Semilla de PyTorch para la reinicialización.
        layer_name: Nombre de la capa explicada.

    Returns:
        Tupla ``(rho, maps)``. ``rho`` tiene forma
        ``(len(CASCADE_STAGES), len(images))`` y ``maps`` es una lista por
        etapa con los mapas de cada imagen.
    """
    torch.manual_seed(seed)
    original = [grad_cam(model, getattr(model, layer_name), x, c).cam
                for x, c in zip(images, classes)]
    randomized = copy.deepcopy(model).eval()
    rho = np.zeros((len(CASCADE_STAGES), len(images)))
    maps = []
    for s, stage in enumerate(CASCADE_STAGES):
        reset_module(getattr(randomized, stage))
        stage_maps = []
        for i, (x, c) in enumerate(zip(images, classes)):
            cam = grad_cam(randomized, getattr(randomized, layer_name), x, c).cam
            rho[s, i] = spearmanr(original[i].ravel(), cam.ravel()).statistic
            stage_maps.append(cam)
        maps.append(stage_maps)
    return rho, maps
