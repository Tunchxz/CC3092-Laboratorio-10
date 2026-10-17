"""Grad-CAM y CAM implementados con hooks de PyTorch."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class GradCAMResult:
    """Resultado de Grad-CAM para una imagen y una clase.

    Attributes:
        activations: Activaciones ``A`` de la capa, forma ``(K, h, w)``.
        gradients: Gradientes ``dy^c/dA``, forma ``(K, h, w)``.
        alpha: Pesos ``a_k^c`` por canal, forma ``(K,)``.
        raw: Combinación lineal ``sum_k a_k^c A^k`` sin ReLU, forma ``(h, w)``.
        cam: Mapa ``ReLU(raw)`` sin normalizar, forma ``(h, w)``.
        heatmap: Mapa ampliado a la entrada con interpolación bilineal y
            normalizado con mínimo y máximo, forma ``(H, W)``.
        logit: Logit ``y^c`` de la clase.
    """

    activations: np.ndarray
    gradients: np.ndarray
    alpha: np.ndarray
    raw: np.ndarray
    cam: np.ndarray
    heatmap: np.ndarray
    logit: float


def min_max(values):
    """Normalizar un arreglo al intervalo [0, 1] con su mínimo y su máximo.

    Devuelve ceros si el arreglo es constante.
    """
    values = np.asarray(values, dtype=np.float64)
    span = values.max() - values.min()
    if span < 1e-12:
        return np.zeros_like(values)
    return (values - values.min()) / span


def grad_cam(model, layer, x, class_idx):
    """Calcular Grad-CAM de una clase sobre una capa convolucional.

    Registra un hook de avance que guarda las activaciones de ``layer`` y un
    hook de retroceso que guarda el gradiente del logit respecto a ellas. El
    gradiente se retropropaga desde el logit de la clase, antes del softmax.
    Los hooks se eliminan al terminar.

    Args:
        model: Clasificador de PyTorch en modo de evaluación.
        layer: Módulo cuya salida se explica (por ejemplo ``model.layer4``).
        x: Imagen normalizada de forma ``(1, 3, H, W)``.
        class_idx: Índice de la clase a explicar.

    Returns:
        Un :class:`GradCAMResult`.
    """
    store = {}

    def forward_hook(module, inputs, output):
        store["A"] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        store["dA"] = grad_output[0].detach()

    handles = [
        layer.register_forward_hook(forward_hook),
        layer.register_full_backward_hook(backward_hook),
    ]
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits[0, class_idx].backward()
    finally:
        for handle in handles:
            handle.remove()

    A = store["A"][0]
    dA = store["dA"][0]
    alpha = dA.mean(dim=(1, 2))
    raw = torch.einsum("k,kij->ij", alpha, A)
    cam = F.relu(raw)
    upsampled = F.interpolate(
        cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False
    )[0, 0]
    return GradCAMResult(
        activations=A.numpy(),
        gradients=dA.numpy(),
        alpha=alpha.numpy(),
        raw=raw.numpy(),
        cam=cam.numpy(),
        heatmap=min_max(upsampled.numpy()),
        logit=float(logits[0, class_idx].detach()),
    )


def cam_from_fc(fc, activations, class_idx):
    """Calcular CAM con los pesos de la capa lineal final, sin gradientes.

    Args:
        fc: Capa ``torch.nn.Linear`` final del modelo.
        activations: Activaciones ``A`` de forma ``(K, h, w)``.
        class_idx: Índice de la clase.

    Returns:
        Tupla ``(M, bias)`` con el mapa ``M_ij = sum_k w_k^c A_ij^k`` de forma
        ``(h, w)`` y el sesgo ``b_c``.
    """
    w = fc.weight.detach()[class_idx].numpy()
    bias = float(fc.bias.detach()[class_idx])
    return np.einsum("k,kij->ij", w, activations), bias


def top_fraction_mask(heatmap, fraction=0.2):
    """Marcar el conjunto de píxeles con la fracción de mayor valor del mapa.

    Args:
        heatmap: Mapa de forma ``(H, W)``.
        fraction: Fracción de píxeles a conservar.

    Returns:
        Máscara booleana de forma ``(H, W)``.
    """
    flat = heatmap.ravel()
    n_top = int(round(fraction * flat.size))
    mask = np.zeros(flat.size, dtype=bool)
    mask[np.argsort(-flat, kind="stable")[:n_top]] = True
    return mask.reshape(heatmap.shape)


def iou(mask_a, mask_b):
    """Calcular la intersección sobre la unión de dos máscaras booleanas."""
    return float((mask_a & mask_b).sum() / (mask_a | mask_b).sum())
