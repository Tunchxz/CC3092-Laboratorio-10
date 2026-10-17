"""Carga del ResNet-18, preprocesado de imágenes y visualización de mapas."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224

_resize_crop = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(IMAGE_SIZE)])
_to_tensor = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
)


def load_resnet18():
    """Cargar ResNet-18 con pesos de ImageNet en modo de evaluación.

    Returns:
        Tupla ``(model, categories)`` con el modelo y la lista de los
        1 000 nombres de clase de ImageNet.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).eval()
    return model, list(weights.meta["categories"])


def load_image(path):
    """Leer y preprocesar una imagen para el ResNet-18.

    Aplica redimensión a 256 píxeles en el lado corto, recorte central de
    224 por 224 y normalización por canal con la media y la desviación
    estándar de ImageNet.

    Args:
        path: Ruta del archivo de imagen.

    Returns:
        Tupla ``(x, rgb)`` con el tensor normalizado de forma
        ``(1, 3, 224, 224)`` y la imagen recortada como arreglo
        ``uint8`` de forma ``(224, 224, 3)``.
    """
    cropped = _resize_crop(Image.open(path).convert("RGB"))
    return _to_tensor(cropped).unsqueeze(0), np.asarray(cropped)


def normalize_pixels(pixels):
    """Normalizar un tensor de píxeles en [0, 1] con las estadísticas de ImageNet.

    Args:
        pixels: Tensor de forma ``(..., 3, H, W)``.

    Returns:
        Tensor normalizado con la misma forma.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (pixels - mean) / std


def gray_like(x, level=0.5):
    """Construir una imagen gris uniforme ya normalizada.

    Args:
        x: Tensor de referencia de forma ``(1, 3, H, W)``.
        level: Intensidad del gris en el espacio de píxeles [0, 1].

    Returns:
        Tensor normalizado con la forma de ``x``.
    """
    return normalize_pixels(torch.full_like(x, level))


@torch.no_grad()
def predict_proba(model, x):
    """Calcular las probabilidades softmax del modelo.

    Args:
        model: Clasificador de PyTorch.
        x: Lote de imágenes normalizadas de forma ``(B, 3, H, W)``.

    Returns:
        Tensor de forma ``(B, n_clases)`` con las probabilidades.
    """
    return torch.softmax(model(x), dim=1)


def top_k(model, x, categories, k=5):
    """Obtener las ``k`` clases de mayor probabilidad de una imagen.

    Args:
        model: Clasificador de PyTorch.
        x: Imagen normalizada de forma ``(1, 3, H, W)``.
        categories: Nombres de las clases.
        k: Número de clases a devolver.

    Returns:
        Lista de tuplas ``(indice, nombre, probabilidad)`` ordenada de mayor
        a menor probabilidad.
    """
    probs = predict_proba(model, x)[0]
    values, indices = probs.topk(k)
    return [(int(i), categories[int(i)], float(p)) for p, i in zip(values, indices)]


def show_overlay(ax, rgb, heatmap, title=None, alpha=0.5, box=None):
    """Dibujar un mapa de calor superpuesto a una imagen.

    Args:
        ax: Eje de Matplotlib.
        rgb: Imagen ``uint8`` de forma ``(H, W, 3)``.
        heatmap: Mapa en [0, 1] de forma ``(H, W)``.
        title: Título del eje.
        alpha: Opacidad del mapa.
        box: Recuadro opcional ``(x0, y0, x1, y1)`` en píxeles.
    """
    ax.imshow(rgb)
    ax.imshow(heatmap, cmap="jet", alpha=alpha, vmin=0, vmax=1)
    if box is not None:
        draw_box(ax, box)
    if title:
        ax.set_title(title, fontsize=9)
    ax.axis("off")


def draw_box(ax, box, color="white"):
    """Dibujar un recuadro ``(x0, y0, x1, y1)`` sobre un eje de Matplotlib."""
    x0, y0, x1, y1 = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color=color, lw=2))
