# Laboratorio 10: Explicabilidad e interpretabilidad (XAI)

Se explican las decisiones de dos modelos con técnicas de XAI y se pone a prueba si esas explicaciones merecen confianza.

- **Visión** (`notebooks/01_vision.ipynb`). Grad-CAM implementado desde cero sobre un ResNet-18 preentrenado en ImageNet, verificación numérica de su relación con CAM y validación de los mapas con deletion, insertion y la prueba de aleatorización de parámetros en cascada (Tasks 1 y 4).

- **Datos tabulares** (`notebooks/02_tabular.ipynb`). Un clasificador de gradient boosting de riesgo de crédito entrenado con German Credit, explicado con valores SHAP (TreeSHAP, KernelSHAP y la fórmula exacta de Shapley), LIME implementado desde cero y explicaciones contrafactuales (Tasks 2 y 3).

## Estructura

```
├── data/raw/images/    Imágenes de prueba y sus fuentes (SOURCES.md)
├── notebooks/          Notebooks del laboratorio
├── src/
│   ├── vision/         Preprocesado, Grad-CAM, CAM y métricas de fidelidad
│   └── tabular/        Datos, Shapley exacto, LIME y contrafactuales
└── requirements.txt
```

## Cómo ejecutar

Los notebooks están pensados para Google Colab.

1. Abrir el notebook en Colab desde GitHub (`Archivo > Abrir notebook > GitHub`) con la URL de este repositorio.
2. Ejecutar todas las celdas (`Entorno de ejecución > Ejecutar todas`). La primera celda clona el repositorio para importar los módulos de `src/` y, en el notebook tabular, instala `shap`.

Para ejecutarlos de forma local, instalar las dependencias y abrir los notebooks desde la carpeta `notebooks/`.

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```
