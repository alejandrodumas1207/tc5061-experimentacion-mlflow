# TC5061 · Experimentación con MLflow (Ames Housing)

Script de entrenamiento parametrizado que predice el precio de venta de viviendas en Ames, Iowa, con una regresión Ridge. Cada corrida registra en MLflow Tracking sus parámetros, métricas, una gráfica de precio real contra predicho y el modelo entrenado.

## Estructura

```
├── train.py           # Script de entrenamiento
├── notebooks/         # Notebook de exploración y modelo base (semana 3)
├── requirements.txt   # Dependencias con versiones fijas
└── README.md
```

## Instalación

Probado con Python 3.12.3 en Ubuntu 24.04 (WSL).

```bash
git clone https://github.com/alejandrodumas1207/tc5061-experimentacion-mlflow.git
cd tc5061-experimentacion-mlflow
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

El dataset se descarga automáticamente desde OpenML (`house_prices`) en la primera ejecución, así que esa primera corrida necesita conexión a internet.

## Ejecución

Con los valores por defecto, que reproducen el modelo base del notebook:

```bash
python train.py
```

Con la configuración de mejor desempeño:

```bash
python train.py --alpha 10 --num-strategy median --max-area 4000 --test-size 0.2 --random-state 42
```

| Argumento | Default | Descripción |
|---|---|---|
| `--alpha` | `1.0` | Fuerza de la regularización de Ridge |
| `--num-strategy` | `median` | Imputación de variables numéricas (`median` o `mean`) |
| `--max-area` | `4000` | Umbral de `GrLivArea` para eliminar outliers |
| `--test-size` | `0.2` | Proporción del conjunto de prueba |
| `--random-state` | `42` | Semilla de la división entrenamiento/prueba |

`python train.py --help` muestra la ayuda completa.

## Corridas del experimento

```bash
python train.py
python train.py --alpha 0.1
python train.py --alpha 10
python train.py --alpha 100
python train.py --num-strategy mean
python train.py --max-area 3000
python train.py
```

## Resultados en MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abrir `http://127.0.0.1:5000` y entrar al experimento `TC5061_Experimentacion_AmesHousing`. Cada corrida registra:

- Parámetros: los cinco argumentos de la tabla.
- Métricas: `mae`, `rmse` y `r2` en dólares sobre el conjunto de prueba, y `r2_train_log` sobre entrenamiento en escala logarítmica.
- Artefactos: `pred_vs_real.png` y el modelo serializado en formato skops.

## Reproducibilidad

La semilla `--random-state` controla la división de los datos. Dos ejecuciones con los mismos argumentos producen las mismas métricas. El notebook de `notebooks/` requiere además `seaborn` e `ipykernel`.

## Uso de IA

El README, la revisión del código y la integración con MLflow se hicieron con apoyo de Claude (Anthropic).
