#Librerías
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

def parse_args():
    """Lee los hiperparámetros desde la línea de comandos."""
    parser = argparse.ArgumentParser(description="Entrena un modelo Ridge para predecir precios de vivienda en Ames")
    parser.add_argument('--alpha', type=float, default=1.0, help='Regularizacion de Ridge')
    parser.add_argument('--num-strategy', type=str, default='median', choices=["median", "mean"], help='Estrategia de imputacion para variables numericas')
    parser.add_argument('--max-area', type=int, default=4000, help='umbral de GrLivArea para eliminar outliers')
    parser.add_argument('--test-size', type=float, default=0.2, help='Tamaño del conjunto de prueba')
    parser.add_argument('--random-state', type=int, default=42, help='Semilla para la reproducibilidad')
    return parser.parse_args()


def load_data():
    """Descarga el dataset Ames Housing desde OpenML."""
    data = fetch_openml("house_prices",as_frame=True)
    df = data.frame
    return df

def clean_data(df, max_area):
    """Limpia el dataset y elimina outliers según el umbral de área."""
    #Eliminamos la columna de ID
    df = df.drop(columns="Id")
    #MSSubClass es un código de tipo de vivienda y no una cantidad por lo que se convierte a texto
    df["MSSubClass"] = df["MSSubClass"].astype(str)
    #Columnas donde NA significa que la casa no tiene la característica
    sin_caracteristica = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
                            "GarageType", "GarageFinish", "GarageQual", "GarageCond",
                            "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
                            "MasVnrType"]
    df[sin_caracteristica] = df[sin_caracteristica].fillna("Ninguno")
    #Sin revestimiento de mampostería, el área es cero
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    #GarageYrBlt repite la información de YearBuilt
    df = df.drop(columns="GarageYrBlt")
    #Eliminamos ouliers de GrLivArea según el umbral definido
    df = df[df["GrLivArea"] <= max_area]
    return df

def split_data(df, test_size, random_state):
    """Separa el dataset en conjunto de entrenamiento y prueba."""
    #Separamos predictoras (X) y objetivo (y)
    #El objetivo se transforma con log1p
    X = df.drop(columns="SalePrice")
    y = np.log1p(df["SalePrice"])
    #División en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def build_model(X, alpha, num_strategy):
    """Construye un pipeline de preprocesamiento y modelo Ridge.""" 
    #Separamos columnas numéricas y categóricas para darles tratamiento distinto
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    #Imputar valores con la estrategia seleccionada y escalar las columnas numéricas
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=num_strategy)),
        ("scaler", StandardScaler())
    ])

    #Para las columnas categóricas: imputar con el valor más frecuente y convertir a columnas 0/1
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    #Aplicamos cada tratamiento a su grupo de columnas
    preprocesador = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    #Modelo base
    modelo = Pipeline([
        ("preprocesador", preprocesador),
        ("ridge", Ridge(alpha=alpha))
    ])

    return modelo

def evaluate_model(modelo, X_test, y_test):
    """Evalúa el modelo en el conjunto de prueba y devuelve métricas."""
    #Predicciones en escala logarítmica, convertidas de vuelta a dólares con expm1
    y_real = np.expm1(y_test)
    y_pred = np.expm1(modelo.predict(X_test))
    metricas = {
        "mae": mean_absolute_error(y_real, y_pred),
        "rmse": root_mean_squared_error(y_real, y_pred),
        "r2": r2_score(y_real, y_pred)
    }
    return metricas, y_real, y_pred

def plot_predictions(y_real, y_pred, ruta="pred_vs_real.png"):
    """Genera un gráfico de dispersión de predicciones vs valores reales."""
    plt.scatter(y_real, y_pred, alpha=0.5)
    plt.plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--', lw=2)
    plt.xlabel("Precio real en dolares")
    plt.ylabel("Precio predicho en dolares")
    plt.title("Predicciones vs Valores Reales")
    plt.savefig(ruta)
    plt.close()
    return ruta


def main():
    """Entrena el modelo y registra el run en MLflow."""
    args = parse_args()
    # Guardamos el registro de experimentos en una base SQLite local
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("TC5061_Experimentacion_AmesHousing")
    df =load_data()
    df = clean_data(df, args.max_area)
    X_train, X_test, y_train, y_test = split_data(df, args.test_size, args.random_state)
    modelo = build_model(X_train, args.alpha, args.num_strategy)
    #Run del modelo
    with mlflow.start_run(run_name=f"ridge_a{args.alpha}_{args.num_strategy}_area{args.max_area}"):
        mlflow.log_params(vars(args))

        modelo.fit(X_train, y_train)

        metricas, y_real, y_pred = evaluate_model(modelo, X_test, y_test)
        mlflow.log_metrics(metricas)
        mlflow.log_metric("r2_train_log", modelo.score(X_train, y_train))

        ruta = plot_predictions(y_real, y_pred)
        mlflow.log_artifact(ruta)

        mlflow.sklearn.log_model(modelo, name="modelo", skops_trusted_types=["numpy.dtype"])
    print(metricas)


if __name__ == "__main__":
    main()