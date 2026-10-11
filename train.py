 #Librerías
import argparse
from html import parser
from pyexpat import model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

def parse_args():
    parser = argparse.ArgumentParser(description="Entrena un modelo Ridge para predecir precios de vivienda en Ames")
    parser.add_argument('--alpha', type=float, default=1.0, help='Regularizacion de Ridge')
    parser.add_argument('--num-strategy', type=str, default='median', choices=["median", "mean"], help='Estrategia de imputacion para variables numericas')
    parser.add_argument('--max-area', type=int, default=4000, help='umbral de GrLivArea para eliminar outliers')
    parser.add_argument('--test-size', type=float, default=0.2, help='Tamaño del conjunto de prueba')
    parser.add_argument('--random-state', type=int, default=42, help='Semilla para la reproducibilidad')
    return parser.parse_args()


def load_data():
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
    #Eliminamos las casas con más de 4,000 pies² habitables por recomendación de la documentación.
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
    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(exclude="number").columns

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



if __name__ == "__main__":
    args = parse_args()
    df = load_data()
    print(df.shape)
    df = clean_data(df, args.max_area)
    print(df.shape)

    X_train, X_test, y_train, y_test = split_data(df, args.test_size, args.random_state)
    print(X_train.shape, X_test.shape)
    modelo = build_model(X_train, args.alpha, args.num_strategy)
    modelo.fit(X_train, y_train)
    print("R2 prueba (log):", round(modelo.score(X_test, y_test), 3))