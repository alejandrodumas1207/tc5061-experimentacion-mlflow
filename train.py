 #Librerías
import argparse
from html import parser
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


if __name__ == "__main__":
    args = parse_args()
    print(args)
