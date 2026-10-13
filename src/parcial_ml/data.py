import os
import pandas as pd


def cargar_datos(carpeta_datos, nombre_archivo):
    """
    Carga un archivo CSV y devuelve las columnas x e y.
    
    Recibe:
    -------
    carpeta_datos : la carpeta donde están los CSV 
    nombre_archivo : el nombre del archivo 
    
    Devuelve:
    --------
    x : los valores de la columna 'x' (lo que usamos para predecir)
    y : los valores de la columna 'y' (lo que queremos predecir)
    """
    # 1. Unimos la carpeta y el nombre para obtener la ruta completa
    ruta_completa = os.path.join(carpeta_datos, nombre_archivo)
    
    # 2. Leemos el CSV con pandas
    tabla = pd.read_csv(ruta_completa)
    
    # 3. Verificamos que tenga las columnas correctas
    if 'x' not in tabla.columns or 'y' not in tabla.columns:
        raise ValueError(f"El archivo {nombre_archivo} no tiene las columnas 'x' y 'y'.")
    
    # 4. Extraemos las columnas
    x = tabla[['x']].values   # Doble corchete: devuelve una matriz 2D
    y = tabla['y'].values     # Corchete simple: devuelve un vector 1D
    
    return x, y