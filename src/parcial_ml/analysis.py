import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .risk import riesgo_empirico

from .data import cargar_datos
from .models import entrenar_modelo_polinomico


def experimento_para_un_dataset(carpeta_datos, archivo_entrenamiento,
                                 archivo_validacion, grados, carpeta_resultados):
    """
    Ejecuta el experimento para UN conjunto de entrenamiento
    sobre varios grados polinómicos.
    
    Recibe:
    -------
    carpeta_datos : dónde están los CSV
    archivo_entrenamiento : qué CSV usar para entrenar
    archivo_validacion : qué CSV usar para validar
    grados : lista de grados a probar [1, 2, 3, ..., 15]
    carpeta_resultados : dónde guardar las gráficas
    
    Devuelve:
    --------
    Una tabla con los resultados de todos los grados
    """
    
    # 1. Cargamos los datos
    x_entrenamiento, y_entrenamiento = cargar_datos(carpeta_datos, archivo_entrenamiento)
    x_validacion, y_validacion = cargar_datos(carpeta_datos, archivo_validacion)
    
    # Aquí guardaremos los resultados de cada grado
    resultados = []
    
    # 2. Recorremos cada grado de la lista
    for grado in grados:
        
        # Entrenamos el modelo para ese grado
        modelo, transformador, riesgo_entrenamiento, riesgo_validacion = entrenar_modelo_polinomico(
            x_entrenamiento, y_entrenamiento,
            x_validacion, y_validacion,
            grado
        )
        
        # Guardamos los resultados en la lista
        resultados.append({
            'Dataset': archivo_entrenamiento.replace('.csv', ''),
            'Degree': grado,
            'Train_Risk': riesgo_entrenamiento,
            'Val_Risk': riesgo_validacion
        })
        
    
    # 4. Devolvemos la tabla
    return pd.DataFrame(resultados)




def analizar_sensibilidad(carpeta_datos, grados, carpeta_resultados, m=300):
    """
    Ejercicio 2: para cada grado, entrena 3 modelos (con A, B y C),
    los evalúa sobre una malla de m puntos en [-2.5, 2.5],
    y calcula el promedio y la varianza.
    """
    # 1. Cargamos las 3 muestras
    muestras = {}
    for nombre in ['train_A', 'train_B', 'train_C']:
        muestras[nombre] = cargar_datos(carpeta_datos, nombre + '.csv')
    
    # 2. Malla común: m puntos entre -2.5 y 2.5
    malla = np.linspace(-2.5, 2.5, m).reshape(-1, 1)
    
    resultados = {}
    colores = ['blue', 'green', 'orange']
    
    # 3. Para cada grado
    for grado in grados:
        predicciones = []
        
        # Entrenamos un modelo por cada muestra
        for nombre in muestras:
            x, y = muestras[nombre]
            modelo, trans, _, _ = entrenar_modelo_polinomico(x, y, x, y, grado)
            predicciones.append(modelo.predict(trans.transform(malla)))
        
        predicciones = np.vstack(predicciones)
        
        # Promedio y varianza en cada punto de la malla
        promedio = predicciones.mean(axis=0)
        varianza_punto = predicciones.var(axis=0)
        
        # Varianza promedio (un solo número por grado)
        V_d = varianza_punto.mean()
        
        resultados[grado] = {
            'malla': malla.flatten(),
            'predicciones': predicciones,
            'promedio': promedio,
            'varianza': varianza_punto,
            'V_d': float(V_d),
        }
        
        # Gráfica del (a): solo las 3 hipótesis
        plt.figure(figsize=(9, 5))
        for i, nombre in enumerate(muestras):
            plt.plot(malla, predicciones[i], color=colores[i],
                     linewidth=2.5, label=f'$\\hat{{h}}_{{{nombre[-1]},{grado}}}$')
        
        plt.title(f'Ejercicio 2(a): Tres hipótesis aprendidas — grado {grado}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(carpeta_resultados, f'ej2a_grado_{grado}.png'), dpi=100)
        plt.show()
    
    return resultados



def riesgo_validacion_promedio(carpeta_datos, grados):
    """
    Para cada grado, entrena 3 modelos (con A, B y C) y calcula
    el riesgo de validación promedio sobre validation.csv.
    """
    # Cargamos train_A, train_B, train_C y validation
    muestras = {}
    for nombre in ['train_A', 'train_B', 'train_C']:
        muestras[nombre] = cargar_datos(carpeta_datos, nombre + '.csv')
    
    x_val, y_val = cargar_datos(carpeta_datos, 'validation.csv')
    
    resultados = {}
    
    for grado in grados:
        riesgos = []
        
        # Entrenamos un modelo por muestra y evaluamos en validación
        for nombre in muestras:
            x, y = muestras[nombre]
            modelo, trans, _, _ = entrenar_modelo_polinomico(x, y, x, y, grado)
            
            # Predecimos sobre validación
            y_val_pred = modelo.predict(trans.transform(x_val))
            
            # Calculamos el riesgo de validación
            r_val = riesgo_empirico(y_val, y_val_pred)
            riesgos.append(r_val)
        
        # Promedio de los 3 riesgos
        resultados[grado] = float(np.mean(riesgos))
    
    return resultados