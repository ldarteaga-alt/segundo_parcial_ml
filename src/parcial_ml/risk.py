import numpy as np


def riesgo_empirico(y_real, y_predicho):
    """
    Calcula el Riesgo Empírico usando la pérdida cuadrática.
    
    Fórmula:
        R(h) = (1/n) * suma( (y_real - y_predicho)^2 )
    
    Recibe:
    -------
    y_real : los valores reales (lo que realmente pasó)
    y_predicho : los valores que predijo el modelo
    
    Devuelve:
    --------
    Un número: el riesgo (mientras más pequeño, mejor)
    """
    # 1. Cuántos datos hay
    n = len(y_real)
    
    # 2. Error de cada dato: real - predicho
    error = y_real - y_predicho
    
    # 3. Elevamos cada error al cuadrado
    error_cuadrado = error ** 2
    
    # 4. Sumamos todos los errores cuadrados
    suma_errores = np.sum(error_cuadrado)
    
    # 5. Dividimos entre n para obtener el promedio
    riesgo = suma_errores / n
    
    return float(riesgo)