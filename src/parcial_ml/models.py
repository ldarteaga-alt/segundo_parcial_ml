from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from .risk import riesgo_empirico


def entrenar_modelo_polinomico(x_entrenamiento, y_entrenamiento,
                                x_validacion, y_validacion, grado):
    """
    Entrena un modelo polinómico del grado indicado y calcula
    el riesgo sobre entrenamiento y validación.
    
    Recibe:
    -------
    x_entrenamiento, y_entrenamiento : datos para que el modelo aprenda
    x_validacion, y_validacion : datos nuevos para evaluar
    grado : qué tan complejo será el polinomio (1, 2, ..., 15)
    
    Devuelve:
    --------
    modelo : el modelo ya entrenado
    transformador : convierte x en potencias [1, x, x², ...]
    riesgo_entrenamiento : qué tan bien predice los datos que vio
    riesgo_validacion : qué tan bien predice datos nuevos
    """
    
    # PASO 1: Creamos el transformador polinómico.
    # Si grado=2, convierte x en [1, x, x²].
    transformador = PolynomialFeatures(degree=grado, include_bias=True)
    
    # PASO 2: Transformamos los datos de entrenamiento.
    x_entrenamiento_transformado = transformador.fit_transform(x_entrenamiento)
    
    # PASO 3: Transformamos los datos de validación.
    # OJO: solo transform, NO fit_transform, para no aprender de validación.
    x_validacion_transformado = transformador.transform(x_validacion)
    
    # PASO 4: Creamos el modelo de regresión lineal (vacío).
    modelo = LinearRegression()
    
    # PASO 5: Entrenamos el modelo con los datos transformados.
    modelo.fit(x_entrenamiento_transformado, y_entrenamiento)
    
    # PASO 6: Predecimos sobre entrenamiento.
    y_entrenamiento_predicho = modelo.predict(x_entrenamiento_transformado)
    
    # PASO 7: Predecimos sobre validación.
    y_validacion_predicho = modelo.predict(x_validacion_transformado)
    
    # PASO 8: Calculamos el riesgo de entrenamiento.
    riesgo_entrenamiento = riesgo_empirico(y_entrenamiento, y_entrenamiento_predicho)
    
    # PASO 9: Calculamos el riesgo de validación.
    riesgo_validacion = riesgo_empirico(y_validacion, y_validacion_predicho)
    
    # PASO 10: Devolvemos todo
    return modelo, transformador, riesgo_entrenamiento, riesgo_validacion
