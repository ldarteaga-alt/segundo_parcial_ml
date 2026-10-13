# Segundo Parcial - Introducción al Machine Learning

**Universidad El Bosque**  
**Programa de Matemáticas y Estadística**  
**Semestre 2026-2**

## Descripción

Proyecto modular que resuelve el segundo examen parcial de Machine Learning. Incluye 3 ejercicios:

1. **Complejidad y generalización:** análisis del riesgo de entrenamiento y validación para grados polinómicos 1 al 15 usando `train_A.csv`.
2. **Sensibilidad a la muestra:** comparación de las hipótesis aprendidas con `train_A`, `train_B` y `train_C` para los grados 2, 6 y 15, incluyendo promedio y varianza.
3. **Selección del modelo:** cálculo del riesgo promedio de validación y selección del mejor grado.

## Estructura del proyecto

segundo_parcial/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── data/
│ ├── train_A.csv
│ ├── train_B.csv
│ ├── train_C.csv
│ └── validation.csv
├── src/
│ └── parcial_ml/
│ ├── init.py
│ ├── data.py
│ ├── models.py
│ ├── risk.py
│ └── analysis.py
├── notebooks/
│ └── parcial.ipynb
└── resultados/
├── ej1b_grafica.png
├── ej2_grado_2.png
├── ej2_grado_6.png
└── ej2_grado_15.png


## Instalación

1. Clonar el repositorio:
   ```bash
   git clone <URL_DEL_REPO>
   cd segundo_parcial

2. Crear un entorno virtual e instalar dependencias:
python -m venv venv
source venv/bin/activate       # En Windows: venv\Scripts\activate
pip install -r requirements.txt


---

## Paso 2: Verificar `.gitignore`

# Entorno virtual
venv/
.venv/

# Cache de Python
__pycache__/
*.pyc
*.pyo

# Cache de Jupyter
.ipynb_checkpoints/

# Sistema operativo
.DS_Store
Thumbs.db

# Editor
.vscode/
.idea/

## Paso 3: Hacer el primer commit 
git init
git add .
git status
