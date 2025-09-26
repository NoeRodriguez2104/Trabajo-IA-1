
---
## ** después de clonar el repositorio::**


1. Instalar Python (si no lo tienen).

2. Crear su propio entorno virtual 

3. Activar el entorno virtual.
4. Ejecutar:
pip install -r requirements.txt
5. pip install ipykernel
python -m ipykernel install --user --name=venv_musica --display-name "Python (venv_musica)"


Ejecutar:**

## **1️⃣ Crear entorno de trabajo en Python**

1. **Instalar Python**:

   * Asegúrate de tener Python ≥ 3.9. Puedes descargarlo desde [python.org](https://www.python.org/downloads/).
   * Marca la opción **“Add Python to PATH”** durante la instalación.

2. **Crear carpeta de proyecto**:

   ```bash
   mkdir ProyectoMusica
   cd ProyectoMusica
   ```

3. **Crear entorno virtual** (para aislar librerías):

   ```bash y windows
   python -m venv env_musica
   ```
   **SI YA LO TIENES CREADO TOCA ELEGIR**
   CTRL + shift + p  luego python selecionar interprete eleiges tu entorno

4. **Activar entorno virtual**:

   * Windows (PowerShell):

     ```powershell
     .\env_musica\Scripts\Activate.ps1
     ```
   * Windows (cmd):

     ```cmd
     .\env_musica\Scripts\activate.bat
     ```
   * Linux/Mac:

     ```bash
     source env_musica/bin/activate
     ```
    
5. **Actualizar pip**:

   ```bash
   python -m pip install --upgrade pip
   ```

---

## **2️⃣ Instalar librerías necesarias**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotnine jupyterlab
```

* `pandas`: manejo de CSV y DataFrames
* `numpy`: operaciones numéricas
* `matplotlib` y `seaborn`: gráficos básicos
* `plotnine`: gráficos estilo ggplot (estilo R)
* `scikit-learn`: preprocesamiento y modelos de ML
* `jupyterlab`: entorno de notebook

---


## **3️⃣ Crear estructura de carpetas**

Dentro de `Trabajo-IA-1`:

```text
Trabajo-IA-1/
├─ datos/                # datasets originales y filtrados
├─ resultados/           # gráficos generados
├─ scripts/              # scripts de Python
└─ notebooks/            # notebooks para análisis interactivo
```

---

## **4️⃣ Cargar el dataset original**

1. Coloca tu CSV (por ejemplo `music_genre.csv`) en la carpeta datos.
2. Crea el Notebook proyecto_musica.ipynb.
3. Primera celda: leer CSV y manejar problemas comunes de formato:

```python
import pandas as pd
import os

# Asegurar carpetas para filtrados y resultados
os.makedirs("datos", exist_ok=True)
os.makedirs("resultados", exist_ok=True)

# Cargar CSV
df = pd.read_csv("datos/music_genre.csv")

# Primeras filas
df.head()
```

---

## **5️⃣ Exploración inicial de datos (EDA)**

```python
# Estadísticas descriptivas de variables clave
print(df[['acousticness', 'loudness', 'music_genre']].describe())

# Valores nulos
print(df[['acousticness', 'loudness', 'music_genre']].isnull().sum())

# Guardar versión filtrada para análisis posterior
df[['acousticness', 'loudness', 'music_genre']].to_csv("datos/music_filtrado.csv", index=False)
```

---

## **6️⃣ Visualización de datos con plotnine**

```python
from plotnine import ggplot, aes, geom_point, labs, theme, element_line, scale_color_manual

# Leer CSV filtrado
df_filtrado = pd.read_csv("datos/music_filtrado.csv")

# Scatter plot: Loudness vs Acousticness por género
plot = (
    ggplot(df_filtrado, aes(x='acousticness', y='loudness', color='music_genre'))
    + geom_point(size=3, alpha=0.7)
    + labs(title='Loudness vs Acousticness por género musical',
           x='Acousticness', y='Loudness')
    + theme(
        axis_line_x=element_line(color='black', size=1),
        axis_line_y=element_line(color='black', size=1)
    )
)

# Mostrar gráfico en el notebook
plot
```

---

## **7️⃣ Preprocesamiento de datos**

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Variables independientes y objetivo
X = df_filtrado[['acousticness','loudness']]
y = df_filtrado['music_genre']

# Codificación de variable categórica
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Normalización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.25, random_state=42
)
```

---

## **8️⃣ Entrenamiento de modelo**

```python
from sklearn.ensemble import RandomForestClassifier

# Crear y entrenar Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```

---

## **9️⃣ Evaluación del modelo**

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Predicciones
y_pred = model.predict(X_test)

# Métricas
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues')
plt.title('Matriz de Confusión')
plt.tight_layout()
plt.show()
```

---

## **🔟 Conclusiones y comunicación**

* Interpretar: ¿Qué géneros son más fáciles de predecir?
* Graficar feature importance:

```python
import matplotlib.pyplot as plt

importances = model.feature_importances_
features = X.columns

plt.bar(features, importances)
plt.title("Importancia de las variables")
plt.xlabel("Variable")
plt.ylabel("Importancia")
plt.show()
```

---
