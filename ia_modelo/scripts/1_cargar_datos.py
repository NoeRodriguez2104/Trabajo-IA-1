# 1_cargar_datos.py
import pandas as pd

# Cargar dataset
df = pd.read_csv('../datos/music_genre.csv')

# Ver primeras filas
print("Primeras filas del dataset:")
print(df.head())

# Estadísticas básicas
print("\nEstadísticas del dataset:")
print(df[['acousticness','loudness','music_genre']].describe())

# Revisar si hay valores nulos
print("\nValores nulos por columna:")
print(df[['acousticness','loudness','music_genre']].isnull().sum())

# Guardar DataFrame filtrado con solo las columnas que nos interesan
df[['acousticness','loudness','music_genre']].to_csv('../datos/music_filtrado.csv', index=False)
