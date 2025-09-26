# 2_graficos_EDA.py
import pandas as pd
from plotnine import ggplot, aes, geom_point, labs, theme, element_line, scale_color_manual

# Cargar CSV filtrado
df = pd.read_csv('../datos/music_filtrado.csv')

# Crear gráfico
plot = (
    ggplot(df, aes(x='acousticness', y='loudness', color='music_genre'))
    + geom_point(shape='o', size=4, fill='white', stroke=1.5, alpha=0.7)
    + scale_color_manual(values={
        'Electronic': 'lightblue',
        'Pop': 'green',
        'Rock': 'orange',
        'Jazz': 'purple'
        # añade más géneros si los hay
    })
    + labs(title='Loudness vs Acousticness por género musical',
           x='Acousticness',
           y='Loudness')
    + theme(
        axis_line_x=element_line(color='black', size=1),
        axis_line_y=element_line(color='black', size=1)
    )
)

# Guardar gráfico
plot.save('../resultados/scatter_genero.png', width=6, height=4, dpi=100)

# Mostrar referencia
print(plot)
