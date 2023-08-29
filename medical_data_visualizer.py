import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Importar datos
df = pd.read_csv('medical_examination.csv')  # Ajusta el nombre del archivo si es necesario.

# Añadir columna 'sobrepeso'
df['IMC'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = df['IMC'].apply(lambda x: 1 if x > 25 else 0)
df.drop('IMC', axis=1, inplace=True)

# Normalizar datos
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

def draw_cat_plot():
    # Crear DataFrame para el gráfico categórico
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    
    # Dibujar el gráfico categórico
    g = sns.catplot(data=df_cat, x='variable', hue='value', col='cardio', kind='count')
    g.set_axis_labels("variable", "total").set_titles("{col_name} cardio")
    
    fig = g.fig
    
    # Guardar la figura
    fig.savefig('catplot.png')
    return fig

def draw_heat_map():
    # Limpiar los datos
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # Calcular la matriz de correlación
    corr = df_heat.corr()

    # Generar una máscara para el triángulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Configurar la figura matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))

    # Dibujar el mapa de calor
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", linewidths=.5, square=True, center=0, cmap='coolwarm', ax=ax)
    
    # Guardar la figura
    fig.savefig('heatmap.png')
    return fig
