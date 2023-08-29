# 🩺 Visualización de Datos Médicos 

Este proyecto tiene como objetivo visualizar datos relacionados con características médicas, tales como presión arterial, colesterol y hábitos de vida (como fumar o beber), y analizar su relación con la presencia o ausencia de enfermedades cardiovasculares.

## 📋 Características del Dataset

El conjunto de datos contiene las siguientes características:

- 📆 Edad (en días)
- 📏 Altura (en cm)
- ⚖️ Peso (en kg)
- 👤 Género
- ❤️ Presión arterial sistólica
- ❤️ Presión arterial diastólica
- 🍳 Colesterol (normal, por encima de lo normal, muy por encima de lo normal)
- 🍬 Glucosa (normal, por encima de lo normal, muy por encima de lo normal)
- 🚬 Fumador (binario)
- 🍺 Consumo de alcohol (binario)
- 🏃‍♂️ Actividad física (binario)
- 💔 Presencia o ausencia de enfermedad cardiovascular (objetivo, binario)

## 🛠 Instalación y Uso

1. Asegúrate de tener instaladas las bibliotecas necesarias. Puedes instalarlas usando:

   ```
   pip install pandas seaborn matplotlib numpy
   ```

2. 📦 Clona este repositorio en tu máquina local:

   ```
   git clone [URL del repositorio]
   ```

3. 📂 Navega al directorio del proyecto:

   ```
   cd [nombre del directorio]
   ```

4. ▶️ Ejecuta el script principal:

   ```
   python main.py
   ```

Esto generará dos visualizaciones: un gráfico categórico y un mapa de calor, y los guardará como `catplot.png` y `heatmap.png` respectivamente.

## 🤝 Contribuciones

Si deseas contribuir al proyecto, por favor, abre un "pull request". Asegúrate de probar cualquier cambio con el conjunto de pruebas proporcionado.

## ⚖️ Licencia

Este proyecto es de código abierto y está disponible bajo la [Licencia MIT](LICENSE).

---


This is the boilerplate for the Medical Data Visualizer project. Instructions for building your project can be found at https://www.freecodecamp.org/learn/data-analysis-with-python/data-analysis-with-python-projects/medical-data-visualizer
