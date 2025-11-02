# ### INFORME DE LABORATORIO #4.
Señales electromiográficas EMG 
---------------
### OBJETIVOS
1. Aplicar el filtrado de señales continuas para procesar una señal electromiográfica
(EMG).
2. Detectar la aparición de fatiga muscular mediante el análisis espectral de
contracciones musculares individuales.
3. Comparar el comportamiento de una señal emulada y una señal real en términos
de frecuencia media y mediana.
4. Emplear herramientas computacionales para el procesamiento, segmentación y
análisis de señales biomédicas. 
### PARTE A
En esta primera etapa, se configuró el generador de señales biológicas en modo electromiografía (EMG) con el objetivo de simular cinco contracciones musculares voluntarias. Este procedimiento permite reproducir de manera controlada la actividad eléctrica generada por el músculo durante contracciones sucesivas. Previo a esto se segmento la señal, se calculo la frecuencia media y mediana para cada contraccion para luego graficar y tabular los resultados obtenidos.

```python
from scipy.signal import welch

# 1. Definir límites de cada contracción
# Usamos los tiempos detectados como referencia y tomamos un margen antes y después
margen = 0.05  # 50 ms antes y después del pico
segmentos = []

for i in range(len(tiempos_picos)):
    inicio = max(tiempos_picos[i] - margen, 0)
    fin = min(tiempos_picos[i] + margen, t.iloc[-1])
    mask = (t >= inicio) & (t <= fin)
    segmentos.append(v[mask].values)

# 2. Calcular frecuencia media y mediana con Welch para cada contracción
frecuencia_media = []
frecuencia_mediana = []

for seg in segmentos:
    fs = 1 / (t.iloc[1] - t.iloc[0])  # frecuencia de muestreo
    f, Pxx = welch(seg, fs=fs, nperseg=len(seg))
    f_media = np.sum(f * Pxx) / np.sum(Pxx)
    f_acumulada = np.cumsum(Pxx)
    f_mediana = f[np.where(f_acumulada >= np.sum(Pxx)/2)[0][0]]
    frecuencia_media.append(f_media)
    frecuencia_mediana.append(f_mediana)

# 3. Crear tabla de resultados
resultados = pd.DataFrame({
    'Contracción': np.arange(1, len(segmentos)+1),
    'Frecuencia Media (Hz)': frecuencia_media,
    'Frecuencia Mediana (Hz)': frecuencia_mediana
})

print(resultados)

# 4. Graficar evolución de las frecuencias
plt.figure(figsize=(10,5))
plt.plot(resultados['Contracción'], resultados['Frecuencia Media (Hz)'], 'o-', label='Frecuencia media')
plt.plot(resultados['Contracción'], resultados['Frecuencia Mediana (Hz)'], 's-', label='Frecuencia mediana')
plt.xlabel('Número de contracción')
plt.ylabel('Frecuencia (Hz)')
plt.title('Evolución de las frecuencias por contracción')
plt.legend()
plt.grid(True)
plt.show()

```

### RESULTADOS OBTENIDOS

![RFSULTADOS OBTENIDOS](https://github.com/TomasCobos-rgb/INFORME-LAB-SENALES-4/blob/main/IMAGENES/imagen_2025-11-01_192653468.png?raw=true)

### PARTE B
En esta fase se realizó la adquisición de la señal electromiográfica (EMG) proveniente de un voluntario sano, colocando los electrodos sobre un grupo muscular específico (como el bíceps o antebrazo). Durante el registro, el sujeto efectuó contracciones repetidas hasta la aparición de la fatiga muscular, permitiendo analizar cómo varían las componentes frecuenciales de la señal en condiciones reales

### PARTE C
En esta etapa se realizó el análisis espectral de la señal EMG mediante la aplicación de la Transformada Rápida de Fourier (FFT) a cada una de las contracciones registradas. Este procedimiento permite observar la distribución de las componentes frecuenciales de la señal y cómo estas varían a lo largo del tiempo. A partir del espectro de amplitud, se compararon las primeras contracciones con las últimas, identificando la disminución del contenido de alta frecuencia y el desplazamiento del pico espectral como indicadores del inicio y progresión de la fatiga muscular.
