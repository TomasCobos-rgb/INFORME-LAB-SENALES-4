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
![DIAGRAMA DE FLUJO](https://github.com/TomasCobos-rgb/INFORME-LAB-SENALES-4/blob/main/IMAGENES/Beige%20Minimal%20Flowchart%20Infographic%20Graph%20(3).png?raw=true)
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

### RESULTADOS OBTENIDOS
La siguente gráfica muestra la señal electromiográfica (EMG) original registrada en el laboratorio. En ella se observan las variaciones de amplitud en el tiempo, que reflejan la actividad eléctrica generada por las fibras musculares del corazón durante las contracciones. Esta señal cruda sirve como punto de partida para las etapas posteriores de procesamiento, como el filtrado y el análisis espectral, permitiendo identificar patrones asociados a la dinámica muscular.
```python
# Señal original completa
axs[0].plot(t, senal, color='gray', linewidth=0.8)
axs[0].set_title('Señal EMG Original Completa')
axs[0].set_xlabel('Tiempo [s]')
axs[0].set_ylabel('Amplitud [mV]')
axs[0].grid(True)
```
<img width="919" height="265" alt="Señal_OG_Real" src="https://github.com/user-attachments/assets/710b52c3-4305-4e67-99ab-fdb564e9d911" />

### APLICACIÓN DEL FILTRO

En esta figura se comparan la señal electromiográfica (EMG) original y la señal filtrada mediante un filtro pasa banda entre 20 y 450 Hz. El proceso de filtrado permite eliminar el ruido de baja frecuencia (como artefactos de movimiento o interferencias de línea base) y atenuar componentes de alta frecuencia no relacionadas con la actividad muscular, conservando las frecuencias características de la señal EMG. Como resultado, la señal filtrada (en azul) muestra una forma más limpia y representativa de la actividad eléctrica muscular, facilitando su análisis en etapas posteriores como la detección de contracciones o el estudio espectral.

![DIAGRAMA DE FLUJO](https://github.com/TomasCobos-rgb/INFORME-LAB-SENALES-4/blob/main/IMAGENES/Beige%20Minimal%20Flowchart%20Infographic%20Graph%20(4).png?raw=true)

```python

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def aplicar_filtro(senal, lowcut=20, highcut=450, fs=1000, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, senal)

senal_filtrada = aplicar_filtro(senal, 20, 450, Fs)
# Original vs Filtrada (superpuestas)
axs[1].plot(t, senal, color='gray', linewidth=0.8, alpha=0.6, label='Original')
axs[1].plot(t, senal_filtrada, color='blue', linewidth=1.2, label='Filtrada (20–450 Hz)')
axs[1].set_title('Comparación: Señal EMG Original vs Filtrada')
axs[1].set_xlabel('Tiempo [s]')
axs[1].set_ylabel('Amplitud [mV]')
axs[1].legend(loc='upper right')
axs[1].grid(True)
```

<img width="1042" height="297" alt="Graf_og_vs_filt" src="https://github.com/user-attachments/assets/9a629d82-fd1b-4e99-b139-956ac7fd0e0f" />

### ACERCAMIENTO A GRAFICA FILTRADA VS ORIGINAL

La figura muestra un acercamiento a los primeros tres segundos de la señal EMG, comparando la forma de onda original (en gris) con la señal filtrada mediante un filtro pasa banda de 20 a 450 Hz (en azul). Este enfoque con zoom permite apreciar con mayor detalle cómo el filtrado elimina las variaciones lentas y el ruido de baja frecuencia, resaltando las oscilaciones rápidas que representan la verdadera actividad eléctrica muscular durante las contracciones. Gracias a este filtrado, la señal resultante conserva las componentes fisiológicamente relevantes y facilita el análisis posterior de la dinámica muscular y la frecuencia de contracción.

![DIAGRAMA DE FLUJO](https://github.com/TomasCobos-rgb/INFORME-LAB-SENALES-4/blob/main/IMAGENES/Beige%20Minimal%20Flowchart%20Infographic%20Graph%20(5).png?raw=true)


```python
  Gráfica comparativa: señal original y filtrada superpuestas ---
plt.figure(figsize=(14,6))

# Señal original en gris
plt.plot(t, senal, color='gray', linewidth=0.8, alpha=0.6, label='Señal original')

# Señal filtrada en azul más destacada
plt.plot(t, senal_filtrada, color='blue', linewidth=1.2, label='Señal filtrada (20–450 Hz)')

plt.title('Comparación: Señal EMG Original vs Filtrada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [mV]')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

#  Gráfico con zoom en una región (por ejemplo, primeros 3 segundos) ---
inicio_zoom = 0       # segundo inicial
fin_zoom = 3          # segundo final
muestras_zoom = slice(int(inicio_zoom * Fs), int(fin_zoom * Fs))

plt.figure(figsize=(14,5))
plt.plot(t[muestras_zoom], senal[muestras_zoom], color='gray', linewidth=0.8, alpha=0.6, label='Original')
plt.plot(t[muestras_zoom], senal_filtrada[muestras_zoom], color='blue', linewidth=1.2, label='Filtrada')
plt.title(f'Zoom de la Señal EMG (entre {inicio_zoom}s y {fin_zoom}s)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [mV]')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

```

<img width="1042" height="299" alt="Zoom_og_vs_filt" src="https://github.com/user-attachments/assets/b32fc835-80d3-4924-8203-a5bc8916061b" />

### SEGMENTACIÓN

En esta etapa del procesamiento, la señal EMG previamente filtrada fue segmentada en 83 contracciones musculares individuales. Esta segmentación permite analizar de forma separada cada evento de contracción, facilitando la observación de su forma de onda, duración y variaciones en la amplitud. Como se observa en las figura, cada segmento representa una contracción distinta del músculo, mostrando cómo la actividad eléctrica varía a lo largo del tiempo. Este procedimiento es esencial para estudios de fatiga muscular, activación motora y análisis de patrones de esfuerzo.

```python
num_contracciones = 83
L = len(senal_filtrada)
segmentos = np.array_split(senal_filtrada, num_contracciones)


```
<img width="1042" height="437" alt="Señal_div" src="https://github.com/user-attachments/assets/b10ce813-d244-4f6b-9de9-e7a3604190fc" />


### CÁLCULO DE MEDIA Y MEDIANA (GRÁFICA)

En esta fase se calcularon la frecuencia media y la frecuencia mediana de cada una de las 83 contracciones musculares segmentadas. Estos parámetros espectrales permiten evaluar la distribución de energía de la señal EMG en el dominio de la frecuencia. La frecuencia media refleja el promedio ponderado de las componentes frecuenciales, mientras que la frecuencia mediana indica el punto que divide el espectro de potencia en dos partes iguales. En la gráfica se observa la evolución de ambas medidas a lo largo de las contracciones, mostrando variaciones que pueden asociarse a cambios en la activación muscular o a la aparición de fatiga durante el registro.

![DIAGRAMA DE FLUJO](https://github.com/TomasCobos-rgb/INFORME-LAB-SENALES-4/blob/main/IMAGENES/Beige%20Minimal%20Flowchart%20Infographic%20Graph%20(6).png?raw=true)

```python
def calcular_frecuencias(segmento, fs):
    f, Pxx = welch(segmento, fs=fs, nperseg=1024)
    Pxx = Pxx / np.sum(Pxx)  # Normalizar el espectro de potencia
    freq_media = np.sum(f * Pxx)
    cum_Pxx = np.cumsum(Pxx)
    freq_mediana = f[np.where(cum_Pxx >= 0.5)[0][0]]
    return freq_media, freq_mediana

frecuencia_media = []
frecuencia_mediana = []

for seg in segmentos:
    f_mean, f_median = calcular_frecuencias(seg, Fs)
    frecuencia_media.append(f_mean)
    frecuencia_mediana.append(f_median)

# %% --- Graficar la evolución de las frecuencias ---
plt.figure(figsize=(10,5))
plt.plot(frecuencia_media, 'o-', label='Frecuencia Media')
plt.plot(frecuencia_mediana, 's-', label='Frecuencia Mediana')
plt.title('Evolución de la Frecuencia Media y Mediana (83 Contracciones)')
plt.xlabel('Número de Contracción')
plt.ylabel('Frecuencia [Hz]')
plt.legend()
plt.grid()
plt.show()

tabla_resultados = pd.DataFrame({
    'Contracción': np.arange(1, num_contracciones + 1),
    'Frecuencia Media (Hz)': np.round(frecuencia_media, 2),
    'Frecuencia Mediana (Hz)': np.round(frecuencia_mediana, 2)
})

print(tabla_resultados)
tabla_resultados.to_csv('/content/drive/MyDrive/frecuencias_EMG.csv', index=False)
print("\nArchivo 'frecuencias_EMG.csv' guardado en tu Drive con los resultados.")
```

<img width="390" height="169" alt="Val_med_83" src="https://github.com/user-attachments/assets/6f1b8df5-0469-411e-8947-0016057a826e" />

<img width="631" height="345" alt="Graf_media_med" src="https://github.com/user-attachments/assets/632bf771-c275-4148-ac95-3a8a68ccdce0" />


### PARTE C
En esta etapa se realizó el análisis espectral de la señal EMG mediante la aplicación de la Transformada Rápida de Fourier (FFT) a cada una de las contracciones registradas. Este procedimiento permite observar la distribución de las componentes frecuenciales de la señal y cómo estas varían a lo largo del tiempo. A partir del espectro de amplitud, se compararon las primeras contracciones con las últimas, identificando la disminución del contenido de alta frecuencia y el desplazamiento del pico espectral como indicadores del inicio y progresión de la fatiga muscular.

### FFT

En esta etapa se aplicó la Transformada Rápida de Fourier (FFT) a la primera y última contracción muscular, con el fin de analizar la distribución espectral de la señal EMG. Esta herramienta permite observar cómo se concentra la energía en diferentes frecuencias, facilitando la identificación de cambios en la actividad muscular a lo largo del tiempo.

![DIAGRAMA DE FLUJO](https://github.com/TomasCobos-rgb/INFORME-LAB-SENALES-4/blob/main/IMAGENES/Beige%20Minimal%20Flowchart%20Infographic%20Graph%20(7).png?raw=true)

```python
def graficar_fft(segmento, fs, titulo):
    N = len(segmento)
    f = np.fft.rfftfreq(N, 1/fs)
    fft_mag = np.abs(np.fft.rfft(segmento))
    plt.plot(f, fft_mag)
    plt.title(titulo)
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Magnitud')
    plt.grid()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
graficar_fft(segmentos[0], Fs, 'FFT Contracción 1 (Inicio)')
plt.subplot(1,2,2)
graficar_fft(segmentos[-1], Fs, 'FFT Contracción 83 (Final)')
plt.tight_layout()
plt.show()


```

<img width="891" height="367" alt="FFT_contr" src="https://github.com/user-attachments/assets/080e56a4-166c-4c6a-80c0-a751d031024b" />


### ESPECTRO DE AMPLITUD 

En esta fase se obtuvo el espectro de amplitud de la señal EMG utilizando la Transformada Rápida de Fourier (FFT) y representándolo en una escala logarítmica de frecuencia. Este análisis permite visualizar cómo se distribuye la energía de la señal en diferentes bandas de frecuencia, proporcionando una visión más detallada del contenido espectral.

![DIAGRAMA DE FLUJO](https://github.com/TomasCobos-rgb/INFORME-LAB-SENALES-4/blob/main/IMAGENES/Beige%20Minimal%20Flowchart%20Infographic%20Graph%20(8).png?raw=true)

```python
# FFT de la señal filtrada
N = len(senal_filtrada)             # número de muestras
fft_vals = np.fft.fft(senal_filtrada)      # transformada compleja
fft_mag = np.abs(fft_vals) / N             # magnitud normalizada
fft_mag = fft_mag[:N//2] * 2               # solo mitad positiva, factor 2 por simetría
freqs = np.fft.fftfreq(N, 1/Fs)[:N//2]     # vector de frecuencias (0 a Fs/2)



# --- Gráfico con eje de frecuencia logarítmico ---
plt.figure(figsize=(12,5))
plt.plot(freqs, fft_mag, color='purple', linewidth=1)
plt.xscale('log')
plt.xlim(10, 500)
plt.title('Espectro de Amplitud (escala logarítmica)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud')
plt.grid(True, which='both', ls='--', alpha=0.6)
plt.show()


```
<img width="762" height="356" alt="Espect_ampl" src="https://github.com/user-attachments/assets/ad5ab001-2b2e-4ee0-a6fd-e869153839c5" />

### ESPECTRO DE AMPLITUD (WELCH)

![DIAGRAMA DE FLUJO](https://github.com/TomasCobos-rgb/INFORME-LAB-SENALES-4/blob/main/IMAGENES/Beige%20Minimal%20Flowchart%20Infographic%20Graph%20(9).png?raw=true)

```python
idx_primera = 0
idx_media = num_contracciones // 2
idx_ultima = num_contracciones - 1
espectros_seleccion = {}

for i, seg in enumerate(segmentos):
    # Si el segmento es muy corto para nperseg, reducir nperseg
    this_nperseg = min(nperseg, len(seg))
    if this_nperseg < 8:
        # segmento demasiado corto; rellenar con ceros o saltar
        f = np.array([0.])
        Pxx = np.array([0.])
    else:
        f, Pxx = welch(seg, fs=Fs, nperseg=this_nperseg, noverlap=this_nperseg//2)

    # guardar espectros seleccionados
    if i in (idx_primera, idx_media, idx_ultima):
        espectros_seleccion[i] = (f, Pxx)

plt.figure(figsize=(14,6))
for idx, (f, Pxx) in espectros_seleccion.items():
    Pxx_db = 10 * np.log10(Pxx + 1e-20)   # en dB
    plt.plot(f, Pxx_db, label=f'Contracción {idx+1}')
plt.xscale('log')
plt.xlim(10, Fs/2)
plt.xlabel('Frecuencia [Hz] (log)')
plt.ylabel('PSD [dB]')
plt.title('Espectros (Welch) - escala log (dB)')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.show()

```

<img width="877" height="412" alt="Espect_Welch_db" src="https://github.com/user-attachments/assets/50c50c4c-ebb7-43da-835c-aca8b4e7f9d8" />

### CÁLCULO DEL PICO ESPECTRAL Y GRÁFICA

![DIAGRAMA DE FLUJO](https://github.com/TomasCobos-rgb/INFORME-LAB-SENALES-4/blob/main/IMAGENES/Beige%20Minimal%20Flowchart%20Infographic%20Graph%20(10).png?raw=true)

```python
 idx_max = np.argmax(Pxx)
    fpico = f[idx_max]
    pico_espectral.append(fpico)

centroid = np.sum(f * Pxx) / np.sum(Pxx)
    freq_centroid.append(centroid)

plt.figure(figsize=(11,5))
plt.plot(np.arange(1, num_contracciones+1), pico_espectral, '-o', label='Pico espectral (Hz)')
plt.plot(np.arange(1, num_contracciones+1), freq_centroid, '-s', label='Centroide (Hz)')
plt.xlabel('Número de contracción')
plt.ylabel('Frecuencia [Hz]')
plt.title('Evolución del pico espectral y centroide a lo largo de las contracciones')
plt.legend()
plt.grid(True)
plt.show()

print(f"- Pico medio (Hz): {np.round(np.mean(pico_espectral),2)} ± {np.round(np.std(pico_espectral),2)}")
print(f"- Centroide medio (Hz): {np.round(np.mean(freq_centroid),2)} ± {np.round(np.std(freq_centroid),2)}")
print(f"- Fracción media >100Hz: {np.round(np.mean(frac_pot_high100),3)}")
print(f"- Fracción media >250Hz: {np.round(np.mean(frac_pot_high250),3)}")
```

<img width="689" height="345" alt="Pico_espect" src="https://github.com/user-attachments/assets/ac27dbc3-4895-40c2-b595-75c8f6578cb2" />

<img width="208" height="54" alt="Val_Punto_C" src="https://github.com/user-attachments/assets/f994d498-6787-44b8-bd98-61ff27dd1bbd" />

### CONCLUSIONES 
- En la FFT de la contracción 1 (inicio) se aprecian picos distribuidos en un rango más amplio de frecuencias, lo que sugiere una mayor variabilidad inicial en la activación de las fibras musculares. En cambio, la FFT de la contracción 83 (final) muestra un pico más definido y concentrado, indicando una posible disminución de la frecuencia media asociada a la aparición de fatiga muscular. Este análisis espectral es fundamental para estudiar la dinámica de la señal EMG y su relación con el rendimiento muscular.
- En el espectro de amplitud de la señal EMG utilizando la Transformada Rápida de Fourier (FFT), la gráfica se observa una concentración principal de energía entre 20 y 150 Hz, rango típico de la actividad electromiográfica muscular. Los picos más altos corresponden a las frecuencias dominantes generadas durante las contracciones musculares. La escala logarítmica facilita identificar variaciones en componentes de baja y alta frecuencia, lo que resulta esencial para comprender el comportamiento de la señal y detectar posibles cambios en la fatiga o en la activación neuromuscular.
