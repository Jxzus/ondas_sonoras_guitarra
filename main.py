import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io.wavfile as wav
import os
import datetime
import glob

def analizar_audio(audio, fs, duracion):
    # Señal en el tiempo
    t = np.linspace(0, duracion, len(audio))
    # Aplicar ventana Hann antes de la FFT
    audio_flat = audio.flatten()
    ventana = np.hanning(len(audio_flat))
    audio_ventaneado = audio_flat * ventana
    # Transformada de Fourier
    N = len(audio_ventaneado)
    frecuencias = np.fft.rfftfreq(N, 1/fs)
    espectro = np.abs(np.fft.rfft(audio_ventaneado))
    espectro[0] = 0  # Ignorar DC
    # Detectar frecuencia fundamental
    indice_max = np.argmax(espectro)
    frecuencia_fundamental = frecuencias[indice_max]
    # Calcular armónicos
    armonicos = [frecuencia_fundamental * n for n in range(1, 4)]
    tabla = pd.DataFrame({
        "Armónico": ["1º", "2º", "3º"],
        "Frecuencia (Hz)": [round(f, 2) for f in armonicos]
    })
    return t, audio, frecuencias, espectro, frecuencia_fundamental, tabla

def mostrar_menu(t, audio, frecuencias, espectro, frecuencia_fundamental, tabla, fs):
    while True:
        print("\nOpciones:")
        print("1. Ver onda sonora")
        print("2. Ver espectro de frecuencias")
        print("3. Ver tabla de armónicos")
        print("4. Escuchar audio")
        print("5. Salir al menú principal")
        opcion = input("Selecciona una opción: ")
        if opcion == '1':
            plt.figure(figsize=(10,4))
            plt.plot(t, audio)
            plt.title("Señal de la guitarra en el tiempo")
            plt.xlabel("Tiempo [s]")
            plt.ylabel("Amplitud")
            plt.show()
        elif opcion == '2':
            plt.figure(figsize=(10,4))
            plt.plot(frecuencias, espectro)
            plt.title("Espectro de frecuencias")
            plt.xlabel("Frecuencia [Hz]")
            plt.ylabel("Amplitud")
            plt.xlim(0, 2000)
            plt.show()
            print(f"Frecuencia fundamental detectada: {frecuencia_fundamental:.2f} Hz")
        elif opcion == '3':
            print("\nTabla de armónicos:")
            print(tabla)
        elif opcion == '4':
            print("Reproduciendo audio...")
            sd.play(audio, fs)
            sd.wait()
        elif opcion == '5':
            break
        else:
            print("Opción no válida.")

# Parámetros de grabación
fs = 44100
duracion = 3.5

while True:
    print("\nMenú principal:")
    print("1. Grabar nueva nota de guitarra")
    print("2. Analizar una grabación anterior")
    print("3. Salir")
    eleccion = input("Selecciona una opción: ")
    if eleccion == '1':
        print("Grabando nota de guitarra...")
        audio = sd.rec(int(duracion * fs), samplerate=fs, channels=1, dtype='float64')
        sd.wait()
        print("Grabación terminada")
        carpeta_grabaciones = "grabaciones"
        os.makedirs(carpeta_grabaciones, exist_ok=True)
        nombre_usuario = input("Escribe un nombre para la grabación (sin espacios ni caracteres especiales): ")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = os.path.join(
            carpeta_grabaciones,
            f"{nombre_usuario}_{timestamp}.wav"
        )
        wav.write(nombre_archivo, fs, (audio * 32767).astype(np.int16))
        print(f"Audio guardado como {nombre_archivo}")
        t, audio_proc, frecuencias, espectro, frecuencia_fundamental, tabla = analizar_audio(audio, fs, duracion)
        mostrar_menu(t, audio_proc, frecuencias, espectro, frecuencia_fundamental, tabla, fs)
    elif eleccion == '2':
        carpeta_grabaciones = "grabaciones"
        archivos = glob.glob(os.path.join(carpeta_grabaciones, "*.wav"))
        if not archivos:
            print("No hay grabaciones guardadas.")
        else:
            print("Grabaciones disponibles:")
            for i, archivo in enumerate(archivos):
                print(f"{i+1}: {os.path.basename(archivo)}")
            seleccion = int(input("Selecciona el número de la grabación: ")) - 1
            archivo_seleccionado = archivos[seleccion]
            fs, audio = wav.read(archivo_seleccionado)
            audio = audio.astype(np.float64) / 32767
            duracion = len(audio) / fs
            print(f"Analizando {archivo_seleccionado}")
            t, audio_proc, frecuencias, espectro, frecuencia_fundamental, tabla = analizar_audio(audio, fs, duracion)
            mostrar_menu(t, audio_proc, frecuencias, espectro, frecuencia_fundamental, tabla, fs)
    elif eleccion == '3':
        print("Saliendo...")
        break
    else:
        print("Opción no válida.")
