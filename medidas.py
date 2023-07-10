# -*- coding: utf-8 -*-
"""
Created on Wed May 10 13:01:55 2023

@author: javil

-------------------------MEDIDAS-------------------------

Script que realiza medidas objetivas sobre los mfcc y .wav sintetizados por 
medio de unit selection. Este código implementa las siguientes medidas:
    -Mel Cepstral Distance (para MFCC's)
    -STOI (para audio)
    -PESQ (para audio)

Adicionalmente, el script obtiene figuras para comparar la forma de audio original
y sintetizada, así como el espectrograma
"""
import numpy as np
import wave
import os
import soundfile as sf
from pystoi import stoi
import math
import matplotlib.pyplot as plt
from scipy import signal
import librosa

#import torch
#from torch_pesq import PesqLoss
plt.close('all')

# Definimos la ruta de nuestro fichero
"""---------------------DETERMINACIÓN DE LOS PATH---------------------"""

#DATOS ORIGINALES
filepath_mfcc = 'data\PMA\TiDigit\TP\world'
filepath_wav = 'data\PMA\TiDigit\TP\wav'

#DATOS SINTETIZADOS
filepath_mfcc_synt = 'data\PMA\TiDigit\TP\mfcc_synt_regresion'
filepath_wav_synt = 'data\PMA\TiDigit\TP\wav_synt_regresion'


"""-------------------------------------------------------------------"""


#Se crea una lista que enumere todos los ficheros de la carpeta que nos interese
extension = '.npy'
filenames = [archivo for archivo in os.listdir(filepath_mfcc) if archivo.endswith(extension)]
filenames = np.array([os.path.splitext(archivo)[0] for archivo in filenames])
ficheros = filenames.shape[0]

#Hacemos lo mismo para los ficheros .wav
extension_wav = '.wav'
filenames = [archivo for archivo in os.listdir(filepath_wav) if archivo.endswith(extension_wav)]
filenames = np.array([os.path.splitext(archivo)[0] for archivo in filenames])
ficheros = filenames.shape[0]


#Creamos la función que nos implementa el Mel-Cepstral Distance
def mcd(x, y, discard_c0=True):
    '''
    Calcula la medidal Mel-Cepstral Distortion (MCD) entre dos matrices de MFCCs calculados con World.
    
    Las matrices deben tener el mismo número de coeficientes MFCC (axis=1). Si tienen distinto número de tramas (axis=0),
    se calcula la MCD para las primeras tramas comunes frames= min(x.shape[0], y.shape[0]).
    
    Parameters
    ----------
    x: matriz de MFCCs calculados con PyWorld de tamaño (frames_x, num_mfccs)
    
    y: matriz de MFCCs calculados con PyWorld de tamaño (frames_y, num_mfccs)
    
    Returns
    -------
    mcd: vector de tamaño min(frames_x, framex_y) con el MCD calculado para cada trama
    '''
    
    log_spec_dB_const = (10.0 * np.sqrt(2.0)) / np.log(10.0) 
    
    frames = min(x.shape[0], y.shape[0])
    idx = 1 if discard_c0 else 0
    
    diff = x[:frames,idx:] - y[:frames,idx:]
    #print(diff.shape)
    return log_spec_dB_const * np.linalg.norm(diff, axis=1)


def plot_audio(original_signal, synthesized_signal, sample_rate):
    
    """Función que realiza una representación en el dominio del tiempo de dos señales de audio"""
    
    #Igualamos las longitudes
    len_wav = min(wav_orig.shape[0], wav_synt.shape[0])
    original_signal = original_signal[0:len_wav]
    synthesized_signal = synthesized_signal[0:len_wav]

    
    # Calcular el eje x en segundos
    duration = len(original_signal) / sample_rate
    t = np.linspace(0, duration, len(original_signal))

    # Crear la figura y los ejes
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Plot del audio original
    ax[0].plot(t, original_signal, color='blue')
    ax[0].set_ylabel('Amplitud')
    ax[0].set_title('Audio Original')

    # Plot del audio sintetizado
    ax[1].plot(t, synthesized_signal, color='red')
    ax[1].set_xlabel('Tiempo (segundos)')
    ax[1].set_ylabel('Amplitud')
    ax[1].set_title('Audio Sintetizado')

    # Ajustar los márgenes
    plt.tight_layout()

    # Mostrar la figura
    plt.show()
    
    #PARA REALIZAR UN PLOT ÚNICAMENTE DE UNA DE LAS SEÑALES
    # Crear la figura y el eje
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot del audio
    ax.plot(t, synthesized_signal, color='blue')
    ax.set_xlabel('Tiempo (segundos)')
    ax.set_ylabel('Amplitud')
    ax.set_title('Audio')

    # Ajustar los márgenes
    plt.tight_layout()

    # Mostrar la figura
    plt.show()









    
def plot_spectrogram(original_signal, synthesized_signal, sample_rate):
           
    #Igualamos las longitudes
    len_wav = min(wav_orig.shape[0], wav_synt.shape[0])
    original_signal = original_signal[0:len_wav]
    synthesized_signal = synthesized_signal[0:len_wav]
    
    # Calcular los espectrogramas
    f, t, S_original = signal.spectrogram(original_signal, fs=sample_rate)
    f, t, S_synthesized = signal.spectrogram(synthesized_signal, fs=sample_rate)

    # Crear la figura y los ejes
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Plot del espectrograma del audio original
    ax[0].imshow(np.log(S_original), aspect='auto', cmap='jet', origin='lower',
                 extent=[t[0], t[-1], f[0], f[-1]])
    ax[0].set_ylabel('Frecuencia (Hz)')
    ax[0].set_title('Espectrograma del Audio Original')

    # Plot del espectrograma del audio sintetizado
    ax[1].imshow(np.log(S_synthesized), aspect='auto', cmap='jet', origin='lower',
                 extent=[t[0], t[-1], f[0], f[-1]])
    ax[1].set_xlabel('Tiempo (segundos)')
    ax[1].set_ylabel('Frecuencia (Hz)')
    ax[1].set_title('Espectrograma del Audio Sintetizado')

    # Ajustar los márgenes
    plt.tight_layout()

    # Mostrar la figura
    plt.show()
    
    #PARA HACER PLOT DE UN ÚNICO ESPECTROGRAMA
    
    # Calcular el espectrograma
    f, t, S_original = signal.spectrogram(synthesized_signal, fs=sample_rate)
    
    # Crear la figura y el eje
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot del espectrograma del audio original
    ax.imshow(np.log(S_original), aspect='auto', cmap='jet', origin='lower',
               extent=[t[0], t[-1], f[0], f[-1]])
    ax.set_xlabel('Tiempo (segundos)')
    ax.set_ylabel('Frecuencia (Hz)')
    ax.set_title('Espectrograma del Audio')
    
    # Ajustar los márgenes
    plt.tight_layout()
    
    # Mostrar la figura
    plt.show()



#Creamos arrays para almacenar las medidas de cada fichero
array_mcd = np.zeros((ficheros,1))
array_stoi = np.zeros((ficheros,1))
array_pesq = np.zeros((ficheros,1))


i = 0 #Iterador para ir rellenando los ficheros

#Iteramos para todos los ficheros de la carpeta
for filename in filenames:
    
    #Importamos los ficheros correspondientes a los MFCC's y los wav
    mfcc_orig = np.load((os.path.join(filepath_mfcc,filename) + '.npy'))
    mfcc_synt = np.load((os.path.join(filepath_mfcc_synt,filename) + '.npy'))
    
    wav_orig, fs = sf.read((os.path.join(filepath_wav,filename) + '.wav'))
    wav_synt, fs = sf.read((os.path.join(filepath_wav_synt,filename) + '.wav'))
    
    mel_cepstral_dist = mcd(mfcc_orig, mfcc_synt) 
    mcd_iter = np.mean(mel_cepstral_dist)
    #print('La medida de Mel-Cepstral distance es: ', mcd_iter, ' dB')
    
    #Almacenamos el valor del mcd
    array_mcd[i] = mcd_iter
    
    #Obtenemos la métrica del STOI para los ficheros .wav (mismo num de elementos)
    len_wav = min(wav_orig.shape[0], wav_synt.shape[0])
    metrica_stoi = stoi(wav_orig[0:len_wav], wav_synt[0:len_wav], fs, extended=False)
    
    #Obtenemos la métrica del PESQ para los ficheros .wav (0.5 no sé para que es)
    #pesq = PesqLoss(0.5, sample_rate = fs, )
    #metrica_pesq = pesq.mos(wav_orig[0:len_wav], wav_synt[0:len_wav])
    
    #Almacenamos el valor del stoi
    array_stoi[i] = metrica_stoi
    #print('La medida del STOI es: ', metrica_stoi)
    
    #Almacenamos el valor del PESQ
    #array_pesq[i] = metrica_pesq
    #print('La medida del PESQ es: ', metrica_pesq)
    
    
    #Sacamos una gráfica del espectrograma y la forma de onda para el primer fichero (o el que quiera)
    
    if (i == 2):
        
        plot_audio(wav_orig, wav_synt, fs)
        plot_spectrogram(wav_orig, wav_synt, fs)
        

    #Actualizamos el iterador
    i = i + 1
    
media_mcd = np.mean(array_mcd)
print('El valor medio del MCD para la carpeta es: ', media_mcd, ' dB')

media_stoi = np.mean(array_stoi)
print('El valor medio del STOI para la carpeta es: ', media_stoi)

#media_pesq = np.mean(array_pesq)
#print('El valor medio del PESQ para la carpeta es: ', media_pesq)






    



