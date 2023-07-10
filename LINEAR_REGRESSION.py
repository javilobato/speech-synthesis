# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:46:07 2023

@author: javil

-------------------------LINEAR REGRESSION-------------------------

Script que implementa el método de Regresión Lineal para una base de datos.
Se parte de una carpeta de ficheros .npy que contengan medidas de sensores
PMA y otra que contenga los MFCC's correspondientes.
El script lleva a cabo una división k-fold que va particionando la base de 
datos para usar unos ficheros para la base de datos y otros para test. 
Finalmente, se recorren todos los ficheros y se obtiene cada fichero original
sintetizado por medio de regresión lineal. 

INPUTS:
    -Path a los ficheros del sensor
    -Path a los ficheros de mfcc
    -Path para los ficheros mfcc sintetizados

OUTPUTS:
    -ficheros de mfcc sintetizados por medio de regresión lineal
    
"""
import os
from sklearn.model_selection import KFold
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
import scipy.spatial as ss
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


"""---------------------DETERMINACIÓN DE LOS PATH---------------------"""
#Se determina el path a cada carpeta

#DATOS ORIGINALES
#filepath_sensor = 'data\sensor'
#filepath_mfcc = 'data\world'

#DATOS PMA ARCTIC   (escribir JG o RM para elegir carpeta)
filepath_sensor = 'data\PMA\TiDigit\LC\sensor'
filepath_mfcc = 'data\PMA\TiDigit\LC\world'


#Si la carpeta no existe, se crea
filepath_mfcc_synthesized = 'data\PMA\TiDigit\LC\mfcc_synt_regresion'
if not os.path.exists(filepath_mfcc_synthesized):
    os.makedirs(filepath_mfcc_synthesized)


"""-------------------------------------------------------------------"""

#Se crea una lista que enumere todos los ficheros .npy de la carpeta que nos interese
extension = '.npy'
filenames = [archivo for archivo in os.listdir(filepath_sensor) if archivo.endswith(extension)]
filenames = np.array(filenames)


#frameshift 5 ms = 0.005
#frameshift 10 ms = 0.01

#Definimos la función que crea las unidades
def units(array, sr, windowLength = 0.02, frameshift = 0.01):
    """
    Función que recibe un array y devuelve otro array con unidades divididas en
    base a la longitud de ventana y solapación de frame requeridos

    Parameters
    ----------
    array : array (muestras, canales)
        array que contiene todas las muestras sin división para varios canales.
    sr : int
        Frecuencia de muestreo de los datos.
    windowLength : float, optional
        Longitud de la ventana en segundos. The default is 0.05.
    frameshift : float, optional
        Solapamiento entre unidades en segundos. The default is 0.005.

    Returns
    -------
    units : array (numwindows, windowlength * canales)
        Array que contiene en cada índice una unidad

    """
    
    #Numero de ventanas
    numWindows = int(np.floor((array.shape[0] - windowLength*sr)/(frameshift*sr)))
    
    #Establecemos la longitud de cada ventana en muestras
    windowlength_muestras = int( np.floor(windowLength * sr))
    
    #Se crea el array de arrays. Cada índice contendrá una unidad completa
    units = np.zeros((numWindows, windowlength_muestras*np.shape(array)[1]))
    
    #Iteramos para la longitud completa del array
    for win in range(numWindows):
        
        #Establecemos el comienzo y el final de la ventana
        start= int(np.floor((win*frameshift)*sr)) 
        stop = int(np.floor(start+windowLength*sr))
        
        units[win, :] = np.reshape(array[start:stop,:],(windowlength_muestras*np.shape(array)[1]))
    return units



#Dividiremos la base de datos en 10 secciones. 9 se usarán para train y una 
#para test
kf = KFold(n_splits=10)
KFold(n_splits=10, random_state=None, shuffle=False)

windowLength = 0.02 #Tamaño de la ventana en segundo
#sr = 200            #Frecuencia de muestreo de los datos
sr = 100            #Frecuencia de muestreo de los datos PARA LOS DE 10 MS DE Ts
canales_pma = 9     #Dimensionalidad de los datos PMA
canales_mfcc = 25   #Dimensionalidad de los datos MFCC


#Este bucle se repite 10 veces
for i, (train_index, test_index) in enumerate(kf.split(filenames)):
    
    #print(f"Fold {i}:")
    #print(f"  Train: index={train_index}")
    #print(f"  Test:  index={test_index}")
    
    #Creamos la base de datos de unidades para la secuencia de train  y test
    units_train_sensor = np.zeros((1, int(windowLength*sr)*canales_pma))
    units_train_mfcc = np.zeros((1, canales_mfcc))
    units_test_sensor = np.zeros((1, int(windowLength*sr)*canales_pma))
    units_test_mfcc = np.zeros((1, canales_mfcc))
    
    #Obtenemos las unidades de entrenamiento para el fold concreto
    for index in train_index:
        
        name = filenames[index] #Recuperamos el nombre del archivo que corresponde con el índice
        
        path_sensor = (os.path.join(filepath_sensor,name))
        sensor = np.load(path_sensor)
        
        path_mfcc = (os.path.join(filepath_mfcc,name))
        mfcc = np.load(path_mfcc)
        
        #Llamamos a la función que nos obtiene las unidades (sólo es necesaria para el sensor)
        units_sensor = units(sensor, sr)

        #Las unidades del mfcc ya se encuentran estructuradas según nuestro interés
        units_mfcc = mfcc
        
        #Igualamos el número de unidades según el menor de los dos arrays
        minimo = min(units_sensor.shape[0],units_mfcc.shape[0]) 
        units_sensor = units_sensor[0:minimo]
        units_mfcc = units_mfcc[0:minimo]
        

        #Concatenamos e incorporamos a la base de datos
        units_train_sensor = np.concatenate((units_train_sensor, units_sensor),0)
        units_train_mfcc = np.concatenate((units_train_mfcc, units_mfcc),0)
     
    """NORMALIZACIÓN DE LOS FEATURES"""
    #Creamos los dos scalers
    scaler_sensor = MinMaxScaler()
    scaler_mfcc = MinMaxScaler()
    
    #Hacemos fit y transformamos ambas bases de datos
    units_train_sensor = scaler_sensor.fit_transform(units_train_sensor)  
    units_train_mfcc = scaler_mfcc.fit_transform(units_train_mfcc)
        
     
        
    """AQUI SE HACE EL FIT DE LOS DATOS DE ENTRENAMIENTO (TRAS EL BUCLE)"""
    #Asociamos las unidades del sensor a las correspondientes de MFCC
    LinReg = LinearRegression().fit(units_train_sensor, units_train_mfcc)
    
    #Creamos la base de datos con estructura de árbol para las unidades de entrenamiento
    #Tree_train = BallTree(units_train_sensor,leaf_size = 40)    
    
    #Realizamos la evaluación con las muestras de test
    for index2 in test_index:
        
        name_test = filenames[index2]
        
        path_sensor = (os.path.join(filepath_sensor,name_test))
        sensor_test = np.load(path_sensor)
        
        path_mfcc = (os.path.join(filepath_mfcc,name_test))
        mfcc_test = np.load(path_mfcc)
        
        #Llamamos a la función que nos obtiene las unidades
        units_sensor_test = units(sensor_test, sr)
        units_mfcc_test = mfcc_test
        
        #Igualamos el número de unidades según el menor de los dos arrays
        minimo = min(units_sensor_test.shape[0],units_mfcc_test.shape[0]) 
        units_sensor_test = units_sensor_test[0:minimo]
        units_mfcc_test = units_mfcc_test[0:minimo]
        
        #Esto sólo está por la estructura de los nombres
        units_test_sensor = units_sensor_test
        units_test_mfcc = units_mfcc_test
        
        
        """NORMALIZACIÓN DE LOS FEATURES"""
        #Ahora transformamos únicamente las unidades de un fichero  
        units_test_sensor = scaler_sensor.transform(units_test_sensor)
        units_test_mfcc = scaler_mfcc.transform(units_test_mfcc)
        
        
        """AQUI SE HACE EL PREDICT USANDO EL MODELO LINEAL"""
        units_mfcc_final = LinReg.predict(units_test_sensor)
        
        """SECCIÓN DE EVALUACIÓN DE LA DISTANCIA"""
        #IMPLEMENTACIÓN DE LA PRIMERA MATRIZ: EVALUACIÓN DE DISTANCIA
        
        #Definimos el número de vecinos más próximos que queremos
        #N_vecinos = 10
        
        #Evaluamos los 10 vecinos más próximos para cada unidad
        #Evaluamos la distancia con un query a BallTree
        #[dist_test, ind_test] = Tree_train.query(units_test_sensor, k = N_vecinos)
                  
        
        #CONSTRUCCIÓN DE LA SEGUNDA MATRIZ: COSTE DE CONCATENACIÓN
        """NOTA: NO SE INCORPORA TODAVÍA NINGÚN PESO A LA DISTANCIA CEPSTRAL NI LAS DISTANCIAS
        SE ENCUENTRAN NORMALIZADAS"""
        
        """
        #Inicializamos una matriz auxiliar para las distancias cepstrales y otra para concatenación
        dist_test_cepstral = np.zeros((ind_test.shape[0],ind_test.shape[1])) 
        cost_matrix = np.zeros((ind_test.shape[0],ind_test.shape[1]))
        ind_vecino_optimo = np.zeros((ind_test.shape[0],1))
        
        #Inicializamos la matriz que va a contener las units de mfcc definitivas
        units_mfcc_final = np.zeros((units_test_mfcc.shape[0],units_test_mfcc.shape[1]))
        
        #Un doble bucle rellena ambas matrices
        for j in range(ind_test.shape[0]):
            
            for k in range(ind_test.shape[1]):
                
                indice = ind_test[j,k]  
                dist_test_cepstral[j,k] = np.linalg.norm(units_test_mfcc[j,:] - units_train_mfcc[ind_test[j,k],:])
                
                #Establecemos que en la primera iteración no se tomen distancias cepstrales
                if (j == 0) : 
                    
                    cost_matrix[j,k] = dist_test[j,k] #Para la primera fila, solo distancias       
            
                elif (j != 0) :   
                    
                    #Rellenamos las unidades finales con el resultado de la fila anterior
                    ind_vecino_optimo = np.argmin(cost_matrix[j-1,:]) #Indice del mejor vecino
                    ind_basedatos_optimo = ind_test[j-1,ind_vecino_optimo] #Indice de la unit en base de datos
                    units_mfcc_final[j-1,:] = units_train_mfcc[ind_basedatos_optimo] #Resultado final de la unit óptima
                    
                    #Completamos la matriz de costes
                    ind_basedatos = ind_test[j,k] #Indice de la unit en base de datos actual
                    cost_matrix[j,k] = dist_test[j,k] + np.linalg.norm(units_train_mfcc[ind_basedatos] - units_train_mfcc[ind_basedatos_optimo])
                
                    
        #Completamos el último elemento de la matriz que no se ha rellenado  
        ind_vecino_optimo = np.argmin(cost_matrix[ind_test.shape[0]-1,:]) #Indice del mejor vecino
        ind_basedatos_optimo = ind_test[ind_test.shape[0]-1,ind_vecino_optimo] #Indice de la unit en base de datos
        units_mfcc_final[ind_test.shape[0]-1,:] = units_train_mfcc[ind_basedatos_optimo] #Resultado final de la unit óptima
        """
        
        #Deshacemos la transformación
        units_mfcc_final = scaler_mfcc.inverse_transform(units_mfcc_final)

        
        #Guardamos los mfcc que se han obtenido por medio de regresión lineal
        np.save(os.path.join(filepath_mfcc_synthesized,name_test), units_mfcc_final)
 


