# -*- coding: utf-8 -*-
import sys
import numpy as np
import random

def sigmoide(thetas, X):
    
    Hx = np.dot(thetas.T, X).A1
    
    for index in range(len(Hx)):
        Hx[index] = 1.0 / (1.0 + np.exp(-Hx[index]))
    
    return Hx

def calcularError(m, X, Y, thetas, Sx, lamb):
    return -1/m * np.sum(Y * np.log(Sx) + (1-Y) * np.log(1-Sx.T) ) + lamb / (2*m) * np.sum(thetas**2)

def actualizarThetas(m, X, Y, thetas, Sx, alfa):
    for index in range(0,len(thetas)):
        thetas[index] = thetas[index] - alfa * np.sum((Y-Sx) * -1 * X[index].A1.T)/m
    return thetas

def regresionLogisticaCrearModelo(X, Y, thetas, umbral, alfa, lamb):
    m = len(X[0])
    X = [[1] * m] + X
    X = np.matrix(X)
    Y = np.array(Y)
    thetas = np.array(thetas)
    Sx = sigmoide(thetas, X)
    
    print(calcularError(m, X, Y, thetas, Sx, lamb))
    cont = 0
    while calcularError(m, X, Y, thetas, Sx, lamb) > umbral:
        if cont%10000 == 0:
            print(calcularError(m, X, Y, thetas, Sx, lamb))
        thetas = actualizarThetas(m, X, Y, thetas, Sx, alfa)
        Sx = sigmoide(thetas, X)
        cont = cont + 1
    return thetas

# Esta función extrae del contenido del archivo cargado el array que indica
# si hay cancer (1) o no hay cancer (0)
def obtenerY(contenido):
    Y = [0 if fila[0].startswith('no') else 1 for fila in contenido]
    return Y

def obtenerPosicion(campo, arreglo):
    return [pos for pos in range(len(arreglo)) if campo == arreglo[pos]][0]
    
def obtenerX(contenido):
    
    age       = ['?', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
    menopause = ['?', 'lt40', 'ge40', 'premeno']
    tumorSize = ['?', '0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59']
    invNodes  = ['?', '0-2', '3-5', '6-8', '9-11', '12-14','15-17', '18-20', '21-23', '24-26', '27-29', '30-32', '33-35', '36-39']
    nodeCaps  = ['?', 'yes', 'no']
    degMalig  = ['?', '1', '2', '3']
    breast    = ['?', 'left', 'right']
    breastQuad= ['?', 'left_up', 'left_low', 'right_up', 'right_low', 'central']
    irradiat  = ['?', 'yes', 'no']
    
    X = [
            [obtenerPosicion(fila[1], age) for fila in contenido],
            [obtenerPosicion(fila[2], menopause) for fila in contenido],
            [obtenerPosicion(fila[3], tumorSize) for fila in contenido],
            [obtenerPosicion(fila[4], invNodes) for fila in contenido],
            [obtenerPosicion(fila[5], nodeCaps) for fila in contenido],
            [obtenerPosicion(fila[6], degMalig) for fila in contenido],
            [obtenerPosicion(fila[7], breast) for fila in contenido],
            [obtenerPosicion(fila[8], breastQuad) for fila in contenido],
            [obtenerPosicion(fila[9], irradiat) for fila in contenido]
    ]
    
    return X

def obtenerAleatoriosPorPorcentaje(porcentaje, total):
    cantidad = (int)(total * porcentaje / 100) 
    restantes = [numero for numero in range(0, total)]
    aleatorios = []
    
    for contador in range(0, cantidad):
        elegido = random.randint(0, len(restantes)-1)
        aleatorios.append(restantes[elegido])
        restantes.remove(restantes[elegido])
        
    return [aleatorios, restantes]

def cargarInformacionArchivo(nombreArchivo):
    filas = []
    
    # Abrimos el archivo    
    with open(nombreArchivo) as archivo:
        # Cargamos todas las lineas del archivo en una lista
        filas = archivo.readlines()
    
    # Aquí parto cada fila por las comas y genero una lista con los campos obtenidos
    # La función strip elimina el salto de linea al final de cada linea
    return [fila.strip().split(',') for fila in filas]
    
def imprimirPruebas(X, Y, thetas, contenido):
    m = len(Y)
    X = [[1] * m] + X
    X = np.matrix(X)
    Y = np.array(Y)
    
    Sx = sigmoide(X, thetas)
    
    total = 0
    
    for indice in range(0, m):
        
        if Sx[indice] >= 0.5:
            if Y[indice] == 1:
                print("Correcto: ",Sx[indice], "\t-> ",contenido[indice])
                total = total + 1
            else:
                print("Falló: ",Sx[indice], "\t-> ",contenido[indice])
        else:
            if Y[indice] == 0:
                total = total + 1
                print("Correcto: ",Sx[indice], "\t-> ",contenido[indice])
            else:
                print("Falló: ",Sx[indice], "\t-> ",contenido[indice])
    
    print(total/m*100)
    
    pass

def main(argv): 
    contenido = cargarInformacionArchivo("breast-cancer.data")
    pruebas, entrenamiento = obtenerAleatoriosPorPorcentaje(20, len(contenido))
    contenidoPruebas = [contenido[posicion] for posicion in range(len(contenido)) if posicion in pruebas]
    contenidoEntrenamiento = [contenido[posicion] for posicion in range(len(contenido)) if posicion in entrenamiento]
    
    XEntrenamiento = obtenerX(contenidoEntrenamiento)
    YEntrenamiento = obtenerY(contenidoEntrenamiento)
    
    thetas = [random.random() for i in range(10)]
    
    umbral = 0.55
    alfa = 0.1
    lamb = 0.1
    
    XPruebas = obtenerX(contenidoPruebas)
    YPruebas = obtenerY(contenidoPruebas)
    
    thetas = regresionLogisticaCrearModelo(XEntrenamiento, YEntrenamiento, thetas, umbral, alfa, lamb);
    print("+++++++++++++++++++++++++++++++++++")
    imprimirPruebas(XPruebas, YPruebas, thetas, contenidoPruebas)
    
    pass

if __name__ == "__main__":
    main(sys.argv)