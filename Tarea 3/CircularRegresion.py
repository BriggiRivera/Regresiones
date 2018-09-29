# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plot
import imageio
import random
import os
import math

"""
Entrada del proceso:
    - puntos: Conjunto de Puntos
    - thetas: conjunto de thetas
    - umbral: margen de error
    - alfa: parametro de aprendizaje
    
Salida del proceso:
    - Archvo gif
"""

def crearImagenCirculo(puntos, cx, cy, R, contador):
    
    nombreArchivo = "circulo{0}.png".format(str(contador).zfill(4))
    
    plot.plot(puntos[0], puntos[1], '.')
    circle = plot.Circle((cx, cy), R, color='r', fill=False)
    plot.gca().add_artist(circle)
    plot.title('Circulo')
    plot.axis([-0, 50, 0, 50])
    
    plot.savefig(nombreArchivo)
    plot.clf()
    
    return nombreArchivo

def crearImagen(X, Y, Hx, contador):
    
    nombreArchivo = "plot{0}.png".format(str(contador).zfill(4))
    
    plot.plot(X, Y, 'o', label='datos')
    plot.plot(X, Hx, label='ajuste')
    plot.xlabel('x')
    plot.ylabel('y')
    plot.title('regresion circular {0}'.format(contador))
    plot.grid()
    plot.legend()
    plot.savefig(nombreArchivo)
    plot.clf()
    
    return nombreArchivo
    
def crearGif(imagenes):
    listaArchivos = []
    for imagen in imagenes:
        listaArchivos.append(imageio.imread(imagen))
    imageio.mimsave("resultado.gif", listaArchivos)
    
    for imagen in imagenes:
        os.remove(imagen)
    pass
    
def regresionLineal(X, Y, thetas, umbral, alfa, puntos, cx, cy):
    
    Xreal = np.array(puntos[0])
    Yreal = np.array(puntos[1])
    
    m = len(Y)
    unos = np.array([1] * m)
    X = (unos, X)
    thetas = np.array(thetas)
    Hx = np.dot(thetas, X)
    error = Y-Hx
    mean_sq_er = np.sum(np.power(error,2))
    mean_sq_er = mean_sq_er/m
        
    contador = 0

    archivos = []
    
    while mean_sq_er > umbral and contador < 100:
        
        error = Y-Hx
        
        mean_sq_er = (np.sum(np.power(error,2)))/(2*m)
        
        thetas[0] = thetas[0] - alfa * np.sum(error * -1 * X[0].T)/m
        
        Hx = np.dot(thetas, X)
        
        print(mean_sq_er)
        
        #archivos.append(crearImagen(X[1].T, Y.T, Hx.T, contador))
        archivos.append(crearImagenCirculo(puntos, cx, cy, thetas[0], contador)) 
        
        contador = contador + 1
        
        Y = np.sqrt((Xreal-cx)**2 + (Yreal-cy)**2)-thetas[0]    
        
        pass
    
    crearGif(archivos)
    
    pass


def obtenerPuntos(caso):
    
    if caso == 1:
        return [ 
                [2, 3, 2, 2, 2, 4, 3, 7, 9, 9.5, 9.2, 9, 7, 5, 5, 6  , 7, 7, 6, 5], 
                [2, 3, 4, 6, 8, 9, 2, 2, 3, 5  , 7  , 9,10,10, 7, 7.5, 7, 6, 5, 6] 
            ]
    elif caso == 2:
        return [
                [40*random.random() for i in range(100)], 
                [40*random.random() for i in range(100)]
            ]
    else:
        
        return [ 
                [10, 20, 20, 30, 12], 
                [20, 10, 30, 20, 25] 
            ]
    pass


def main(argv):
    puntos = obtenerPuntos(3)
    
    X = np.array(puntos[0])
    Y = np.array(puntos[1])
    
    cx = np.average(X)
    cy = np.average(Y)
    
    thetas = [random.randint(1, 50), 0]
    umbral = 20
    alfa = 0.001
    
    X = np.array([ indice for indice in range(1,len(X)+1) ])
    Y = np.sqrt((X-cx)**2 + (Y-cy)**2)-thetas[0]
    
    print(X)
    print(Y)
    
    regresionLineal(X, Y, thetas, umbral, alfa, puntos, cx, cy)
    
    pass

if __name__ == "__main__":
    main(sys.argv)# -*- coding: utf-8 -*-


