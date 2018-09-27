# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plot
import imageio
import os

"""
Entrada del proceso:
    - puntos: Conjunto de Puntos
    - thetas: conjunto de thetas
    - umbral: margen de error
    - alfa: parametro de aprendizaje
    
Salida del proceso:
    - Archvo gif
"""

def crearImagen(X, Y, Hx, contador):
    
    nombreArchivo = "plot{0}.png".format(str(contador).zfill(4))
    
    plot.plot(X, Y, 'o', label='datos')
    plot.plot(X, Hx, label='ajuste')
    plot.xlabel('x')
    plot.ylabel('y')
    plot.title('Mi primera regresion lineal')
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
    
def regresionLineal(puntos, thetas, umbral, alfa):
    m = len(puntos[0])
    unos = [1] * m
    X = np.matrix([unos, puntos[0]])
    Y = np.array(puntos[1])
    thetas = np.array(thetas)
    Hx = np.dot(thetas, X)
    error = Y-Hx
    mean_sq_er = np.sum(np.power(error,2))
    mean_sq_er = mean_sq_er/m
        
    contador = 0

    archivos = []
    
    while mean_sq_er > umbral and contador < 1000:
        error = Y-Hx
        mean_sq_er = (np.sum(np.power(error,2)))/(2*m)
        
        for index in range(0,len(thetas)):
            thetas[index] = thetas[index] - alfa * np.sum(error * -1 * X[index].T)/m
        
        Hx = np.dot(thetas, X)
        
        archivos.append(crearImagen(X[1].T, Y.T, Hx.T, contador))
               
        contador = contador + 1
        pass
    
    crearGif(archivos)
    
    pass


def main(argv):
    puntos = [ [1, 3, 3, 4, 6, 7, 9, 11, 12], [2, 2, 2, 3, 4, 5, 6, 8, 10] ]
    thetas = [-4, -0.8]
    umbral = 0.5
    alfa = 0.0004
    
    regresionLineal(puntos, thetas, umbral, alfa)
    
    pass

if __name__ == "__main__":
    main(sys.argv)# -*- coding: utf-8 -*-

