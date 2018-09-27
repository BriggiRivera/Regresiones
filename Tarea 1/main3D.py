import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plot
import imageio
import os
import random

"""
Entrada del proceso:
    - puntos: Conjunto de Puntos
    - thetas: conjunto de thetas
    - umbral: margen de error
    - alfa: parametro de aprendizaje
    
Salida del proceso:
    - Archvo gif
"""

def crearPuntosDePlano(X, m, thetas):
    
    celdasMalla = 25
    
    mallaX1 = np.linspace(min(X[1].A1), max(X[1].A1), celdasMalla)
    mallaX2 = np.linspace(min(X[2].A1), max(X[2].A1), celdasMalla)
    
    plane = np.meshgrid(mallaX1, mallaX2)
    
    Hx = np.zeros((celdasMalla, celdasMalla))
    
    Hx = Hx + thetas[0]
    
    for index in range(1, len(thetas)):
        Hx = Hx + plane[index-1] * thetas[index]
        
    plane.append(Hx)
    
    return plane
  

def crearImagen(X, Y, thetas, contador):
    
    nombreArchivo = "plot{0}.png".format(str(contador).zfill(4))
    
    fig = plot.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot3D(X[1].A1, X[2].A1, Y, 'o', label='datos', color='red')
    
    plane = crearPuntosDePlano(X, len(Y), thetas)
    
    ax.plot_wireframe(plane[0], plane[1], plane[2], color='slategray')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    ax.set_title('Regresión con plano - {0}'.format(contador))
    
    ax.grid()
    ax.legend()
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
    
def calcularError(X, Y, thetas, m):
    Hx = np.dot(thetas, X)
    return np.sum(np.power(Y-Hx,2)) / (2*m)

def regresionLineal(puntos, thetas, umbral, alfa):
    archivos = []
    m = len(puntos[0])
    X = np.matrix([[1] * m, puntos[0], puntos[1]])
    Y = np.array(puntos[2])
    thetas = np.array(thetas)
    
    contador = 0
    
    while calcularError(X, Y, thetas, m) > umbral and contador < 200:
        Hx = np.dot(thetas, X)
        
        for index in range(0,len(thetas)):
            thetas[index] = thetas[index] - alfa * np.sum((Y-Hx) * -1 * X[index].T)/m
        
        archivos.append(crearImagen(X, Y, thetas, contador))
        contador = contador + 1
        
    crearGif(archivos)
    

def main(argv):
    
    puntos = [ 
            [200*random.random() for i in range(200)], 
            [200*random.random() for i in range(200)], 
            [50*random.random() for i in range(200)]
            ]
    thetas = [10, -0.5, 0.1]
    umbral = 50
    alfa = 0.000005
    
    regresionLineal(puntos, thetas, umbral, alfa)
    

    pass

if __name__ == "__main__":
    main(sys.argv)# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

