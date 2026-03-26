from threading import Barrier
from gpu_memory import GPUMemory
from sm_memory import SMMemory

#Dejamos estas importaciones por si se quieren descomentar las observaciones del código en producto escalar

import time
import random

def incr(id_nucleo: int, gpu_mem: GPUMemory, sm_mem: SMMemory, _: Barrier) -> None:
    idx = sm_mem.ini_bloque + id_nucleo
    if id_nucleo < sm_mem.tam_bloque:
        gpu_mem.res[idx] = gpu_mem.dato1[idx] + 1


def sumar(id_nucleo: int, gpu_mem: GPUMemory, sm_mem: SMMemory, _: Barrier) -> None:
    if id_nucleo < sm_mem.tam_bloque:
        idx = sm_mem.ini_bloque + id_nucleo
        gpu_mem.res[idx] = gpu_mem.dato1[idx] + gpu_mem.dato2[idx]


# FUNCIÓN DIFUMINAR GENERALIZADA PARA UN RADIO R

'''
Aunque podíamos haberlo hecho más sencillo añadiendo un "halo", hemos decidido que si el elemento se encuentra en una posición en la 
que su intervalo de radio r se sale del vector (ya sea por la izquierda o por la derecha), haremos la media solamente con los elementos 
que están en el vector original. 
'''

def difuminar(id_nucleo: int, gpu_mem: GPUMemory, sm_mem: SMMemory, barrera: Barrier) -> None:
    #Realizamos un copiado de la memoria global a la memoria local

    #FASE DE COPIADO
    radio = gpu_mem.radio_difuminar.value # radio genérico
    idx = sm_mem.ini_bloque + id_nucleo #indice global

    #el primer nucleo del bloque copiará el primer elemento de bloque y los r anteriores (a menos que sea el borde izquierdo del vector)
    #los copiará en las primeras posiciones de la memoria local (por tanto los datos estarán movidos r elementos a la derecha en la memoria local)
    if id_nucleo == 0:
        r = 1
        while r <= radio:
            if idx >= r: #comprobación de que no se trata de un caso de la frontera
                sm_mem.datos[radio - r] = gpu_mem.dato1[idx - r]
            r += 1

    #el último núcleo del bloque trabajará con el último elemento del bloque y los r posteriores (a menos que sea el borde derecho del vector)
    #los copiará en las últimas posiciones de la memoria local
    if id_nucleo == sm_mem.tam_bloque - 1:
        r = 1
        while r <= radio:
            if idx + r < gpu_mem.tam_datos.value: #comprobación de que no se trata de un caso de la frontera
                sm_mem.datos[(id_nucleo + radio) + r] = gpu_mem.dato1[idx+r]
            r += 1

    #desplazamos r posiciones los elementos del bloque
    if id_nucleo < sm_mem.tam_bloque:
        sm_mem.datos[id_nucleo+ radio] = gpu_mem.dato1[idx]

    barrera.wait() #Esperamos a que todos los procesos acaben de copiar su elemento en la memoria local para continuar

    #FASE DE CÁLCULO
    #Tras el copiado ahora trabajamos en la memoria local
    if id_nucleo < sm_mem.tam_bloque:
        suma = 0.0
        nuevo_centro = id_nucleo + radio
        num_elementos = 0

        for i in range(-radio, radio+1):
            componente = nuevo_centro + i
            if 0 <= idx+i < gpu_mem.tam_datos.value: #solo contaremos los elementos del propio vector (no trabajamos con halo)
                suma += sm_mem.datos[componente]
                num_elementos += 1
        gpu_mem.res[idx] = suma / num_elementos  # Actualizamos en la memoria global


'''
En clase observamos que sin necesidad de usar el Lock del Value nos funcionaba el código porque la operación era muy rápida  
Surgió la duda de si realmente nuestro programa estaba trabajando en paralelo, así que hemos añadido unas líneas comentadas 
para que, en caso de que se desee, se pueda comprobar que realmente sí se está trabajando con paralelismo.
A parte vimos que si poníamos un sleep para aumentar el tiempo en el que toma la suma total y le suma la parcial del bloque, el resultado
cambiaba si no usábamos el lock del Value con el with, demostrando una vez más que se está trabajando con paralelismo.    
'''

def escalar(id_nucleo: int, gpu_mem: GPUMemory, sm_mem: SMMemory, barrera: Barrier) -> None:
    idx = sm_mem.ini_bloque + id_nucleo #índice global
    # Cada núcleo multiplica su par de datos y lo guarda en su posición local
    if id_nucleo < sm_mem.tam_bloque:
        sm_mem.datos[id_nucleo] = gpu_mem.dato1[idx] * gpu_mem.dato2[idx]

    barrera.wait() # esperamos a que todos hayan escrito su resultado en la memoria local del sm

    # Solo el núcleo 0 hace la suma del bloque
    if id_nucleo == 0:
        suma_bloque = 0.0
        for i in range(sm_mem.tam_bloque):
            suma_bloque += sm_mem.datos[i]

        #Podemos añadir la siguiente línea para, efectivamente, ver que se está trabajando en paralelo
        #time.sleep(random.uniform(0.5, 2.0))

        with gpu_mem.prod_escalar.get_lock(): #accedemos de manera segura
            gpu_mem.prod_escalar.value += suma_bloque #sumamos al total la suma parcial del bloque

            #Print para la comprobación de paralelismo
            #print(f"Bloque {idx}: suma {suma_bloque} -> Total: {gpu_mem.prod_escalar.value}")

INCR = 1
SUMAR = 2
DIFUMINAR = 3
ESCALAR = 4

KERNELS = {
    INCR: incr,
    SUMAR: sumar,
    DIFUMINAR: difuminar,
    ESCALAR: escalar,
}