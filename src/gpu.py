from multiprocessing import Queue
from sm import SM
from gpu_memory import GPUMemory
import random


if __name__ == '__main__':
    # parámetros de construcción de la GPU
    cant_sms = 4
    cant_nucleos_por_sm = 5
    tam_mem_gpu = 1_000
    mem_gpu = GPUMemory(tam_mem_gpu)
    tam_mem_sm = 100
    CENTINELA = None
    q_bloques = Queue()

    #AQUÍ ELEGIMOS QUE KERNEL EJECUTAR
    # 1: INCR | 2: SUMAR | 3: DIFUMINAR | 4: ESCALAR (Producto Escalar)
    KERNEL_A_PROBAR = 4

    #Pasamos como parametro de entrada el radio de difuminación
    #PRECONDICIÓN: radio >= 0
    mem_gpu.radio_difuminar.value = 2

    mem_gpu.kernel.value = KERNEL_A_PROBAR
    valor = 501  # Tamaño del array de prueba
    mem_gpu.tam_datos.value = valor

    # DATOS DE ENTRADA
    mem_gpu.dato1[:valor] = [float(random.randint(-10, 10)) for i in range(valor)]
    mem_gpu.dato2[:valor] = [2.0] * valor  # [2, 2, 2, 2...]

    #CREAMOS LAS SMs Y CREAMOS LA COLA CON LOS BLOQUES DEL VECTOR CORRESPONDIENTE
    sms = [SM(cant_nucleos_por_sm, mem_gpu, tam_mem_sm, q_bloques)
           for _ in range(cant_sms)]
    for s in sms:
        s.start()
    for block_start in range(0, mem_gpu.tam_datos.value, cant_nucleos_por_sm):
        block_size = min(cant_nucleos_por_sm, mem_gpu.tam_datos.value - block_start)
        q_bloques.put((block_start, block_size))
    for _ in range(cant_sms):  # añadimos un CENTINELA por SM
        q_bloques.put(CENTINELA)
    # Espera a que todos los SMs acaben para dar la solución
    for sm in sms:
        sm.join()

    '''
    RESULTADOS (añadimos una comprobación generada por la IA de los resultados, la cual se puede omitir para mejorar los costes en tiempo)
    '''
    print()
    print('Dato1', mem_gpu.dato1[:valor])
    print('Dato2', mem_gpu.dato2[:valor])

    print("=" * 40)

    if KERNEL_A_PROBAR == 1:
        print('Incremento (GPU)  :', mem_gpu.res[:valor]) #NUESTRO VALOR CALCULADO PARALELAMENTE

        #IA
        esperado = [x + 1 for x in mem_gpu.dato1[:valor]]
        print('Resultado esperado:', esperado)

    elif KERNEL_A_PROBAR == 2:
        print('Sumar (GPU)       :', mem_gpu.res[:valor]) #NUESTRO VALOR CALCULADO PARALELAMENTE

        #IA
        esperado = [mem_gpu.dato1[i] + mem_gpu.dato2[i] for i in range(valor)]
        print('Resultado esperado:', esperado)

    elif KERNEL_A_PROBAR == 3:
        print('Difuminar (GPU)   :', mem_gpu.res[:valor]) #NUESTRO VALOR CALCULADO PARALELAMENTE

        #IA
        esperado = []
        for i in range(valor):
            inicio = max(0, i - mem_gpu.radio_difuminar.value)
            fin = min(valor, i + 1 + mem_gpu.radio_difuminar.value)
            vecinos = mem_gpu.dato1[inicio:fin]
            media = sum(vecinos) / len(vecinos)
            esperado.append(media)
        print('Resultado esperado:', esperado)

    elif KERNEL_A_PROBAR == 4:
        print(f'Producto escalar (GPU): {mem_gpu.prod_escalar.value}') #NUESTRO VALOR CALCULADO PARALELAMENTE

        #IA
        esperado = sum(mem_gpu.dato1[i] * mem_gpu.dato2[i] for i in range(valor))
        print(f'Resultado esperado    : {esperado}')