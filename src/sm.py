from multiprocessing import Process , Queue
from nucleo import Nucleo
from gpu_memory import GPUMemory
from threading import Barrier
from sm_memory import SMMemory

class SM(Process):

    def __init__(self,cant_nucleos_por_sm : int , mem_gpu : GPUMemory, tam_mem_sm : int , q_bloques : Queue) -> None:
        super().__init__()
        self.cant_nucleos_por_sm = cant_nucleos_por_sm
        self.mem_gpu = mem_gpu
        self.tam_mem_sm = tam_mem_sm
        self.q_bloques = q_bloques

    def run(self) -> None:
        #Creamos la memoria del sm
        sm_memory = SMMemory(self.tam_mem_sm)

        #Podriamos crear una única barrera para barrera_inicio y barrera_final pero preferimos dejar las dos para aportar claridad al código

        #BARRERA con objetivo: garantizar que ningún núcleo empiece hasta que el SM haya preparado el bloque de datos completo
        barrera_inicio = Barrier(self.cant_nucleos_por_sm+1) #Participan todos los nucleos del SM y la propia hebra principal
        #BARRERA con objetivo: garantizar que el SM no pida un bloque nuevo (sobreescribiendo la info local) hasta que todos los núcleos hayan acabado con el bloque anterior
        barrera_final = Barrier(self.cant_nucleos_por_sm+1)
        #BARRERA con objetivo: ser utilizada en kernels como difuminar o escalar para evitar las condiciones de carrera
        barrera_interna  = Barrier(self.cant_nucleos_por_sm) #Participan solo los nucleos de la SM
        #CREAMOS LOS NÚCLEOS
        nucleos = [Nucleo(indice, self.mem_gpu, sm_memory, barrera_inicio, barrera_final, barrera_interna) for indice in range(self.cant_nucleos_por_sm)] #índice es el id de cada nucleo

        for n in nucleos:
            n.start()

        bloque = self.q_bloques.get()
        while bloque is not None: # Mientras el bloque no sea un centinela
            #cada bloque guarda su posición global de inicio y el tamaño de este mismo (será importante para el último bloque, que puede no ser de tamaño máximo)
            ini_bloque = bloque[0]
            tam_bloque = bloque[1]

            sm_memory.ini_bloque = ini_bloque # La primera componente del bloque es la posición de inicio
            sm_memory.tam_bloque = tam_bloque # La segunda componente del bloque es el tamaño del bloque

            barrera_inicio.wait() # Ya he creado el bloque completo (los núcleos estaban esperando)
            barrera_final.wait() # El SM espera a que los nucleos terminen de trabajar
            bloque = self.q_bloques.get()

        sm_memory.terminado = True # Decimos a los nucleos que no trabajen más
        barrera_inicio.wait() # Espera a que lleguen todos y ya acaba