from threading import Thread, Barrier
from gpu_memory import GPUMemory
from sm_memory import SMMemory
import kernels

class Nucleo(Thread):
    def __init__(self, indice: int , gpu_mem: GPUMemory, sm_memory: SMMemory, barrera_inicio : Barrier, barrera_final : Barrier, barrera_interna : Barrier)-> None:
        super().__init__()
        self.indice = indice
        self.gpu_mem = gpu_mem
        self.sm_memory = sm_memory
        self.barrera_inicio = barrera_inicio
        self.barrera_final = barrera_final
        self.barrera_interna = barrera_interna


    def run(self) -> None:
        funcion = kernels.KERNELS[self.gpu_mem.kernel.value] # elegimos la función a ejecutar
        self.barrera_inicio.wait()  # espera a que el SM le asigne un bloque

        while not self.sm_memory.terminado:
            funcion(self.indice, self.gpu_mem, self.sm_memory, self.barrera_interna)
            self.barrera_final.wait()  # avisa al SM de que ha terminado (la SM estaba esperando)
            self.barrera_inicio.wait() # el nucleo espera a que el SM coja el siguiente bloque de la cola

