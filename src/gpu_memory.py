from multiprocessing import Array, Value

# Estoy suponiendo que la memoria de la GPU está estructurada con los campos que me interesan.
# Es poco realista, pero aceptable para esta simulación.

# Para mayor precisión de los resultados, ponemos los arrays de tipo double (no simplemente float)
class GPUMemory:
    def __init__(self, tam_max: int) -> None:
        self.tam_max = Value('i', tam_max)
        self.dato1 = Array('d', tam_max)
        self.dato2 = Array('d', tam_max)
        self.res = Array('d', tam_max)
        self.prod_escalar = Value('d', 0)
        self.tam_datos = Value('i', 0)
        self.kernel = Value('i', 0)
        self.radio_difuminar = Value('i', 0)