# Estoy suponiendo que la memoria de cada SM está estructurada con los campos que me interesan.
# Es poco realista, pero aceptable para esta simulación.


class SMMemory:
    def __init__(self, tam: int) -> None:
        self.datos = [0.0] * tam
        self.tam_bloque = 0
        self.ini_bloque = 0
        #Añadimos este atributo para decir a los núcleos cuando deben de acabar su ejecución
        self.terminado = False
