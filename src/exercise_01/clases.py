class Brocha:
    def __init__(self, Color: str, tamannio: int):
        self.color = Color
        self.tamannio = tamannio

        def pintar(self, color: str, tamannio: str):
            print(f"Pintando con brocha de color {self.color} y tamaño {self.tamannio}")

        def guardar_dibujo(self, nombre_archivo: str):
            print(f"Guardando dibujo en el archivo {nombre_archivo}" )

if __name__ == "__main__":
    brocha_azul = Brocha("azul", 5)
    brocha_verde_gigante = Brocha("verde", 10)
    brocha_azul_pequeña = Brocha("azul", 2)

    brocha_azul.pintar()