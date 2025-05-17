import cv2
import numpy as np
from tkinter import Tk, Button, filedialog, Canvas, simpledialog, Toplevel
from PIL import Image, ImageTk

# Cargar el clasificador Haar Cascade para detectar rostros
# Asegúrate de que 'haarcascade_frontalface_default.xml' está en el directorio donde se ejecuta el script,
# o que cv2.data.haarcascades apunta a la ubicación correcta.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Función para detectar rostros con Viola-Jones
def detectar_rostros(imagen):
    """
    Detecta rostros en una imagen usando el clasificador Haar Cascade.

    Args:
        imagen (np.ndarray): Imagen de entrada en formato BGR de OpenCV.

    Returns:
        list: Lista de rectángulos [x, y, w, h] que contienen los rostros detectados.
    """
    if imagen is None or imagen.size == 0:
        print("Error en detectar_rostros: imagen de entrada vacía.")
        return []
        
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # minNeighbors: Cuántos vecinos debe tener cada rectángulo candidato para conservarlo.
    # minSize: Tamaño mínimo posible para el objeto. Objetos más pequeños son ignorados.
    rostros = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return rostros

# Función para aplicar un filtro pasa bajos en el dominio frecuencial usando FFT
# NOTA: Esta función se mantiene del código original pero NO se usa para el
# efecto principal de suavizado del "filtro de belleza" con máscara.
# Es solo para visualización o propósitos de comparación.
def filtro_pasa_bajos(imagen_color_o_gris, cutoff=30):
    """
    Aplica un filtro pasa bajos a una imagen (preferiblemente gris) en el dominio frecuencial.

    Args:
        imagen_color_o_gris (np.ndarray): Imagen de entrada (BGR o Gris).
        cutoff (int): Frecuencia de corte. Frecuencias por debajo de este valor pasan.

    Returns:
        np.ndarray: Imagen resultante después de aplicar el filtro pasa bajos (en gris, 0-255).
    """
    if imagen_color_o_gris is None or imagen_color_o_gris.size == 0:
        print("Error en filtro_pasa_bajos: imagen de entrada vacía.")
        return np.zeros((10,10), dtype=np.uint8)

    if len(imagen_color_o_gris.shape) == 3:
        imagen_gris = cv2.cvtColor(imagen_color_o_gris, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gris = imagen_color_o_gris.copy()

    f = np.fft.fft2(imagen_gris)
    fshift = np.fft.fftshift(f)
    rows, cols = imagen_gris.shape
    crow, ccol = rows // 2, cols // 2

    # Crear la máscara de filtro pasa bajos (círculo en el centro)
    mask = np.zeros((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            # Distancia al centro
            if np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) <= cutoff:
                mask[i, j] = 1

    # Aplicar la máscara en el dominio frecuencial
    fshift_filtrado = fshift * mask

    # Transformada inversa
    f_ishift = np.fft.ifftshift(fshift_filtrado)
    img_reconstruida_complex = np.fft.ifft2(f_ishift)
    img_reconstruida_abs = np.abs(img_reconstruida_complex)

    # Normalizar al rango 0-255
    if np.max(img_reconstruida_abs) > 0:
        img_reconstruida_norm = (img_reconstruida_abs / np.max(img_reconstruida_abs)) * 255.0
    else:
        img_reconstruida_norm = img_reconstruida_abs # Opcional: manejar caso de imagen completamente negra

    return np.uint8(img_reconstruida_norm)

# Función de segmentación de rostro usando el crecimiento de regiones
def crecimiento_de_regiones(imagen_gris_rostro, semilla_en_rostro, umbral):
    """
    Realiza crecimiento de regiones en una imagen en escala de grises a partir de una semilla.

    Args:
        imagen_gris_rostro (np.ndarray): Imagen en escala de grises (recorte del rostro).
        semilla_en_rostro (tuple): Coordenadas (x, y) de la semilla DENTRO del recorte del rostro.
        umbral (int): Máxima diferencia absoluta permitida para incluir un píxel en la región.

    Returns:
        np.ndarray: Máscara binaria (0 o 255) del área segmentada.
    """
    if imagen_gris_rostro is None or imagen_gris_rostro.size == 0:
        print("Error en crecimiento_de_regiones: imagen de entrada vacía.")
        return np.zeros((10,10), dtype=np.uint8)

    rows, cols = imagen_gris_rostro.shape
    mascara_resultado = np.zeros((rows, cols), np.uint8)

    semilla_x_rel, semilla_y_rel = semilla_en_rostro[0], semilla_en_rostro[1]

    if not (0 <= semilla_y_rel < rows and 0 <= semilla_x_rel < cols):
        print(f"Error: Semilla relativa ({semilla_x_rel},{semilla_y_rel}) fuera de los límites de la imagen del rostro ({cols},{rows}).")
        return mascara_resultado # Retorna máscara vacía

    valor_pixel_semilla = int(imagen_gris_rostro[semilla_y_rel, semilla_x_rel])

    # Implementación simple de crecimiento de regiones iterando sobre píxeles.
    # Para implementaciones más eficientes se usarían colas o stacks para explorar vecinos.
    # Esta versión itera sobre todos los píxeles y comprueba la diferencia con la semilla.
    # Es menos eficiente que una basada en conectividad pero funciona para áreas compactas.
    for r_idx in range(rows):
        for c_idx in range(cols):
            # Comprobar si la diferencia absoluta con el valor de la semilla es menor que el umbral
            if abs(int(imagen_gris_rostro[r_idx, c_idx]) - valor_pixel_semilla) < umbral:
                mascara_resultado[r_idx, c_idx] = 255 # Marcar como parte de la región

    return mascara_resultado


class FiltroBellezaGUI:
    def __init__(self, main_root_window):
        """
        Inicializa la interfaz gráfica.

        Args:
            main_root_window (Tk): Ventana principal de Tkinter.
        """
        self.root = main_root_window
        self.root.title("Filtro de Belleza")
        self.img_tk_semilla_ref = None # Referencia para evitar que la imagen de Tkinter sea recolectada por el recolector de basura
        self.imagen_cv_original_cargada = None # Imagen original cargada por el usuario (BGR)
        self.ventana_semilla_dialogo = None # Referencia a la ventana de selección de semilla

        # Botón para cargar imagen
        Button(self.root, text="Cargar Imagen", command=self.cargar_imagen_para_semilla).pack(pady=20)

    def aplicar_filtro_con_semilla_y_umbral(self, semilla_global_coords, umbral_elegido):
        """
        Aplica el filtro de belleza (suavizado selectivo) basado en la semilla y el umbral.

        Args:
            semilla_global_coords (tuple): Coordenadas (x, y) de la semilla en la imagen original.
            umbral_elegido (int): Umbral para el crecimiento de regiones.
        """
        if self.imagen_cv_original_cargada is None:
            print("No hay imagen cargada para aplicar el filtro.")
            return

        imagen_a_procesar = self.imagen_cv_original_cargada.copy() # Usar una copia para no modificar la original

        rostros_detectados = detectar_rostros(imagen_a_procesar)

        if len(rostros_detectados) > 0:
            # Solo procesamos el primer rostro detectado para simplificar
            x_r, y_r, w_r, h_r = rostros_detectados[0]

            # Recortar la región del rostro de la imagen original
            rostro_recortado_bgr = imagen_a_procesar[y_r:y_r+h_r, x_r:x_r+w_r]

            if rostro_recortado_bgr is None or rostro_recortado_bgr.size == 0:
                print("Error: El rostro recortado está vacío (dimensiones 0).")
                return

            # Calcular las coordenadas de la semilla RELATIVAS al recorte del rostro
            semilla_rostro_x_rel = semilla_global_coords[0] - x_r
            semilla_rostro_y_rel = semilla_global_coords[1] - y_r

            # Verificar si la semilla relativa está dentro de los límites del rostro recortado
            if not (0 <= semilla_rostro_x_rel < w_r and 0 <= semilla_rostro_y_rel < h_r):
                print(f"Advertencia: Semilla global ({semilla_global_coords[0]},{semilla_global_coords[1]}) cae fuera del rostro detectado ({x_r},{y_r},{w_r},{h_r}). Usando el centro del rostro como semilla.")
                semilla_final_para_rostro = (w_r // 2, h_r // 2)
            else:
                semilla_final_para_rostro = (semilla_rostro_x_rel, semilla_rostro_y_rel)

            # Convertir el recorte del rostro a gris para el crecimiento de regiones (que trabaja con intensidad)
            rostro_recortado_gris_para_mascara = cv2.cvtColor(rostro_recortado_bgr, cv2.COLOR_BGR2GRAY)

            # Generar la máscara binaria del área de la piel usando crecimiento de regiones
            # La máscara será 0 o 255
            mascara_binaria_generada = crecimiento_de_regiones(
                rostro_recortado_gris_para_mascara,
                semilla_final_para_rostro,
                umbral_elegido
            )

            # --- Aplicar el filtro de suavizado Bilateral al rostro recortado ---
            # Este filtro suaviza la textura pero preserva los bordes
            # Parámetros:
            # 9: Diámetro del vecindario para el suavizado. Mayor = más suavizado, más lento.
            # 75: Sigma Color. Mayor = píxeles con colores más diferentes se mezclarán.
            # 75: Sigma Espacio. Mayor = píxeles más distantes se influenciarán mutuamente.
            # Puedes experimentar con estos valores.
            rostro_suavizado_bgr = cv2.bilateralFilter(rostro_recortado_bgr, 9, 75, 75)

            # --- Combinar la imagen original del rostro con la imagen suavizada usando la máscara ---
            # Donde la máscara es 255, usamos el píxel de 'rostro_suavizado_bgr'.
            # Donde la máscara es 0, usamos el píxel de 'rostro_recortado_bgr' (original).

            # Asegurarse de que la máscara tiene el mismo número de canales que las imágenes BGR
            # y que es de tipo flotante entre 0.0 y 1.0 para la mezcla ponderada.
            mascara_blend = mascara_binaria_generada.astype(float) / 255.0
            mascara_blend_3_canales = np.stack([mascara_blend] * 3, axis=-1) # Replicar la máscara en los 3 canales BGR

            # Realizar la mezcla lineal: resultado = original * (1-máscara) + suavizado * máscara
            # np.where es otra opción: np.where(mascara_booleana_3_canales, rostro_suavizado_bgr, rostro_recortado_bgr)
            # Pero la mezcla lineal es común para blending suave si la máscara no fuera binaria.
            rostro_resultado_bgr = (rostro_recortado_bgr * (1 - mascara_blend_3_canales) + rostro_suavizado_bgr * mascara_blend_3_canales).astype(np.uint8)


            # --- Mostrar resultados en ventanas de OpenCV ---
            # Muestra el rostro original recortado
            cv2.imshow("1. Rostro Original (Recortado)", rostro_recortado_bgr)

            # Muestra la máscara binaria generada por crecimiento de regiones
            # Esta máscara define el area que sera suavizada
            cv2.imshow(f"2. Máscara Binaria (Define Área, Umbral: {umbral_elegido})", mascara_binaria_generada)
            
            # Opcional: Mostrar el rostro suavizado completo (sin enmascarar) para comparación
            # cv2.imshow("3. Rostro Suavizado Completo (Sin Máscara)", rostro_suavizado_bgr)

            # Muestra el resultado final: rostro original con el area de la máscara suavizada
            cv2.imshow("3. Resultado (Area de Mascara Suavizada con Filtro Bilateral)", rostro_resultado_bgr)


            print("Presiona cualquier tecla en una ventana de OpenCV para continuar...")
            cv2.waitKey(0) # Espera indefinidamente hasta que se presione una tecla
            cv2.destroyAllWindows() # Cierra todas las ventanas de OpenCV

        else:
            print("No se detectaron rostros en la imagen.")
            # Si no hay rostros, cerramos las ventanas de seleccion de semilla/umbral si estan abiertas
            if self.ventana_semilla_dialogo is not None and self.ventana_semilla_dialogo.winfo_exists():
                 self.ventana_semilla_dialogo.destroy()
                 self.img_tk_semilla_ref = None


    def cargar_imagen_para_semilla(self):
        """
        Abre un diálogo para seleccionar una imagen y la muestra en una nueva ventana
        para que el usuario elija un punto semilla.
        """
        path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.png *.bmp")])
        if not path:
            return # El usuario canceló

        # Cargar la imagen con OpenCV (formato BGR por defecto)
        self.imagen_cv_original_cargada = cv2.imread(path)
        if self.imagen_cv_original_cargada is None:
            print(f"Error al cargar la imagen: {path}. Verifica la ruta y el formato.")
            self.imagen_cv_original_cargada = None # Asegurarse de que es None si falla
            return

        # Crear una nueva ventana Toplevel para mostrar la imagen y seleccionar la semilla
        self.ventana_semilla_dialogo = Toplevel(self.root)
        self.ventana_semilla_dialogo.title("Seleccionar Semilla (Clic en la imagen)")
        self.ventana_semilla_dialogo.grab_set() # Hace que esta ventana sea modal (bloquea la principal)

        # Convertir la imagen de OpenCV (BGR) a formato compatible con PIL (RGB)
        img_rgb_para_pil = cv2.cvtColor(self.imagen_cv_original_cargada, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb_para_pil)

        # Convertir la imagen de PIL a formato compatible con Tkinter
        self.img_tk_semilla_ref = ImageTk.PhotoImage(image=img_pil)

        # Crear un Canvas para mostrar la imagen y capturar clics
        canvas_seleccion = Canvas(self.ventana_semilla_dialogo,
                                  width=self.imagen_cv_original_cargada.shape[1],
                                  height=self.imagen_cv_original_cargada.shape[0])
        canvas_seleccion.pack()
        canvas_seleccion.create_image(0, 0, image=self.img_tk_semilla_ref, anchor="nw")

        # Función que se ejecuta al hacer clic en el canvas
        def callback_seleccionar_semilla(event):
            # Las coordenadas del evento son relativas al widget Canvas
            coordenadas_semilla_global = (event.x, event.y)
            print(f"Semilla seleccionada en coordenadas globales: {coordenadas_semilla_global}")

            # Cerrar la ventana de selección de semilla
            self.ventana_semilla_dialogo.destroy()
            self.img_tk_semilla_ref = None # Liberar la referencia de la imagen de Tkinter

            # Pedir al usuario que ingrese el valor del umbral para el crecimiento de regiones
            umbral_ingresado = simpledialog.askinteger(
                "Umbral para Máscara",
                "Introduce el valor del umbral para definir el área de la piel (ej. 10-50):",
                parent=self.root, # La ventana padre es la principal
                minvalue=1,      # Umbral mínimo
                maxvalue=254,    # Umbral máximo (para evitar umbrales que incluyan todo)
                initialvalue=20  # Valor inicial sugerido
            )

            # Si el usuario ingresó un umbral (no canceló el diálogo)
            if umbral_ingresado is not None:
                print(f"Umbral seleccionado: {umbral_ingresado}")
                # Llamar a la función principal para aplicar el filtro
                self.aplicar_filtro_con_semilla_y_umbral(coordenadas_semilla_global, umbral_ingresado)
            else:
                print("Selección de umbral cancelada por el usuario.")


        # Vincular el clic izquierdo del ratón (Button-1) a la función callback
        canvas_seleccion.bind("<Button-1>", callback_seleccionar_semilla)

        # Iniciar el bucle de eventos de la ventana de diálogo hasta que se cierre
        self.root.wait_window(self.ventana_semilla_dialogo)


# Bloque principal para iniciar la aplicación Tkinter
if __name__ == "__main__":
    root_principal = Tk() # Crear la ventana principal de Tkinter
    app_gui = FiltroBellezaGUI(root_principal) # Instanciar la clase de la GUI
    root_principal.mainloop() # Iniciar el bucle principal de eventos de Tkinter