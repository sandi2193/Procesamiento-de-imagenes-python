import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2

# --- Utility Functions ---
def mostrar_imagen(imagen_np, axes_img, max_ancho=400, max_alto=300):
    """Muestra una imagen NumPy array en un widget Label, manteniendo la proporción."""
    img_pil = Image.fromarray(imagen_np)
    ancho, alto = img_pil.size  # Obtener ancho y alto de la imagen PIL
    proporcion = ancho / alto

    if ancho > max_ancho or alto > max_alto:
        if ancho / max_ancho > alto / max_alto:
            nuevo_ancho = max_ancho
            nuevo_alto = int(max_ancho / proporcion)
        else:
            nuevo_alto = max_alto
            nuevo_ancho = int(max_alto * proporcion)
        img_pil = img_pil.resize((nuevo_ancho, nuevo_alto), Image.LANCZOS)  # Usar LANCZOS para mejor calidad

    img_tk = ImageTk.PhotoImage(image=img_pil)
    axes_img.img = img_tk
    axes_img.config(image=img_tk)

def actualizar_histograma(histograma, axes_hist, title):
    """Actualiza el histograma en los ejes proporcionados."""
    axes_hist.clear()
    axes_hist.bar(range(len(histograma)), histograma)
    axes_hist.set_title(title)
    axes_hist.set_xlabel('Nivel de Gris')
    axes_hist.set_ylabel('Frecuencia')
    axes_hist.figure.canvas.draw()

def cargar_imagen(parent):
    """Carga una imagen y la retorna como array NumPy."""
    file_path = filedialog.askopenfilename(
        title="Seleccionar Imagen",
        filetypes=(("Archivos de imagen", "*.jpg;*.jpeg;*.png;*.bmp;*.gif; *.jfif"), ("Todos los archivos", "*.*"))
    )
    if file_path:
        try:
            imagen = cv2.imread(file_path)
            return imagen
        except Exception as e:
            messagebox.showerror("Error al cargar la imagen", str(e))
            return None
    else:
        return None
# --- 1. Umbralización ---
def umbralizacion_app():
    ventana_umbralizacion = tk.Toplevel()
    ventana_umbralizacion.title("Umbralización")
    ventana_umbralizacion.geometry("1200x600+100+100")  # Adjusted size and position

    frame_imagenes = tk.Frame(ventana_umbralizacion)
    frame_imagenes.place(x=10, y=10, width=800, height=300)  # Positioned at the top

    ax_original_image_tk = tk.Label(frame_imagenes)
    ax_original_image_tk.place(x=0, y=0, width=400, height=300)
    ax_resultado_image_tk = tk.Label(frame_imagenes)
    ax_resultado_image_tk.place(x=400, y=0, width=400, height=300)

    frame_histogramas = tk.Frame(ventana_umbralizacion)
    frame_histogramas.place(x=10, y=320, width=800, height=200)  # Below images

    fig_hist_original = plt.Figure(figsize=(4, 2), dpi=100)
    ax_hist_original = fig_hist_original.add_subplot(111)
    canvas_hist_original = FigureCanvasTkAgg(fig_hist_original, master=frame_histogramas)
    canvas_hist_original_widget = canvas_hist_original.get_tk_widget()
    canvas_hist_original_widget.place(x=0, y=0, width=400, height=200)

    fig_hist_resultado = plt.Figure(figsize=(4, 2), dpi=100)
    ax_hist_resultado = fig_hist_resultado.add_subplot(111)
    canvas_hist_resultado = FigureCanvasTkAgg(fig_hist_resultado, master=frame_histogramas)
    canvas_hist_resultado_widget = canvas_hist_resultado.get_tk_widget()
    canvas_hist_resultado_widget.place(x=400, y=0, width=400, height=200)

    imagen_original = None
    imagen_gris = None
    imagen_umbralizada = None

    def actualizar_umbral(val):
        """Actualiza la umbralización y muestra los resultados."""
        global imagen_gris, imagen_umbralizada  # Make sure imagen_gris is accessible
        if imagen_gris is not None:  # Check if imagen_gris has been loaded
            umbral = int(val)
            _, imagen_umbralizada = cv2.threshold(imagen_gris, umbral, 255, cv2.THRESH_BINARY)
            mostrar_imagen(imagen_umbralizada, ax_resultado_image_tk)
            actualizar_histograma(cv2.calcHist([imagen_umbralizada], [0], None, [256], [0, 256]).flatten(), ax_hist_resultado,
                                  f'Histograma Umbral ({umbral})')

    def cargar_y_procesar():
        """Carga la imagen, calcula histogramas y aplica umbralización."""
        global imagen_original, imagen_gris, imagen_umbralizada, imagen_umbral_mediana, imagen_umbral_derivada, imagen_ecualizada  # Make variables global

        imagen_original = cargar_imagen(ventana_umbralizacion)
        if imagen_original is None:
            return

        imagen_gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)

        mostrar_imagen(imagen_gris, ax_original_image_tk)
        actualizar_histograma(cv2.calcHist([imagen_gris], [0], None, [256], [0, 256]).flatten(), ax_hist_original,
                              'Histograma Original')

        # Umbralización inicial (ejemplo: umbral 128)
        _, imagen_umbralizada = cv2.threshold(imagen_gris, 128, 255, cv2.THRESH_BINARY)
        mostrar_imagen(imagen_umbralizada, ax_resultado_image_tk)
        actualizar_histograma(cv2.calcHist([imagen_umbralizada], [0], None, [256], [0, 256]).flatten(), ax_hist_resultado,
                              'Histograma Umbral (128)')

        # --- Umbral Automático (Mediana) ---
        umbral_mediana = np.median(imagen_gris)
        _, imagen_umbral_mediana = cv2.threshold(imagen_gris, int(umbral_mediana), 255, cv2.THRESH_BINARY)

        # --- Umbral Automático (Derivada) ---
        hist = cv2.calcHist([imagen_gris], [0], None, [256], [0, 256]).flatten()
        derivada = np.diff(hist)
        umbral_derivada = np.argmax(derivada)  # Simple aproximación
        _, imagen_umbral_derivada = cv2.threshold(imagen_gris, int(umbral_derivada), 255, cv2.THRESH_BINARY)

        # --- Ecualización de Histograma ---
        imagen_ecualizada = cv2.equalizeHist(imagen_gris)

        # --- Interfaz de Umbral Manual ---
        frame_umbral = tk.Frame(ventana_umbralizacion)
        frame_umbral.place(x=10, y=530, width=400, height=30)

        tk.Label(frame_umbral, text="Umbral Manual:").pack(side=tk.LEFT)
        sld_umbral = tk.Scale(frame_umbral, from_=0, to=255, orient=tk.HORIZONTAL, command=actualizar_umbral)
        sld_umbral.set(128)
        sld_umbral.pack(side=tk.LEFT)

        # Llamar a actualizar_umbral con el valor inicial DESPUÉS de cargar la imagen
        actualizar_umbral(128)

        # --- Mostrar Umbrales Automáticos ---
        mensaje_umbrales = f"Mediana: {umbral_mediana}, Derivada: {umbral_derivada}"
        tk.Label(ventana_umbralizacion, text=mensaje_umbrales).place(x=420, y=530)

    # --- Botones ---
    frame_botones = tk.Frame(ventana_umbralizacion)  # Frame para los botones
    frame_botones.place(x=10, y=560, width=800, height=30)

    btn_cargar = tk.Button(frame_botones, text="Cargar Imagen", command=cargar_y_procesar)
    btn_cargar.pack(side=tk.LEFT, padx=5)
    btn_mostrar_mediana = tk.Button(frame_botones, text="Mostrar Umbral Mediana",
                                 command=lambda: (mostrar_imagen(imagen_umbral_mediana, ax_resultado_image_tk),
                                                   actualizar_histograma(
                                                       cv2.calcHist([imagen_umbral_mediana], [0], None, [256], [0, 256]).flatten(),
                                                       ax_hist_resultado, "Umbral Mediana")))
    btn_mostrar_mediana.pack(side=tk.LEFT, padx=5)
    btn_mostrar_derivada = tk.Button(frame_botones, text="Mostrar Umbral Derivada",
                                   command=lambda: (mostrar_imagen(imagen_umbral_derivada, ax_resultado_image_tk),
                                                     actualizar_histograma(
                                                         cv2.calcHist([imagen_umbral_derivada], [0], None, [256],
                                                         [0, 256]).flatten(), ax_hist_resultado, "Umbral Derivada")))

    btn_mostrar_derivada.pack(side=tk.LEFT, padx=5)
    btn_mostrar_ecualizacion = tk.Button(frame_botones, text="Mostrar Ecualización",
                                     command=lambda: (
                                         mostrar_imagen(imagen_ecualizada.copy(), ax_resultado_image_tk),
                                         actualizar_histograma(
                                             cv2.calcHist([imagen_ecualizada], [0], None, [256], [0, 256]).flatten(),
                                             ax_hist_resultado, "Ecualización")))
    btn_mostrar_ecualizacion.pack(side=tk.LEFT, padx=5)

    ventana_umbralizacion.mainloop()
# --- 2. Transformaciones de Intensidad ---

def transformaciones_intensidad_app():
    ventana_transformaciones = tk.Toplevel()
    ventana_transformaciones.title("Transformaciones de Intensidad")
    ventana_transformaciones.geometry("1200x700+100+100")

    # --- Frames para organizar la interfaz ---
    frame_imagenes = tk.Frame(ventana_transformaciones)
    frame_imagenes.pack(pady=10)

    frame_histogramas = tk.Frame(ventana_transformaciones)
    frame_histogramas.pack(pady=10)

    frame_controles = tk.Frame(ventana_transformaciones)
    frame_controles.pack(pady=10)

    # --- Widgets para mostrar imágenes e histogramas ---
    lbl_original_image = tk.Label(frame_imagenes)
    lbl_original_image.pack(side=tk.LEFT, padx=10)

    lbl_resultado_image = tk.Label(frame_imagenes)
    lbl_resultado_image.pack(side=tk.LEFT, padx=10)

    fig_hist_original = plt.Figure(figsize=(4, 2), dpi=100)
    ax_hist_original = fig_hist_original.add_subplot(111)
    canvas_hist_original = FigureCanvasTkAgg(fig_hist_original, master=frame_histogramas)
    canvas_hist_original_widget = canvas_hist_original.get_tk_widget()
    canvas_hist_original_widget.pack(side=tk.LEFT, padx=10)

    fig_hist_resultado = plt.Figure(figsize=(4, 2), dpi=100)
    ax_hist_resultado = fig_hist_resultado.add_subplot(111)
    canvas_hist_resultado = FigureCanvasTkAgg(fig_hist_resultado, master=frame_histogramas)
    canvas_hist_resultado_widget = canvas_hist_resultado.get_tk_widget()
    canvas_hist_resultado_widget.pack(side=tk.LEFT, padx=10)

    # --- Variables para almacenar la imagen ---
    imagen_original = None

    # --- Funciones para cargar y mostrar la imagen (MOVIDA ARRIBA) ---
    def cargar_y_mostrar():
        nonlocal imagen_original
        imagen_original = cargar_imagen(ventana_transformaciones)
        if imagen_original is not None:
            mostrar_imagen(imagen_original, lbl_original_image)
            actualizar_histograma(cv2.calcHist([imagen_original], [0], None, [256], [0, 256]).flatten(), ax_hist_original, "Histograma Original")

    # --- Funciones de Transformación ---
    def aplicar_negativa(imagen):
        """Aplica la transformación negativa a la imagen."""
        return 255 - imagen

    def aplicar_logaritmica(imagen, c):
        """Aplica la transformación logarítmica a la imagen."""
        return (c * np.log1p(imagen)).astype(np.uint8)

    def aplicar_exponencial(imagen, gamma):
        """Aplica la transformación exponencial (gamma) a la imagen."""
        return np.clip(255 * (imagen / 255) ** gamma, 0, 255).astype(np.uint8)

    # --- Función para aplicar la transformación seleccionada ---
    def aplicar_transformacion():
        nonlocal imagen_original
        if imagen_original is None:
            messagebox.showerror("Error", "Primero carga una imagen.")
            return

        transformacion = combo_transformacion.get()
        try:
            if transformacion == "Negativa":
                imagen_transformada = aplicar_negativa(imagen_original)
            elif transformacion == "Logarítmica":
                c = float(entry_valor.get())
                imagen_transformada = aplicar_logaritmica(imagen_original, c)
            elif transformacion == "Exponencial (Gamma)":
                gamma = float(entry_valor.get())
                imagen_transformada = aplicar_exponencial(imagen_original, gamma)
            else:
                messagebox.showerror("Error", "Transformación no válida.")
                return

            mostrar_imagen(imagen_transformada, lbl_resultado_image)
            actualizar_histograma(cv2.calcHist([imagen_transformada], [0], None, [256], [0, 256]).flatten(), ax_hist_resultado, f"Histograma {transformacion}")

        except ValueError:
            messagebox.showerror("Error", "Valor inválido. Ingrese un número.")

    # --- Interfaz de Control ---
    frame_controles_carga = tk.Frame(frame_controles)
    frame_controles_carga.pack(pady=5)

    btn_cargar = tk.Button(frame_controles_carga, text="Cargar Imagen", command=lambda: cargar_y_mostrar())
    btn_cargar.pack(side=tk.LEFT, padx=5)

    frame_seleccion = tk.Frame(frame_controles)
    frame_seleccion.pack(pady=5)

    tk.Label(frame_seleccion, text="Transformación:").pack(side=tk.LEFT, padx=5)
    combo_transformacion = ttk.Combobox(frame_seleccion, values=["Negativa", "Logarítmica", "Exponencial (Gamma)"])
    combo_transformacion.set("Negativa")  # Valor por defecto
    combo_transformacion.pack(side=tk.LEFT, padx=5)

    tk.Label(frame_seleccion, text="Valor:").pack(side=tk.LEFT, padx=5)
    entry_valor = tk.Entry(frame_seleccion)
    entry_valor.insert(0, "1")  # Valor por defecto
    entry_valor.pack(side=tk.LEFT, padx=5)

    btn_aplicar = tk.Button(frame_controles, text="Aplicar Transformación", command=aplicar_transformacion)
    btn_aplicar.pack(pady=10)

    ventana_transformaciones.mainloop()
    
# --- 3. Transformaciones de Intensidad por Tramos ---
def transformaciones_tramos_app():
    ventana_tramos = tk.Toplevel()
    ventana_tramos.title("Transformaciones de Intensidad por Tramos")
    ventana_tramos.geometry("1200x700+100+100")

    # --- Frames para organizar la interfaz ---
    frame_imagenes = tk.Frame(ventana_tramos)
    frame_imagenes.pack(pady=10)

    frame_histogramas = tk.Frame(ventana_tramos)
    frame_histogramas.pack(pady=10)

    frame_controles = tk.Frame(ventana_tramos)
    frame_controles.pack(pady=10)

    # --- Widgets para mostrar imágenes e histogramas ---
    lbl_original_image = tk.Label(frame_imagenes)
    lbl_original_image.pack(side=tk.LEFT, padx=10)

    lbl_resultado_image = tk.Label(frame_imagenes)
    lbl_resultado_image.pack(side=tk.LEFT, padx=10)

    fig_hist_original = plt.Figure(figsize=(4, 2), dpi=100)
    ax_hist_original = fig_hist_original.add_subplot(111)
    canvas_hist_original = FigureCanvasTkAgg(fig_hist_original, master=frame_histogramas)
    canvas_hist_original_widget = canvas_hist_original.get_tk_widget()
    canvas_hist_original_widget.pack(side=tk.LEFT, padx=10)

    fig_hist_resultado = plt.Figure(figsize=(4, 2), dpi=100)
    ax_hist_resultado = fig_hist_resultado.add_subplot(111)
    canvas_hist_resultado = FigureCanvasTkAgg(fig_hist_resultado, master=frame_histogramas)
    canvas_hist_resultado_widget = canvas_hist_resultado.get_tk_widget()
    canvas_hist_resultado_widget.pack(side=tk.LEFT, padx=10)

    # --- Variables para almacenar la imagen y los puntos de control ---
    imagen_original = None
    puntos_control = []  # Lista de tuplas (x, y)

    # --- Funciones ---
    def cargar_y_mostrar():
        nonlocal imagen_original
    imagen = cargar_imagen(ventana_tramos)
    if imagen is not None:
        if len(imagen.shape) == 3:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
        imagen_original = imagen
        mostrar_imagen(imagen_original, lbl_original_image)
        actualizar_histograma(
            cv2.calcHist([imagen_original], [0], None, [256], [0, 256]).flatten(),
            ax_hist_original,
            "Histograma Original"
        )

    def agregar_punto():
        x = simpledialog.askinteger("Ingresar Punto", "Valor X (0-255):")
        y = simpledialog.askinteger("Ingresar Punto", "Valor Y (0-255):")
        if x is not None and y is not None:
            puntos_control.append((x, y))
            puntos_control.sort()  # Ordenar los puntos por X
            actualizar_lista_puntos()

    def eliminar_punto():
        seleccion = lista_puntos.curselection()
        if seleccion:
            indice = seleccion[0]
            del puntos_control[indice]
            actualizar_lista_puntos()

    def actualizar_lista_puntos():
        lista_puntos.delete(0, tk.END)
        for punto in puntos_control:
            lista_puntos.insert(tk.END, f"({punto[0]}, {punto[1]})")

    def aplicar_transformacion_tramos():
        nonlocal imagen_original
        if imagen_original is None:
            messagebox.showerror("Error", "Primero carga una imagen.")
            return

        if len(puntos_control) < 2:
            messagebox.showerror("Error", "Se necesitan al menos 2 puntos de control.")
            return

        imagen_transformada = np.zeros_like(imagen_original)
        for i in range(imagen_original.shape[0]):
            for j in range(imagen_original.shape[1]):
                valor_original = imagen_original[i, j]
                valor_transformado = calcular_transformacion(valor_original)
                imagen_transformada[i, j] = np.clip(valor_transformado, 0, 255)

        mostrar_imagen(imagen_transformada, lbl_resultado_image)
        actualizar_histograma(cv2.calcHist([imagen_transformada], [0], None, [256], [0, 256]).flatten(), ax_hist_resultado,
                              "Histograma Transformado")

    def calcular_transformacion(valor):
        """Calcula el nuevo valor de píxel según los puntos de control."""
        if valor < puntos_control[0][0]:
            return puntos_control[0][1]
        if valor > puntos_control[-1][0]:
            return puntos_control[-1][1]

        for i in range(len(puntos_control) - 1):
            x1, y1 = puntos_control[i]
            x2, y2 = puntos_control[i + 1]
            if x1 <= valor <= x2:
                m = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
                b = y1 - m * x1
                return m * valor + b
        return valor  # Si no se encuentra el intervalo, devuelve el valor original

    # --- Interfaz de Control ---
    frame_controles_carga = tk.Frame(frame_controles)
    frame_controles_carga.pack(pady=5)

    btn_cargar = tk.Button(frame_controles_carga, text="Cargar Imagen", command=cargar_y_mostrar)
    btn_cargar.pack(side=tk.LEFT, padx=5)

    frame_puntos = tk.Frame(frame_controles)
    frame_puntos.pack(pady=5)

    btn_agregar_punto = tk.Button(frame_puntos, text="Agregar Punto", command=agregar_punto)
    btn_agregar_punto.pack(side=tk.LEFT, padx=5)

    btn_eliminar_punto = tk.Button(frame_puntos, text="Eliminar Punto", command=eliminar_punto)
    btn_eliminar_punto.pack(side=tk.LEFT, padx=5)

    lista_puntos = tk.Listbox(frame_puntos, height=5)
    lista_puntos.pack(side=tk.LEFT, padx=5)

    btn_aplicar = tk.Button(frame_controles, text="Aplicar Transformación", command=aplicar_transformacion_tramos)
    btn_aplicar.pack(pady=10)

    # Inicializar la lista de puntos
    actualizar_lista_puntos()

    ventana_tramos.mainloop()
    
# --- 4. Procesamiento Local del Histograma ---
def procesamiento_local_histograma_app():
    ventana_local_hist = tk.Toplevel()
    ventana_local_hist.title("Procesamiento Local del Histograma")
    ventana_local_hist.geometry("1200x700+100+100")

    # --- Frames para organizar la interfaz ---
    frame_imagenes = tk.Frame(ventana_local_hist)
    frame_imagenes.pack(pady=10)

    frame_histogramas = tk.Frame(ventana_local_hist)
    frame_histogramas.pack(pady=10)

    frame_controles = tk.Frame(ventana_local_hist)
    frame_controles.pack(pady=10)

    # --- Widgets para mostrar imágenes e histogramas ---
    lbl_original_image = tk.Label(frame_imagenes)
    lbl_original_image.pack(side=tk.LEFT, padx=10)

    lbl_resultado_image = tk.Label(frame_imagenes)
    lbl_resultado_image.pack(side=tk.LEFT, padx=10)

    fig_hist_original = plt.Figure(figsize=(4, 2), dpi=100)
    ax_hist_original = fig_hist_original.add_subplot(111)
    canvas_hist_original = FigureCanvasTkAgg(fig_hist_original, master=frame_histogramas)
    canvas_hist_original_widget = canvas_hist_original.get_tk_widget()
    canvas_hist_original_widget.pack(side=tk.LEFT, padx=10)

    fig_hist_resultado = plt.Figure(figsize=(4, 2), dpi=100)
    ax_hist_resultado = fig_hist_resultado.add_subplot(111)
    canvas_hist_resultado = FigureCanvasTkAgg(fig_hist_resultado, master=frame_histogramas)
    canvas_hist_resultado_widget = canvas_hist_resultado.get_tk_widget()
    canvas_hist_resultado_widget.pack(side=tk.LEFT, padx=10)

    # --- Variables para almacenar la imagen y el tamaño de la ventana ---
    imagen_original = None
    tamano_ventana = tk.IntVar(value=3)  # Valor por defecto: 3x3

    # --- Funciones ---
    def cargar_y_mostrar():
        nonlocal imagen_original
        imagen_original = cargar_imagen(ventana_local_hist)
        if imagen_original is not None:
            imagen_gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)
            mostrar_imagen(imagen_gris, lbl_original_image)
            actualizar_histograma(cv2.calcHist([imagen_gris], [0], None, [256], [0, 256]).flatten(), ax_hist_original,
                                  "Histograma Original")

    def aplicar_ecualizacion_local():
        nonlocal imagen_original
        if imagen_original is None:
            messagebox.showerror("Error", "Primero carga una imagen.")
            return

        try:
            ventana = tamano_ventana.get()
            if ventana % 2 == 0 or ventana < 3:
                messagebox.showerror("Error", "El tamaño de la ventana debe ser impar y mayor o igual que 3.")
                return

            imagen_gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)
            imagen_ecualizada_local = ecualizar_histograma_local(imagen_gris, ventana)
            mostrar_imagen(imagen_ecualizada_local, lbl_resultado_image)
            actualizar_histograma(cv2.calcHist([imagen_ecualizada_local], [0], None, [256], [0, 256]).flatten(),
                                  ax_hist_resultado, "Histograma Ecualizado Local")

        except tk.TclError:
            messagebox.showerror("Error", "Ingrese un tamaño de ventana válido.")

    def ecualizar_histograma_local(imagen, tamano_ventana):
        """Aplica la ecualización local del histograma."""

        filas, columnas = imagen.shape
        imagen_ecualizada = np.zeros_like(imagen)
        offset = tamano_ventana // 2

        for i in range(offset, filas - offset):
            for j in range(offset, columnas - offset):
                ventana_local = imagen[i - offset:i + offset + 1, j - offset:j + offset + 1]
                imagen_ecualizada[i, j] = cv2.equalizeHist(ventana_local).flat[offset * tamano_ventana + offset]

        return imagen_ecualizada

    # --- Interfaz de Control ---
    frame_controles_carga = tk.Frame(frame_controles)
    frame_controles_carga.pack(pady=5)

    btn_cargar = tk.Button(frame_controles_carga, text="Cargar Imagen", command=cargar_y_mostrar)
    btn_cargar.pack(side=tk.LEFT, padx=5)

    frame_ventana = tk.Frame(frame_controles)
    frame_ventana.pack(pady=5)

    tk.Label(frame_ventana, text="Tamaño Ventana:").pack(side=tk.LEFT, padx=5)
    entry_tamano_ventana = tk.Entry(frame_ventana, textvariable=tamano_ventana, width=5)
    entry_tamano_ventana.pack(side=tk.LEFT, padx=5)

    btn_aplicar = tk.Button(frame_controles, text="Aplicar Ecualización Local", command=aplicar_ecualizacion_local)
    btn_aplicar.pack(pady=10)

    ventana_local_hist.mainloop()
    
# --- 5. Realce Local Media y Varianza ---
def realce_local_media_varianza_app():
    ventana_realce = tk.Toplevel()
    ventana_realce.title("Realce Local Media y Varianza")
    ventana_realce.geometry("1200x800+100+100")

    # --- Frames para organizar la interfaz ---
    frame_imagenes = tk.Frame(ventana_realce)
    frame_imagenes.pack(pady=10)

    frame_histogramas = tk.Frame(ventana_realce)
    frame_histogramas.pack(pady=10)

    frame_controles = tk.Frame(ventana_realce)
    frame_controles.pack(pady=10)

    # --- Widgets para mostrar imágenes e histogramas ---
    lbl_original_image = tk.Label(frame_imagenes)
    lbl_original_image.pack(side=tk.LEFT, padx=10)

    lbl_resultado_image = tk.Label(frame_imagenes)
    lbl_resultado_image.pack(side=tk.LEFT, padx=10)

    fig_hist_original = plt.Figure(figsize=(4, 2), dpi=100)
    ax_hist_original = fig_hist_original.add_subplot(111)
    canvas_hist_original = FigureCanvasTkAgg(fig_hist_original, master=frame_histogramas)
    canvas_hist_original_widget = canvas_hist_original.get_tk_widget()
    canvas_hist_original_widget.pack(side=tk.LEFT, padx=10)

    fig_hist_resultado = plt.Figure(figsize=(4, 2), dpi=100)
    ax_hist_resultado = fig_hist_resultado.add_subplot(111)
    canvas_hist_resultado = FigureCanvasTkAgg(fig_hist_resultado, master=frame_histogramas)
    canvas_hist_resultado_widget = canvas_hist_resultado.get_tk_widget()
    canvas_hist_resultado_widget.pack(side=tk.LEFT, padx=10)

    # --- Variables para almacenar la imagen y los parámetros ---
    imagen_original = None
    k0 = tk.DoubleVar(value=0.2)
    k1 = tk.DoubleVar(value=0.02)
    k2 = tk.DoubleVar(value=0.4)
    e_global = tk.DoubleVar(value=4.0)

    # --- Funciones ---
    def cargar_y_mostrar():
        nonlocal imagen_original
        imagen_original = cargar_imagen(ventana_realce)
        if imagen_original is not None:
            mostrar_imagen(imagen_original, lbl_original_image)
            if len(imagen_original.shape) == 3:  # Imagen a color
                imagen_gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)
            else:
                imagen_gris = imagen_original.copy()  # Imagen ya en escala de grises
            actualizar_histograma(cv2.calcHist([imagen_gris], [0], None, [256], [0, 256]).flatten(), ax_hist_original,
                                  "Histograma Original")

    def aplicar_realce_local():
        nonlocal imagen_original
        if imagen_original is None:
            messagebox.showerror("Error", "Primero carga una imagen.")
            return

        try:
            nonlocal k0, k1, k2, e_global
            k0_val = k0.get()
            k1_val = k1.get()
            k2_val = k2.get()
            e_global_val = e_global.get()

            imagen_realzada = realce_local_media_varianza(imagen_original, k0_val, k1_val, k2_val, e_global_val)
            mostrar_imagen(imagen_realzada.astype(np.uint8), lbl_resultado_image)
            if len(imagen_realzada.shape) == 3:
                imagen_gris_realzada = cv2.cvtColor(imagen_realzada.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            else:
                imagen_gris_realzada = imagen_realzada.astype(np.uint8).copy()
            actualizar_histograma(cv2.calcHist([imagen_gris_realzada], [0], None, [256], [0, 256]).flatten(),
                                  ax_hist_resultado, "Histograma Realzado")

        except ValueError:
            messagebox.showerror("Error", "Ingrese valores numéricos válidos para los parámetros.")

    def realce_local_media_varianza(imagen, k0, k1, k2, e_global):
        """Aplica el realce local basado en media y varianza."""

        if len(imagen.shape) == 3:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen.copy()

        media_global = np.mean(imagen_gris)
        varianza_global = np.var(imagen_gris)
        imagen_realzada = imagen.copy().astype(np.float64)

        for c in range(imagen.shape[2] if len(imagen.shape) == 3 else 1):
            for i in range(imagen.shape[0]):
                for j in range(imagen.shape[1]):
                    if len(imagen.shape) == 3:
                        intensidad_local = imagen[i - 1:i + 2, j - 1:j + 2, c]
                    else:
                        intensidad_local = imagen[i - 1:i + 2, j - 1:j + 2]

                    if intensidad_local.size < 9:
                        continue  # Omitir bordes

                    media_local = np.mean(intensidad_local)
                    varianza_local = np.var(intensidad_local)

                    if (k0 * media_global <= media_local <= k1 * media_global) and (
                            k2 * varianza_global <= varianza_local):
                        if len(imagen.shape) == 3:
                            imagen_realzada[i, j, c] = np.clip(e_global * imagen[i, j, c], 0, 255)
                        else:
                            imagen_realzada[i, j] = np.clip(e_global * imagen[i, j], 0, 255)

        return imagen_realzada

    # --- Interfaz de Control ---
    frame_controles_carga = tk.Frame(frame_controles)
    frame_controles_carga.pack(pady=5)

    btn_cargar = tk.Button(frame_controles_carga, text="Cargar Imagen", command=cargar_y_mostrar)
    btn_cargar.pack(side=tk.LEFT, padx=5)

    frame_parametros = tk.Frame(frame_controles)
    frame_parametros.pack(pady=5)

    tk.Label(frame_parametros, text="k0:").pack(side=tk.LEFT, padx=2)
    entry_k0 = tk.Entry(frame_parametros, textvariable=k0, width=5)
    entry_k0.pack(side=tk.LEFT, padx=2)

    tk.Label(frame_parametros, text="k1:").pack(side=tk.LEFT, padx=2)
    entry_k1 = tk.Entry(frame_parametros, textvariable=k1, width=5)
    entry_k1.pack(side=tk.LEFT, padx=2)

    tk.Label(frame_parametros, text="k2:").pack(side=tk.LEFT, padx=2)
    entry_k2 = tk.Entry(frame_parametros, textvariable=k2, width=5)
    entry_k2.pack(side=tk.LEFT, padx=2)

    tk.Label(frame_parametros, text="E (Global):").pack(side=tk.LEFT, padx=2)
    entry_e = tk.Entry(frame_parametros, textvariable=e_global, width=5)
    entry_e.pack(side=tk.LEFT, padx=2)

    btn_aplicar = tk.Button(frame_controles, text="Aplicar Realce Local", command=aplicar_realce_local)
    btn_aplicar.pack(pady=10)

    ventana_realce.mainloop()

def filtros_espaciales_app():
    ventana_filtros = tk.Toplevel()
    ventana_filtros.title("Filtros Espaciales")
    ventana_filtros.geometry("1200x800+100+100")

    # --- Frames para organizar la interfaz ---
    frame_imagenes = tk.Frame(ventana_filtros)
    frame_imagenes.pack(pady=10)

    frame_histogramas = tk.Frame(ventana_filtros)
    frame_histogramas.pack(pady=10)

    frame_controles = tk.Frame(ventana_filtros)
    frame_controles.pack(pady=10)

    # --- Widgets para mostrar imágenes e histogramas ---
    lbl_original_image = tk.Label(frame_imagenes)
    lbl_original_image.pack(side=tk.LEFT, padx=10)

    lbl_resultado_image = tk.Label(frame_imagenes)
    lbl_resultado_image.pack(side=tk.LEFT, padx=10)

    fig_hist_original = plt.Figure(figsize=(4, 2), dpi=100)
    ax_hist_original = fig_hist_original.add_subplot(111)
    canvas_hist_original = FigureCanvasTkAgg(fig_hist_original, master=frame_histogramas)
    canvas_hist_original_widget = canvas_hist_original.get_tk_widget()
    canvas_hist_original_widget.pack(side=tk.LEFT, padx=10)

    fig_hist_resultado = plt.Figure(figsize=(4, 2), dpi=100)
    ax_hist_resultado = fig_hist_resultado.add_subplot(111)
    canvas_hist_resultado = FigureCanvasTkAgg(fig_hist_resultado, master=frame_histogramas)
    canvas_hist_resultado_widget = canvas_hist_resultado.get_tk_widget()
    canvas_hist_resultado_widget.pack(side=tk.LEFT, padx=10)

    # --- Variables para almacenar la imagen y el kernel ---
    imagen_original = None
    kernel_size = tk.IntVar(value=3)  # Tamaño del kernel (por defecto 3x3)
    kernel_values = {}  # Diccionario para almacenar los valores del kernel

    # --- Funciones ---
    def cargar_y_mostrar():
        nonlocal imagen_original
        imagen_original = cargar_imagen(ventana_filtros)
        if imagen_original is not None:
            imagen_gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)
            mostrar_imagen(imagen_gris, lbl_original_image)
            actualizar_histograma(cv2.calcHist([imagen_gris], [0], None, [256], [0, 256]).flatten(), ax_hist_original,
                                  "Histograma Original")
            crear_entradas_kernel()  # Crear entradas para el kernel

    def crear_entradas_kernel():
        """Crea las entradas para el kernel en la interfaz."""
        nonlocal kernel_size, frame_kernel
        size = kernel_size.get()
        # Limpiar el frame antes de crear nuevas entradas
        for widget in frame_kernel.winfo_children():
            widget.destroy()

        for i in range(size):
            for j in range(size):
                entry = tk.Entry(frame_kernel, width=3)
                entry.grid(row=i, column=j, padx=2, pady=2)
                kernel_values[(i, j)] = entry
                entry.insert(0, "1" if size == 3 else "0")  # Valor por defecto

    def aplicar_filtro():
        nonlocal imagen_original
        if imagen_original is None:
            messagebox.showerror("Error", "Primero carga una imagen.")
            return

        try:
            size = kernel_size.get()
            if size % 2 == 0 or size < 3:
                messagebox.showerror("Error", "El tamaño del kernel debe ser impar y mayor o igual que 3.")
                return

            # Obtener valores del kernel desde las entradas
            kernel = np.zeros((size, size), dtype=np.float32)
            for i in range(size):
                for j in range(size):
                    kernel[i, j] = float(kernel_values[(i, j)].get())

            # Normalizar el kernel (opcional, pero común para suavizado)
            kernel /= np.sum(kernel)

            imagen_gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY) if len(imagen_original.shape) == 3 else imagen_original.copy()
            imagen_filtrada = cv2.filter2D(imagen_gris, -1, kernel)  # Usar cv2.filter2D

            mostrar_imagen(imagen_filtrada, lbl_resultado_image)
            actualizar_histograma(cv2.calcHist([imagen_filtrada], [0], None, [256], [0, 256]).flatten(),
                                  ax_hist_resultado, "Histograma Filtrado")

        except ValueError:
            messagebox.showerror("Error", "Ingrese valores numéricos válidos para el kernel.")
        except cv2.error as e:
            messagebox.showerror("Error", f"Error de OpenCV: {e}")

    def aplicar_filtro_opencv(filtro_type):
        nonlocal imagen_original
        if imagen_original is None:
            messagebox.showerror("Error", "Primero carga una imagen.")
            return

        imagen_gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY) if len(imagen_original.shape) == 3 else imagen_original.copy()
        if filtro_type == "Promedio":
            imagen_filtrada = cv2.blur(imagen_gris, (5, 5))  # Filtro promedio
        elif filtro_type == "Gaussiano":
            imagen_filtrada = cv2.GaussianBlur(imagen_gris, (5, 5), 0)  # Filtro Gaussiano
        elif filtro_type == "Mediana":
            imagen_filtrada = cv2.medianBlur(imagen_gris, 5)  # Filtro Mediana
        elif filtro_type == "Maximo":
            imagen_filtrada = cv2.dilate(imagen_gris, np.ones((5, 5), np.uint8))
        elif filtro_type == "Minimo":
            imagen_filtrada = cv2.erode(imagen_gris, np.ones((5, 5), np.uint8))
        else:
            messagebox.showerror("Error", "Tipo de filtro no válido.")
            return

        mostrar_imagen(imagen_filtrada, lbl_resultado_image)
        actualizar_histograma(cv2.calcHist([imagen_filtrada], [0], None, [256], [0, 256]).flatten(),
                              ax_hist_resultado, f"Histograma {filtro_type}")

    # --- Interfaz de Control ---
    frame_controles_carga = tk.Frame(frame_controles)
    frame_controles_carga.pack(pady=5)

    btn_cargar = tk.Button(frame_controles_carga, text="Cargar Imagen", command=cargar_y_mostrar)
    btn_cargar.pack(side=tk.LEFT, padx=5)

    frame_tamano_kernel = tk.Frame(frame_controles)
    frame_tamano_kernel.pack(pady=5)

    tk.Label(frame_tamano_kernel, text="Tamaño Kernel:").pack(side=tk.LEFT, padx=5)
    entry_tamano_kernel = tk.Entry(frame_tamano_kernel, textvariable=kernel_size, width=5)
    entry_tamano_kernel.pack(side=tk.LEFT, padx=5)

    btn_actualizar_kernel = tk.Button(frame_tamano_kernel, text="Actualizar Kernel", command=crear_entradas_kernel)
    btn_actualizar_kernel.pack(side=tk.LEFT, padx=5)

    frame_kernel = tk.Frame(frame_controles)
    frame_kernel.pack(pady=5)

    frame_aplicar = tk.Frame(frame_controles)
    frame_aplicar.pack(pady=5)

    btn_aplicar = tk.Button(frame_aplicar, text="Aplicar Filtro Personalizado", command=aplicar_filtro)
    btn_aplicar.pack(side=tk.LEFT, padx=5)

    frame_opencv_filtros = tk.Frame(frame_controles)
    frame_opencv_filtros.pack(pady=5)

    ttk.Label(frame_opencv_filtros, text="Filtros OpenCV:").pack(side=tk.LEFT, padx=5)
    combo_opencv_filtros = ttk.Combobox(frame_opencv_filtros, values=["Promedio", "Gaussiano", "Mediana", "Maximo", "Minimo"])
    combo_opencv_filtros.pack(side=tk.LEFT, padx=5)
    combo_opencv_filtros.set("Promedio")  # Valor por defecto

    btn_aplicar_opencv = tk.Button(frame_opencv_filtros, text="Aplicar Filtro OpenCV", command=lambda: aplicar_filtro_opencv(combo_opencv_filtros.get()))
    btn_aplicar_opencv.pack(side=tk.LEFT, padx=5)

    ventana_filtros.mainloop()

# --- 7. Detección de Bordes Avanzada ---
def deteccion_bordes_app():
    ventana_bordes = tk.Toplevel()
    ventana_bordes.title("Detección de Bordes Avanzada")
    ventana_bordes.geometry("1400x900+100+100")

    # --- Frames para organizar la interfaz ---
    frame_imagenes = tk.Frame(ventana_bordes)
    frame_imagenes.pack(pady=10)

    frame_histogramas = tk.Frame(ventana_bordes)
    frame_histogramas.pack(pady=10)

    frame_controles = tk.Frame(ventana_bordes)
    frame_controles.pack(pady=10)

    # --- Widgets para mostrar imágenes e histogramas ---
    lbl_original_image = tk.Label(frame_imagenes)
    lbl_original_image.pack(side=tk.LEFT, padx=10)

    lbl_resultado_image = tk.Label(frame_imagenes)
    lbl_resultado_image.pack(side=tk.LEFT, padx=10)

    fig_hist_original = plt.Figure(figsize=(4, 2), dpi=100)
    ax_hist_original = fig_hist_original.add_subplot(111)
    canvas_hist_original = FigureCanvasTkAgg(fig_hist_original, master=frame_histogramas)
    canvas_hist_original_widget = canvas_hist_original.get_tk_widget()
    canvas_hist_original_widget.pack(side=tk.LEFT, padx=10)

    fig_hist_resultado = plt.Figure(figsize=(4, 2), dpi=100)
    ax_hist_resultado = fig_hist_resultado.add_subplot(111)
    canvas_hist_resultado = FigureCanvasTkAgg(fig_hist_resultado, master=frame_histogramas)
    canvas_hist_resultado_widget = canvas_hist_resultado.get_tk_widget()
    canvas_hist_resultado_widget.pack(side=tk.LEFT, padx=10)

    # --- Variables ---
    imagen_original = None
    kernel_type = tk.StringVar(value="Laplaciano")
    custom_kernel_size = tk.StringVar(value="3x3")
    custom_kernel_entries = []

    # --- Funciones Locales ---
    def cargar_y_mostrar():
        nonlocal imagen_original
        imagen_original = cargar_imagen(ventana_bordes)
        if imagen_original is not None:
            imagen_gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY)
            mostrar_imagen(imagen_gris, lbl_original_image)
            actualizar_histograma(cv2.calcHist([imagen_gris], [0], None, [256], [0, 256]).flatten(), ax_hist_original,
                                    "Histograma Original")
            mostrar_imagen(None, lbl_resultado_image)
            actualizar_histograma([], ax_hist_resultado, "Histograma Resultado")
            actualizar_interfaz_kernel_personalizado()

    def obtener_custom_kernel():
        size_str = custom_kernel_size.get()
        size = int(size_str[0])
        kernel = np.zeros((size, size), dtype=np.float32)
        if len(custom_kernel_entries) == size:
            for i in range(size):
                row_entries = custom_kernel_entries[i]
                if len(row_entries) == size:
                    for j, entry in enumerate(row_entries):
                        try:
                            kernel[i, j] = float(entry.get())
                        except ValueError:
                            messagebox.showerror("Error", "Por favor, ingrese valores numéricos en el kernel personalizado.")
                            return None
                else:
                    messagebox.showerror("Error", f"La fila {i+1} del kernel personalizado tiene un número incorrecto de valores.")
                    return None
        else:
            messagebox.showerror("Error", "El número de filas en el kernel personalizado no coincide con el tamaño seleccionado.")
            return None
        return kernel

    def aplicar_deteccion_bordes():
        nonlocal imagen_original
        if imagen_original is None:
            messagebox.showerror("Error", "Primero carga una imagen.")
            return

        imagen_gris = cv2.cvtColor(imagen_original, cv2.COLOR_BGR2GRAY) if len(imagen_original.shape) == 3 else imagen_original.copy()
        tipo_kernel = kernel_type.get()
        imagen_bordes = None

        if tipo_kernel == "Laplaciano Personalizado":
            kernel = obtener_custom_kernel()
            if kernel is not None:
                imagen_bordes = cv2.filter2D(imagen_gris, cv2.CV_64F, kernel)
                imagen_bordes = cv2.convertScaleAbs(imagen_bordes)
        elif tipo_kernel == "Laplaciano":
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
            imagen_bordes = cv2.filter2D(imagen_gris, cv2.CV_64F, kernel)
            imagen_bordes = cv2.convertScaleAbs(imagen_bordes)
        elif tipo_kernel == "Laplaciano Variacion 1":
            kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
            imagen_bordes = cv2.filter2D(imagen_gris, cv2.CV_64F, kernel)
            imagen_bordes = cv2.convertScaleAbs(imagen_bordes)
        elif tipo_kernel == "Laplaciano Variacion 2":
            kernel1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
            kernel2 = np.array([[1, 0, 1], [0, -4, 0], [1, 0, 1]], dtype=np.float32)
            imagen_lap1 = cv2.filter2D(imagen_gris, cv2.CV_64F, kernel1)
            imagen_lap2 = cv2.filter2D(imagen_gris, cv2.CV_64F, kernel2)
            imagen_bordes = cv2.bitwise_or(cv2.convertScaleAbs(imagen_lap1), cv2.convertScaleAbs(imagen_lap2))
        elif tipo_kernel == "Prewitt":
            kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
            kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
            imagen_prewitt_x = cv2.filter2D(imagen_gris, cv2.CV_64F, kernel_x)
            imagen_prewitt_y = cv2.filter2D(imagen_gris, cv2.CV_64F, kernel_y)
            imagen_bordes = cv2.addWeighted(cv2.convertScaleAbs(imagen_prewitt_x), 0.5, cv2.convertScaleAbs(imagen_prewitt_y), 0.5, 0)
        elif tipo_kernel == "Sobel":
            imagen_sobel_x = cv2.Sobel(imagen_gris, cv2.CV_64F, 1, 0, ksize=3)
            imagen_sobel_y = cv2.Sobel(imagen_gris, cv2.CV_64F, 0, 1, ksize=3)
            imagen_bordes = cv2.addWeighted(cv2.convertScaleAbs(imagen_sobel_x), 0.5, cv2.convertScaleAbs(imagen_sobel_y), 0.5, 0)
        elif tipo_kernel == "Roberts":
            kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
            kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
            imagen_roberts_x = cv2.filter2D(imagen_gris, cv2.CV_64F, kernel_x)
            imagen_roberts_y = cv2.filter2D(imagen_gris, cv2.CV_64F, kernel_y)
            imagen_bordes = cv2.addWeighted(cv2.convertScaleAbs(imagen_roberts_x), 0.5, cv2.convertScaleAbs(imagen_roberts_y), 0.5, 0)
        elif tipo_kernel == "Canny":
            imagen_bordes = cv2.Canny(imagen_gris, 100, 200)
        elif tipo_kernel == "Scharr":
            imagen_scharr_x = cv2.Scharr(imagen_gris, cv2.CV_64F, 1, 0)
            imagen_scharr_y = cv2.Scharr(imagen_gris, cv2.CV_64F, 0, 1)
            imagen_bordes = cv2.addWeighted(cv2.convertScaleAbs(imagen_scharr_x), 0.5, cv2.convertScaleAbs(imagen_scharr_y), 0.5, 0)
        else:
            messagebox.showerror("Error", "Tipo de kernel no válido.")
            return

        if imagen_bordes is not None:
            mostrar_imagen(imagen_bordes, lbl_resultado_image)
            actualizar_histograma(cv2.calcHist([imagen_bordes], [0], None, [256], [0, 256]).flatten(),
                                    ax_hist_resultado, f"Histograma Bordes ({tipo_kernel})")
        else:
            mostrar_imagen(None, lbl_resultado_image)
            actualizar_histograma([], ax_hist_resultado, "Histograma Resultado")

    def actualizar_interfaz_kernel_personalizado():
        nonlocal custom_kernel_entries
        # Destruir todos los widgets dentro del frame de entrada del kernel
        for widget in frame_custom_kernel_input.winfo_children():
            widget.destroy()
        frame_custom_kernel_input.pack_forget()
        custom_kernel_entries.clear()

        if kernel_type.get() == "Laplaciano Personalizado":
            frame_custom_kernel_input.pack(pady=5)
            size_str = custom_kernel_size.get()
            size = int(size_str[0])
            custom_kernel_entries = [[] for _ in range(size)] # Inicializar la lista de filas
            for i in range(size):
                row_frame = tk.Frame(frame_custom_kernel_input)
                row_frame.pack(side=tk.TOP)
                row_entries = []
                for j in range(size):
                    entry = tk.Entry(row_frame, width=5)
                    entry.pack(side=tk.LEFT, padx=2, pady=2)
                    row_entries.append(entry)
                if len(custom_kernel_entries) > i:
                    custom_kernel_entries[i] = row_entries

    def generar_kernel_laplaciano_default():
        custom_kernel_size.set("3x3")
        actualizar_interfaz_kernel_personalizado()
        # Llena los campos de entrada con el kernel Laplaciano estándar
        default_kernel = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        for i in range(3):
            for j in range(3):
                custom_kernel_entries[i][j].insert(0, str(default_kernel[i][j]))

    # --- Interfaz de Control ---
    frame_controles_carga = tk.Frame(frame_controles)
    frame_controles_carga.pack(pady=5)

    btn_cargar = tk.Button(frame_controles_carga, text="Cargar Imagen", command=cargar_y_mostrar)
    btn_cargar.pack(side=tk.LEFT, padx=5)

    frame_tipo_kernel = tk.Frame(frame_controles)
    frame_tipo_kernel.pack(pady=5)

    tk.Label(frame_tipo_kernel, text="Tipo de Kernel:").pack(side=tk.LEFT, padx=5)
    combo_tipo_kernel = ttk.Combobox(frame_tipo_kernel, textvariable=kernel_type,
                                        values=["Laplaciano", "Laplaciano Variacion 1", "Laplaciano Variacion 2",
                                                "Prewitt", "Sobel", "Roberts", "Canny", "Scharr",
                                                "Laplaciano Personalizado"])
    combo_tipo_kernel.pack(side=tk.LEFT, padx=5)
    combo_tipo_kernel.set("Laplaciano")
    combo_tipo_kernel.bind("<<ComboboxSelected>>", lambda event: actualizar_interfaz_kernel_personalizado())

    # --- Frame para Kernel Personalizado ---
    frame_custom_kernel_config = tk.Frame(frame_controles)
    frame_custom_kernel_config.pack(pady=5)
    tk.Label(frame_custom_kernel_config, text="Tamaño Kernel Pers.:").pack(side=tk.LEFT, padx=5)
    combo_tamano_kernel = ttk.Combobox(frame_custom_kernel_config, textvariable=custom_kernel_size,
                                         values=["3x3", "5x5", "7x7"])
    combo_tamano_kernel.pack(side=tk.LEFT, padx=5)
    combo_tamano_kernel.set("3x3")
    combo_tamano_kernel.bind("<<ComboboxSelected>>", lambda event: actualizar_interfaz_kernel_personalizado())

    frame_custom_kernel_input = tk.Frame(frame_controles) # Se llenará dinámicamente

    btn_generar_laplaciano = tk.Button(frame_controles, text="Generar Kernel Laplaciano", command=generar_kernel_laplaciano_default)
    btn_generar_laplaciano.pack(pady=5)

    btn_aplicar = tk.Button(frame_controles, text="Aplicar Detección de Bordes", command=aplicar_deteccion_bordes)
    btn_aplicar.pack(pady=10)

    actualizar_interfaz_kernel_personalizado() # Inicialmente oculto si no es Laplaciano Personalizado

    ventana_bordes.mainloop()
# --- Menu Principal ---
def main_menu():
    ventana_principal = tk.Tk()
    ventana_principal.title("Procesamiento de Imágenes")
    ventana_principal.geometry("400x300+100+100")

    lbl_instrucciones = tk.Label(ventana_principal, text="Seleccione una opción:")
    lbl_instrucciones.pack(pady=10)  # Espaciado vertical

    # --- Botones del Menú ---
    btn_umbralizacion = tk.Button(ventana_principal, text="1. Umbralización", command=umbralizacion_app)
    btn_umbralizacion.pack(fill=tk.X, padx=10, pady=5)  # Expande horizontalmente

    btn_transformaciones_intensidad = tk.Button(ventana_principal, text="2. Transformaciones de Intensidad",
                                             command=transformaciones_intensidad_app)
    btn_transformaciones_intensidad.pack(fill=tk.X, padx=10, pady=5)

    btn_transformaciones_tramos = tk.Button(ventana_principal, text="3. Transformaciones por Tramos",
                                         command=transformaciones_tramos_app)
    btn_transformaciones_tramos.pack(fill=tk.X, padx=10, pady=5)

    btn_procesamiento_local_hist = tk.Button(ventana_principal, text="4. Procesamiento Local Histograma",
                                           command=procesamiento_local_histograma_app)
    btn_procesamiento_local_hist.pack(fill=tk.X, padx=10, pady=5)

    btn_realce_local = tk.Button(ventana_principal, text="5. Realce Local Media y Varianza",
                                 command=realce_local_media_varianza_app)
    btn_realce_local.pack(fill=tk.X, padx=10, pady=5)

    btn_filtros_espaciales = tk.Button(ventana_principal, text="6. Filtros Espaciales",
                                      command=filtros_espaciales_app)
    btn_filtros_espaciales.pack(fill=tk.X, padx=10, pady=5)

    btn_deteccion_bordes = tk.Button(ventana_principal, text="7. Detección de Bordes", command=deteccion_bordes_app)
    btn_deteccion_bordes.pack(fill=tk.X, padx=10, pady=5)

    ventana_principal.mainloop()


if __name__ == "__main__":
    main_menu()