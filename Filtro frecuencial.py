import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Button, Label, filedialog, ttk, StringVar
from PIL import Image, ImageTk

# -------------------------------------
# Creación de máscaras de filtro frecuencial
# -------------------------------------
def crear_mascara(shape, tipo_paso, metodo, D0=30, D1=10, D2=60, n=2):
    M, N = shape
    u = np.arange(M)
    v = np.arange(N)
    U, V = np.meshgrid(v, u) # Correcto: primero v (columnas), luego u (filas)
    # Centrar las coordenadas para el cálculo de la distancia
    D = np.sqrt((U - N / 2) ** 2 + (V - M / 2) ** 2)


    if metodo == "ideal":
        if tipo_paso == "low":
            H = D <= D0
        elif tipo_paso == "high":
            H = D > D0
        elif tipo_paso == "band":
            # Asegurar D1 < D2 para un pasa-banda coherente
            if D1 >= D2: D1, D2 = D2, D1 # Intercambiar si D1 no es menor que D2
            H = np.logical_and(D >= D1, D <= D2)
        else:
            H = np.ones(shape) # Filtro neutro si el tipo no es reconocido

    elif metodo == "butterworth":
        if tipo_paso == "low":
            # Evitar división por cero si D es cero y D0 es cero (aunque D0 usualmente > 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                H = 1 / (1 + (D / D0) ** (2 * n))
            H[D == 0 & (D0 == 0)] = 1 # Caso especial para el centro si D0 es 0
        elif tipo_paso == "high":
            # Evitar división por cero si D es cero
            # Si D es 0, (D0/D) tiende a infinito, el denominador es infinito, H tiende a 0.
            # Sin embargo, la frecuencia cero (DC) debería ser completamente atenuada por un HPF ideal.
            # Para Butterworth, si D=0, H = 1 / (1 + (D0/0)**(2n)) -> 0
            # Si D es muy pequeño, D0/D es grande, (D0/D)**(2n) es grande, H es pequeño.
            with np.errstate(divide='ignore', invalid='ignore'):
                H = 1 / (1 + (D0 / D) ** (2 * n))
            H[D == 0] = 0 # Asegurar que la componente DC sea 0 para HPF
        elif tipo_paso == "band":
            if D1 >= D2: D1, D2 = D2, D1
            W = D2 - D1 # Ancho de banda
            # Evitar división por cero si D*W es cero
            with np.errstate(divide='ignore', invalid='ignore'):
                # Fórmula más común para Butterworth Pasa-Banda
                # H = 1 / (1 + ((D**2 - D0**2)/(D*W))**(2*n)) donde D0 es la frecuencia central sqrt(D1*D2)
                # O una aproximación usando producto de Pasa-Bajo y Pasa-Alto
                # Aquí D0 es la frecuencia central, W el ancho de banda.
                # Usaremos la fórmula provista en el código original, pero con cuidado para D=0
                # D0 se interpreta como la frecuencia central del pasabanda.
                # (D**2 - D0**2) / (D * (D2-D1))
                term = (D**2 - D0**2) / (D * W)
                H = 1 / (1 + term**(2*n))
            # Correcciones para D=0 o donde el término no esté bien definido.
            # La fórmula original `1 / (1 + (((D ** 2 - D0 ** 2) / (D * (D2 - D1))) ** (2 * n)))`
            # Asume D0 como la frecuencia central de la banda de rechazo para un filtro notch,
            # o D0 como frecuencia central de la banda de paso.
            # Para un filtro pasa-banda Butterworth, D0 es la frecuencia central geométrica.
            # Aquí usaré D0 como la frecuencia central aritmética (D1+D2)/2 y W = D2-D1
            # H_lp_D2 = 1 / (1 + (D / D2) ** (2 * n))  (Pasa bajo con corte en D2)
            # H_hp_D1 = 1 / (1 + (D1 / D) ** (2 * n))  (Pasa alto con corte en D1)
            # H = H_lp_D2 * H_hp_D1 (aproximación)
            # O la forma: 1 / (1 + ( (D*W) / (D**2 - D0_center**2) )**(2*n) ) donde D0_center es la frec central
            # La fórmula original puede tener problemas si D0 no es la frecuencia central geométrica.
            # Por simplicidad, usamos la fórmula original asegurando W > 0
            if W <= 0: W = 1 # Evitar división por cero o W negativo
            epsilon = 1e-8 # para evitar división por cero en D
            H = 1.0 / (1.0 + ((D**2 - D0**2) / (D * W + epsilon))**(2*n))
            # Para un pasa-banda, las frecuencias muy bajas y muy altas deben atenuarse.
            # Si D es muy bajo (cercano a 0), term puede ser grande y negativo. (D^2-D0^2) es negativo. (-)^2n es positivo.
            # Si D = D0, term = 0, H = 1 (centro de la banda).
        else:
            H = np.ones(shape)

    elif metodo == "gaussian":
        if tipo_paso == "low":
            H = np.exp(-(D ** 2) / (2 * (D0 ** 2)))
        elif tipo_paso == "high":
            H = 1 - np.exp(-(D ** 2) / (2 * (D0 ** 2)))
        elif tipo_paso == "band":
            if D1 >= D2: D1, D2 = D2, D1
            # D0 es la frecuencia central, (D2-D1) es el ancho de banda (W)
            # La fórmula original usa (D2-D1)/2 como 'sigma' para la gaussiana
            W = D2 - D1
            if W <= 0: W = 1 # Evitar sigma cero o negativo
            sigma_band = W / 2.0
            # D0 debería ser la frecuencia central (D1+D2)/2
            D_center = (D1 + D2) / 2.0
            H = np.exp(-((D - D_center) ** 2) / (2 * (sigma_band ** 2)))
            # Asegurar que H sea 0 lejos de la banda de paso
            # H[D < D1 - sigma_band*0.5] = 0 # Atenuación más agresiva fuera de la banda, si se desea
            # H[D > D2 + sigma_band*0.5] = 0
        else:
            H = np.ones(shape)
    else:
        H = np.ones(shape) # Filtro neutro si el método no es reconocido


    return H.astype(float)

# -------------------------------------
# Aplicar filtro frecuencial
# -------------------------------------
def aplicar_filtro_frecuencial(imagen, H):
    if imagen.ndim > 2: # Si es color, tomar el primer canal o convertir a gris
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gris = imagen

    f = np.fft.fft2(imagen_gris)
    f_shift = np.fft.fftshift(f)
    f_filtrado = f_shift * H
    f_ishift = np.fft.ifftshift(f_filtrado)
    img_filtrada = np.abs(np.fft.ifft2(f_ishift))
    return img_filtrada

# -------------------------------------
# Aplicar filtro Gaussiano Espacial (Convolución)
# -------------------------------------
def aplicar_filtro_gaussiano_espacial(imagen, kernel_size=(15,15), sigmaX=5):
    """Aplica un filtro Gaussiano pasa bajo en el dominio espacial."""
    if imagen.ndim > 2:
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        imagen_gris = imagen
    # Asegurar que el tamaño del kernel sea impar
    k_h = kernel_size[0] if kernel_size[0] % 2 != 0 else kernel_size[0] + 1
    k_w = kernel_size[1] if kernel_size[1] % 2 != 0 else kernel_size[1] + 1
    img_filtrada_espacial = cv2.GaussianBlur(imagen_gris, (k_h, k_w), sigmaX)
    return img_filtrada_espacial

# -------------------------------------
# Interfaz Gráfica
# -------------------------------------
class FiltroFrecuencialGUI:
    def __init__(self, root):
        self.root = root
        root.title("Filtrado Frecuencial y Espacial de Imágenes")

        # Variables de filtro
        self.metodo_var = StringVar(value="gaussian") # Cambiado a metodo_var
        self.tipo_paso_var = StringVar(value="low")   # Cambiado a tipo_paso_var

        # --- Controles ---
        controls_frame = ttk.Frame(root, padding="10")
        controls_frame.pack(pady=10, padx=10, fill="x")

        Button(controls_frame, text="Cargar Imagen", command=self.cargar_imagen).grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        Label(controls_frame, text="Tipo de Filtro Frecuencial:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        ttk.Combobox(controls_frame, textvariable=self.metodo_var, values=["ideal", "butterworth", "gaussian"]).grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        Label(controls_frame, text="Tipo de Paso Frecuencial:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        ttk.Combobox(controls_frame, textvariable=self.tipo_paso_var, values=["low", "high", "band"]).grid(row=2, column=1, padx=5, pady=2, sticky="ew")

        # Parámetros D0, D1, D2, n
        Label(controls_frame, text="D0 (Corte/Centro):").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        self.D0_var = StringVar(value="30")
        ttk.Entry(controls_frame, textvariable=self.D0_var, width=7).grid(row=3, column=1, padx=5, pady=2, sticky="w")

        Label(controls_frame, text="D1 (Banda Inferior):").grid(row=4, column=0, padx=5, pady=2, sticky="w")
        self.D1_var = StringVar(value="10")
        ttk.Entry(controls_frame, textvariable=self.D1_var, width=7).grid(row=4, column=1, padx=5, pady=2, sticky="w")

        Label(controls_frame, text="D2 (Banda Superior):").grid(row=5, column=0, padx=5, pady=2, sticky="w")
        self.D2_var = StringVar(value="60")
        ttk.Entry(controls_frame, textvariable=self.D2_var, width=7).grid(row=5, column=1, padx=5, pady=2, sticky="w")
        
        Label(controls_frame, text="n (Orden Butterworth):").grid(row=6, column=0, padx=5, pady=2, sticky="w")
        self.n_var = StringVar(value="2")
        ttk.Entry(controls_frame, textvariable=self.n_var, width=7).grid(row=6, column=1, padx=5, pady=2, sticky="w")


        Button(controls_frame, text="Aplicar Filtro", command=self.procesar_filtro).grid(row=7, column=0, columnspan=2, padx=5, pady=10, sticky="ew")

        controls_frame.columnconfigure(1, weight=1)

        # --- Previsualización de Imagen ---
        preview_frame = ttk.LabelFrame(root, text="Previsualización Imagen Cargada", padding="10")
        preview_frame.pack(pady=10, padx=10, fill="x")
        self.label_preview = Label(preview_frame)
        self.label_preview.pack()


    def cargar_imagen(self):
        path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif")])
        if not path:
            print("No se seleccionó ninguna imagen.")
            return

        try:
            self.img_color_original = cv2.imread(path)
            if self.img_color_original is None:
                print(f"Error: OpenCV no pudo cargar la imagen desde {path}")
                return

            self.img_rgb_original = cv2.cvtColor(self.img_color_original, cv2.COLOR_BGR2RGB)
            self.img_gray_original = cv2.cvtColor(self.img_color_original, cv2.COLOR_BGR2GRAY)

            # Previsualización más pequeña
            h, w = self.img_rgb_original.shape[:2]
            max_dim = 200
            if h > w:
                new_h = max_dim
                new_w = int(w * (max_dim / h))
            else:
                new_w = max_dim
                new_h = int(h * (max_dim / w))
            
            preview = cv2.resize(self.img_rgb_original, (new_w, new_h))
            self.imgtk = ImageTk.PhotoImage(Image.fromarray(preview))

            self.label_preview.configure(image=self.imgtk)
            self.label_preview.image = self.imgtk # Guardar referencia

            print("Imagen cargada exitosamente.")
        except Exception as e:
            print(f"Error al cargar la imagen: {e}")
            # Limpiar referencias si falla
            if hasattr(self, 'img_color_original'): del self.img_color_original
            if hasattr(self, 'img_rgb_original'): del self.img_rgb_original
            if hasattr(self, 'img_gray_original'): del self.img_gray_original


    def procesar_filtro(self):
        if not hasattr(self, "img_gray_original"):
            print("No hay imagen cargada.")
            return

        metodo = self.metodo_var.get()
        tipo = self.tipo_paso_var.get()

        try:
            D0 = float(self.D0_var.get())
            D1 = float(self.D1_var.get())
            D2 = float(self.D2_var.get())
            n_butter = int(self.n_var.get())
        except ValueError:
            print("Error: D0, D1, D2 deben ser números y 'n' un entero.")
            return


        M, N = self.img_gray_original.shape
        # D0_auto = min(M, N) // 6 # Ya no se usa D0 automático, se toma de la GUI
        # D1_auto = D0_auto // 2
        # D2_auto = D0_auto + 20

        H = crear_mascara((M, N), tipo, metodo, D0=D0, D1=D1, D2=D2, n=n_butter)

        # Aplicar filtro frecuencial a gris
        img_filt_gray_freq = aplicar_filtro_frecuencial(self.img_gray_original, H)

        # Aplicar filtro frecuencial a color (canal por canal)
        canales_filtrados_freq = []
        for i in range(3): # Para R, G, B
            canal = self.img_rgb_original[:, :, i]
            canal_filt_freq = aplicar_filtro_frecuencial(canal, H) # Reutilizar la función para un solo canal
            # Normalizar y convertir el canal filtrado si es necesario
            canal_filt_freq = np.clip(canal_filt_freq, 0, 255)
            canales_filtrados_freq.append(canal_filt_freq.astype(np.uint8))
        img_filt_color_freq = cv2.merge(canales_filtrados_freq)


        # Aplicar filtro Gaussiano espacial (pasa bajo por convolución) para comparación
        # Usar valores fijos o permitir configuración en GUI para kernel_size y sigmaX
        img_filt_gray_espacial_gauss = aplicar_filtro_gaussiano_espacial(self.img_gray_original, kernel_size=(15,15), sigmaX=5)


        # Mostrar resultados
        self.mostrar_resultados(
            self.img_gray_original,
            self.img_rgb_original,
            img_filt_gray_freq,
            img_filt_color_freq,
            H,
            img_filt_gray_espacial_gauss
        )

    def mostrar_resultados(self, img_gray_orig, img_color_orig, img_filt_gray_freq, img_filt_color_freq, H, img_filt_gray_espacial):
        plt.figure(figsize=(18, 10)) # Ajustado para 2x4

        # Fila 1
        plt.subplot(2, 4, 1)
        plt.title("Original (Gris)")
        plt.imshow(img_gray_orig, cmap='gray')
        plt.axis('off')

        fft_original_gray = np.fft.fftshift(np.fft.fft2(img_gray_orig))
        plt.subplot(2, 4, 2)
        plt.title("FFT Original (Gris)")
        plt.imshow(np.log1p(np.abs(fft_original_gray)), cmap='gray')
        plt.axis('off')

        plt.subplot(2, 4, 3)
        plt.title("Máscara Filtro Frecuencial")
        plt.imshow(H, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 4, 4)
        plt.title("Original (Color)")
        plt.imshow(img_color_orig) # img_color_orig es RGB
        plt.axis('off')

        # Fila 2
        plt.subplot(2, 4, 5)
        plt.title("Filtrada Frec. (Gris)")
        plt.imshow(img_filt_gray_freq, cmap='gray')
        plt.axis('off')

        fft_filtrada_gray_freq = np.fft.fftshift(np.fft.fft2(img_filt_gray_freq))
        plt.subplot(2, 4, 6)
        plt.title("FFT Filtrada Frec. (Gris)")
        plt.imshow(np.log1p(np.abs(fft_filtrada_gray_freq)), cmap='gray')
        plt.axis('off')
        
        plt.subplot(2, 4, 7)
        plt.title("Filtrada Espacial Gauss. (Gris)")
        plt.imshow(img_filt_gray_espacial, cmap='gray')
        plt.axis('off')

        plt.subplot(2, 4, 8)
        plt.title("Filtrada Frec. (Color)")
        plt.imshow(img_filt_color_freq) # img_filt_color_freq ya está en RGB
        plt.axis('off')


        plt.tight_layout(pad=1.5) # Añadir un poco de padding
        plt.show()

# -------------------------------------
# Ejecutar Aplicación
# -------------------------------------
if __name__ == "__main__":
    root = Tk()
    app = FiltroFrecuencialGUI(root)
    root.mainloop()