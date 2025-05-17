import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# ---------------------------
# 1. Seleccionar imagen
# ---------------------------
Tk().withdraw()
file_path = filedialog.askopenfilename(title='Selecciona una imagen',
                                       filetypes=[("Archivos de imagen", "*.jpg *.png *.jpeg *.bmp")])
if not file_path:
    raise ValueError("No se seleccionó ninguna imagen.")

img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread(file_path)
img_color_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

if img_gray is None or img_color is None:
    raise ValueError("No se pudo cargar la imagen.")

# ---------------------------
# 2. Algoritmo de crecimiento de regiones (gris)
# ---------------------------
def region_growing_gray(image, seed, thresh):
    visited = np.zeros_like(image, dtype=bool)
    region = np.zeros_like(image, dtype=np.uint8)
    h, w = image.shape
    seed_value = image[seed[1], seed[0]]
    stack = [seed]

    while stack:
        x, y = stack.pop()
        if visited[y, x]:
            continue
        visited[y, x] = True
        diff = abs(int(image[y, x]) - int(seed_value))
        if diff <= thresh:
            region[y, x] = 255
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if not visited[ny, nx]:
                            stack.append((nx, ny))
    return region

# ---------------------------
# 3. Selección de semilla con clic
# ---------------------------
def on_click(event):
    if event.xdata is None or event.ydata is None:
        return
    x, y = int(event.xdata), int(event.ydata)
    plt.close()

    # ---------------------------
    # 4. Procesamiento tras clic
    # ---------------------------
    print(f"Semilla seleccionada: ({x}, {y})")

    threshold_gray = 10
    region_gray = region_growing_gray(img_gray, (x, y), threshold_gray)

    # Comparación con umbral fijo
    _, threshold_img = cv2.threshold(img_gray, img_gray[y, x], 255, cv2.THRESH_BINARY)

    # Crecimiento en color
    h, w, _ = img_color.shape
    visited = np.zeros((h, w), dtype=bool)
    region_mask = np.zeros((h, w), dtype=np.uint8)
    seed_color = img_color[y, x]
    threshold_color = 30
    stack = [(x, y)]

    while stack:
        x2, y2 = stack.pop()
        if visited[y2, x2]:
            continue
        visited[y2, x2] = True
        pixel = img_color[y2, x2]
        diff = np.linalg.norm(pixel.astype(int) - seed_color.astype(int))
        if diff <= threshold_color:
            region_mask[y2, x2] = 1
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x2 + dx, y2 + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if not visited[ny, nx]:
                            stack.append((nx, ny))

    img_result = img_color_rgb.copy()
    img_result[region_mask == 1] = [255, 0, 0]

    # Visualización de resultados
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.title("Imagen en escala de grises")
    plt.imshow(img_gray, cmap='gray')

    plt.subplot(2, 2, 2)
    plt.title("Segmentación por crecimiento")
    plt.imshow(region_gray, cmap='gray')

    plt.subplot(2, 2, 3)
    plt.title("Segmentación por umbral")
    plt.imshow(threshold_img, cmap='gray')

    plt.subplot(2, 2, 4)
    plt.title("Crecimiento en imagen a color")
    plt.imshow(img_result)

    plt.tight_layout()
    plt.show()

# ---------------------------
# 5. Mostrar imagen y esperar clic
# ---------------------------
fig, ax = plt.subplots()
ax.set_title("Haz clic para seleccionar semilla")
ax.imshow(img_gray, cmap='gray')
cid = fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()
