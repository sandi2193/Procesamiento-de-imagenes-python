import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Button, filedialog

# Cargar el modelo de detección de rostros (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Función de supresión de no-máximos CORREGIDA
def non_max_suppression(rostros, overlapThresh=0.3):
    if len(rostros) == 0:
        return []

    # Si las cajas delimitadoras son enteros, convertirlas a flotantes
    # ya que realizaremos algunas divisiones
    if rostros.dtype.kind == "i":
        rostros = rostros.astype("float")

    # Inicializar la lista de índices seleccionados
    seleccionadas_indices = []

    # Obtener las coordenadas de las cajas delimitadoras
    x1 = rostros[:,0]
    y1 = rostros[:,1]
    w_orig = rostros[:,2] # ancho original
    h_orig = rostros[:,3] # alto original
    x2 = x1 + w_orig
    y2 = y1 + h_orig

    # Calcular el área de las cajas delimitadoras y ordenar las cajas
    # por la coordenada y inferior de la caja delimitadora (o por área como en tu original)
    # Continuaremos usando el área para mantener la consistencia con tu código original.
    area = w_orig * h_orig
    idxs = np.argsort(area)[::-1] # Ordenar por área, de mayor a menor

    # Bucle mientras queden algunos índices en la lista de índices
    while len(idxs) > 0:
        # Tomar el último índice en la lista de índices (el de mayor área en este caso)
        # y agregar el índice a la lista de seleccionados
        last_pos = len(idxs) - 1 # Posición del último elemento
        i_original_idx = idxs[last_pos] # Índice original del rostro actual
        seleccionadas_indices.append(i_original_idx)

        # Crear una lista de índices a suprimir (para np.delete)
        # Incluye el índice actual 'last_pos'
        suppress_positions = [last_pos]

        # Iterar sobre los índices restantes en idxs (todos excepto el último)
        for pos_in_idxs in range(last_pos): # Iterar sobre las posiciones 0 a last_pos-1
            # Obtener el índice original del rostro a comparar
            j_original_idx = idxs[pos_in_idxs]

            # Encontrar las coordenadas (x,y) más grandes para el inicio de
            # la caja delimitadora y las coordenadas (x,y) más pequeñas
            # para el final de la caja delimitadora
            xx1 = np.maximum(x1[i_original_idx], x1[j_original_idx])
            yy1 = np.maximum(y1[i_original_idx], y1[j_original_idx])
            xx2 = np.minimum(x2[i_original_idx], x2[j_original_idx])
            yy2 = np.minimum(y2[i_original_idx], y2[j_original_idx])

            # Calcular el ancho y alto de la caja de intersección
            w_intersect = np.maximum(0, xx2 - xx1)
            h_intersect = np.maximum(0, yy2 - yy1)
            intersection_area = w_intersect * h_intersect

            # Calcular la relación de superposición como IoU (Intersection over Union)
            # IoU = IntersectionArea / (Area_i + Area_j - IntersectionArea)
            # O la métrica que usaste: IntersectionArea / Area_j
            # Mantendremos tu métrica original para la menor disrupción:
            overlap = intersection_area / area[j_original_idx]
            # Alternativamente, para IoU:
            # union_area = area[i_original_idx] + area[j_original_idx] - intersection_area
            # overlap = intersection_area / union_area if union_area > 0 else 0


            # Si hay suficiente superposición, añadir la posición del índice a la lista de supresión
            if overlap > overlapThresh:
                suppress_positions.append(pos_in_idxs)

        # Eliminar de idxs los índices en las posiciones de supresión
        idxs = np.delete(idxs, suppress_positions)

    # Devolver solo las cajas delimitadoras seleccionadas (como enteros)
    return rostros[seleccionadas_indices].astype("int")


# Función para detectar rostros con el modelo preentrenado
def detectar_rostros(imagen):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostros con el clasificador Haar Cascade
    rostros_crudos = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Aplicar supresión de no-máximos (NMS) para eliminar detecciones redundantes
    if len(rostros_crudos) > 0:
        # Usar la función NMS corregida
        rostros_procesados = non_max_suppression(rostros_crudos)
    else:
        rostros_procesados = [] # Devuelve una lista vacía si no hay detecciones crudas
    
    return rostros_procesados

# Función para visualizar los resultados
def visualizar_resultados(imagen, rostros):
    for (x, y, w, h) in rostros:
        cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Mostrar la imagen con los rostros detectados
    plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Función para cargar una imagen y detectar rostros
def cargar_imagen():
    path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.png *.bmp")])
    if not path:
        return

    imagen = cv2.imread(path)
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen desde {path}")
        return
        
    rostros = detectar_rostros(imagen)

    if len(rostros) > 0:
        print(f"Rostros detectados: {len(rostros)}")
        visualizar_resultados(imagen.copy(), rostros) # Usar .copy() para no dibujar sobre la original si se reutiliza
    else:
        print("No se detectaron rostros.")

# Interfaz gráfica para cargar imágenes
class DeteccionRostrosGUI:
    def __init__(self, root_window):
        self.root = root_window
        root_window.title("Detección de Rostros")
        
        # Botón para cargar imagen
        Button(root_window, text="Cargar Imagen", command=cargar_imagen).pack(pady=20)

# Ejecutar la aplicación
if __name__ == "__main__":
    root = Tk()
    app = DeteccionRostrosGUI(root)
    root.mainloop()