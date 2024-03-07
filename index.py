from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

def generarImagenOjoRecortada(image, lineas_ojo, detection_result):
    
# Definir las dimensiones de la imagen
    image_rows, image_cols, _ = image.shape

    # Crear una máscara de ceros del mismo tamaño que la imagen original
    mascara = np.zeros((image_rows, image_cols), dtype=np.uint8)
    def generate_eye_lines(lineas):
        lista_lineas = []
        for linea in lineas:
            punto_1 = detection_result.face_landmarks[0][linea[0]]
            punto_2 = detection_result.face_landmarks[0][linea[1]]
            punto_1_normalizado = solutions.drawing_utils._normalized_to_pixel_coordinates(punto_1.x, punto_1.y, image_cols, image_rows)
            punto_2_normalizado = solutions.drawing_utils._normalized_to_pixel_coordinates(punto_2.x, punto_2.y, image_cols, image_rows)
            # linea = [(int(punto_1_normalizado[0]), int(punto_1_normalizado[1])), (int(punto_2_normalizado[0]), int(punto_2_normalizado[1]))]
            lista_lineas.append((int(punto_1_normalizado[0]), int(punto_1_normalizado[1])))
            lista_lineas.append((int(punto_2_normalizado[0]), int(punto_2_normalizado[1])))
        return np.array(lista_lineas)
    # Lineas para el ojo izquierdo


    # Dibujar líneas para los ojos en la máscara
    lista_lineas = generate_eye_lines(lineas_ojo) 

    # Cerrar el polígono agregando el primer punto al final de la lista
    puntos_poligono = np.vstack((lista_lineas, lista_lineas[0]))

    cv2.fillPoly(mascara, [puntos_poligono], (255,255,255))

    # Aplicar la máscara binaria a la imagen original para obtener la región recortada
    region_recortada = cv2.bitwise_and(image, image, mask=mascara)

    # Encontrar los límites de la región recortada
    (x, y, w, h) = cv2.boundingRect(mascara)
    # Recortar la región recortada utilizando los límites encontrados
    region_recortada = region_recortada[y:y+h, x:x+w]

    return region_recortada

# ---------------------------------------------------------------------------------------------------

base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

lineas_ojo_izquierdo = [(263, 249), (249, 390), (390, 373), (373, 374),
                        (374, 380), (380, 381), (381, 382), (382, 362),
                        (263, 466), (466, 388), (388, 387), (387, 386),
                        (386, 385), (385, 384), (384, 398), (398, 362)]

lineas_ojo_derecho = [(33, 7), (7, 163), (163, 144), (144, 145),
                      (145, 153), (153, 154), (154, 155), (155, 133),
                      (33, 246), (246, 161), (161, 160), (160, 159),
                      (159, 158), (158, 157), (157, 173), (173, 133)]


captura = cv2.VideoCapture(0)  # 0 para la cámara integrada, puedes cambiarlo según el índice de la cámara

while True:
    # Capturar frame por frame
    ret, frame = captura.read()

    # Si no se pudo capturar el frame, salir del bucle
    if not ret:
        break

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # STEP 4: Detect face landmarks from the input image.
    detection_result = detector.detect(mp_image)
    # print(detection_result)
    # Mostrar el frame
    recorte_ojo_izquierdo = generarImagenOjoRecortada(frame, lineas_ojo_izquierdo, detection_result)
    recorte_ojo_derecho = generarImagenOjoRecortada(frame, lineas_ojo_derecho, detection_result)


    recorte_grande_izquierdo = cv2.resize(recorte_ojo_izquierdo, None, fx=10, fy=10)

    cv2.imshow("recorte_ojo_izquierdo", recorte_grande_izquierdo)
    cv2.imshow("recorte_ojo_derecho", recorte_ojo_derecho)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar todas las ventanas
captura.release()
cv2.destroyAllWindows()
