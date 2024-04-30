from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

def generarImagenOjoRecortada(image, lineas_ojo,lineas_iris, detection_result):
    
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

    def generate_iris_lines(lineas):
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
    
    # Dibujar líneas para los ojos en la máscara
    lista_lineas = generate_eye_lines(lineas_ojo) 
    
    
    # Cerrar el polígono agregando el primer punto al final de la lista
    puntos_poligono = np.vstack((lista_lineas, lista_lineas[0]))

    cv2.fillPoly(mascara, [puntos_poligono], (255,255,255))

    image = cv2.bitwise_not(image)

    # Aplicar la máscara binaria a la imagen original para obtener la región recortada
    region_recortada = cv2.bitwise_and(image, image, mask=mascara)

    region_recortada = cv2.bitwise_not(region_recortada)

    lista_lineas_iris = generate_iris_lines(lineas_iris)
    
    promedio_x = np.mean(lista_lineas_iris[:, 0])
    promedio_y = np.mean(lista_lineas_iris[:, 1])
    region_recortada[int(promedio_y), int(promedio_x)] = (0, 0, 255)
    # Encontrar los límites de la región recortada
    (x, y, w, h) = cv2.boundingRect(mascara)
    # Recortar la región recortada utilizando los límites encontrados
    region_recortada = region_recortada[y:y+h, x:x+w]

    # region_recortada = cv2.medianBlur(region_recortada, 3)

    region_recortada = cv2.GaussianBlur(region_recortada, (3,3), 0)

    # Cambia la imagen a gris
    # gris = cv2.cvtColor(region_recortada, cv2.COLOR_BGR2GRAY)

    # Emborrona un poco el area del circulo
    # gris = cv2.GaussianBlur(gris, (3,3), 0)

    # Dejar en blanco todo lo que este en negro y negro todo lo demás
    # _, umbral = cv2.threshold(gris, 31, 255, cv2.THRESH_BINARY_INV)

    # umbral = cv2.cvtColor(gris, cv2.COLOR_GRAY2RGB)

    # return gris

    # contornos, _ = cv2.findContours(umbral, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # contornos = sorted(contornos, key=lambda x: cv2.contourArea(x), reverse=True)

    # Dejar para el final --------------------------------------------------------------------------- 
    # Pintar el pixel del centro del rectángulo en rojo
    # forma = umbral.shape
    # y_centro = int(forma[0]/2)
    # x_centro = int(forma[1]/2)

    
    # umbral[y_centro, x_centro] = (0, 0, 255)

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

lineas_iris_izquierdo = [(474, 475), (475, 476), (476, 477),
                                 (477, 474)]

lineas_iris_derecho = [(469, 470), (470, 471), (471, 472),
                                 (472, 469)]

# captura = cv2.VideoCapture('http://192.168.1.3:4747/video')  # 0 para la cámara integrada, puedes cambiarlo según el índice de la cámara
captura = cv2.VideoCapture(0)
while True:
    # Capturar frame por frame
    ret, frame = captura.read()

    # Si no se pudo capturar el frame, salir del bucle
    if not ret:
        break
    
    frame = cv2.flip(frame,1)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    try:
    # STEP 4: Detect face landmarks from the input image.
        detection_result = detector.detect(mp_image)
        # print(detection_result)
        # Mostrar el frame
        recorte_ojo_izquierdo = generarImagenOjoRecortada(frame, lineas_ojo_izquierdo,lineas_iris_izquierdo, detection_result)
        recorte_ojo_derecho = generarImagenOjoRecortada(frame, lineas_ojo_derecho,lineas_iris_derecho, detection_result)


        recorte_grande_izquierdo = cv2.resize(recorte_ojo_izquierdo, None, fx=10, fy=10)

        cv2.imshow("recorte_ojo_izquierdo", recorte_grande_izquierdo)
        cv2.imshow("recorte_ojo_derecho", recorte_ojo_derecho)
    except:
        print("no se ve")
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar todas las ventanas
captura.release()
cv2.destroyAllWindows()
