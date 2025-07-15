import mediapipe as mp
import cv2 as cv
from scipy.spatial import distance as dis
import threading
import winsound
import time
import matplotlib.pyplot as plt

def alerta_sonoro(frequencia, duracao):
    winsound.Beep(frequencia, duracao)

frequencia = 500
duracao = 1000

def draw_landmarks(image, outputs, land_mark, color):
    height, width = image.shape[:2]

    for face in land_mark:
        point = outputs.multi_face_landmarks[0].landmark[face]

        point_scale = ((int)(point.x * width), (int)(point.y*height))

        cv.circle(image, point_scale, 2, color, 1)

def euclidean_distance(image, top, bottom):
    height, width = image.shape[0:2]

    point1 = int(top.x * width), int(top.y * height)
    point2 = int(bottom.x * width), int(bottom.y * height)

    distance = dis.euclidean(point1, point2)
    return distance

def get_aspect_ratio(image, outputs, top_bottom, left_right):
    landmark = outputs.multi_face_landmarks[0]

    top = landmark.landmark[top_bottom[0]]
    bottom = landmark.landmark[top_bottom[1]]

    top_bottom_dis = euclidean_distance(image, top, bottom)

    left = landmark.landmark[left_right[0]]
    right = landmark.landmark[left_right[1]]

    left_right_dis = euclidean_distance(image, left, right)

    aspect_ratio = left_right_dis / top_bottom_dis

    return aspect_ratio

face_mesh = mp.solutions.face_mesh
hand_mesh = mp.solutions.hands
draw_utils = mp.solutions.drawing_utils
landmark_style = draw_utils.DrawingSpec((0, 255, 0), thickness=1, circle_radius=1)
connection_style = draw_utils.DrawingSpec((0, 0, 255), thickness=1, circle_radius=1)

STATIC_IMAGE = False
MAX_NO_FACES = 2
DETECTION_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.5

COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)

LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
        185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

LEFT_EYE_TOP_BOTTOM = [386, 374]
LEFT_EYE_LEFT_RIGHT = [263, 362]

RIGHT_EYE_TOP_BOTTOM = [159, 145]
RIGHT_EYE_LEFT_RIGHT = [133, 33]

UPPER_LOWER_LIPS = [13, 14]
LEFT_RIGHT_LIPS = [78, 308]

FACE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
        377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

face_model = face_mesh.FaceMesh(static_image_mode=STATIC_IMAGE,
                                max_num_faces=MAX_NO_FACES,
                                min_detection_confidence=DETECTION_CONFIDENCE,
                                min_tracking_confidence=TRACKING_CONFIDENCE)

hand_model = hand_mesh.Hands()

capture = cv.VideoCapture(0)

frame_count = 0
min_frame = 6
min_tolerance = 5.0

# Inicialização das listas para armazenar os valores de aspect_ratio e timestamps
eye_aspect_ratios = []
timestamps = []

plt.ion()  # Ativar o modo interativo do Matplotlib

while True:
    result, image = capture.read()

    if result:
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        outputs = face_model.process(image_rgb)
        outputs_hand = hand_model.process(image_rgb)

        if outputs.multi_face_landmarks and len(outputs.multi_face_landmarks) > 1:
            # Mais de um rosto detectado, acionar o alarme
            alerta_sonoro(frequencia, duracao)
            print("Mais de um rosto detectado!")

        elif outputs.multi_face_landmarks:
            draw_landmarks(image, outputs, FACE, COLOR_RED)
            draw_landmarks(image, outputs, LEFT_EYE_TOP_BOTTOM, COLOR_BLUE)
            draw_landmarks(image, outputs, LEFT_EYE_LEFT_RIGHT, COLOR_BLUE)

            ratio_left = get_aspect_ratio(image, outputs, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)

            draw_landmarks(image, outputs, RIGHT_EYE_TOP_BOTTOM, COLOR_BLUE)
            draw_landmarks(image, outputs, RIGHT_EYE_LEFT_RIGHT, COLOR_BLUE)

            ratio_right = get_aspect_ratio(image, outputs, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)

            ratio = (ratio_left + ratio_right) / 2.0

            if ratio > min_tolerance:
                frame_count += 1
            else:
                frame_count = 0

            if frame_count > min_frame:  # olhos
                alerta_sonoro(frequencia, duracao)

            draw_landmarks(image, outputs, UPPER_LOWER_LIPS, COLOR_BLUE)
            draw_landmarks(image, outputs, LEFT_RIGHT_LIPS, COLOR_BLUE)

            ratio_lips = get_aspect_ratio(image, outputs, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)
            if ratio_lips < 1.8:
                alerta_sonoro(frequencia, duracao)

            # Adicionar valores de aspect_ratio e timestamps às listas
            timestamps.append(time.time())
            eye_aspect_ratios.append(ratio)

            # Limite o histórico de valores armazenados
            if len(timestamps) > 100:
                timestamps = timestamps[-100:]
                eye_aspect_ratios = eye_aspect_ratios[-100:]

            # Plotar o gráfico em tempo real
            plt.clf()
            plt.plot(timestamps, eye_aspect_ratios, label='Eye Aspect Ratio')
            plt.xlabel('Timestamp')
            plt.ylabel('Aspect Ratio')
            plt.title('Eye Aspect Ratio over Time')
            plt.legend()
            plt.draw()
            plt.pause(0.01)

        if outputs_hand.multi_hand_landmarks:
            num_hands = len(outputs_hand.multi_hand_landmarks)

            for landmarks in outputs_hand.multi_hand_landmarks:
                draw_utils.draw_landmarks(image, landmarks, hand_mesh.HAND_CONNECTIONS,
                                          landmark_drawing_spec=landmark_style, connection_drawing_spec=connection_style)

            if num_hands == 2:
                alerta_sonoro(frequencia, duracao)
                print(f"Número de mãos detectadas: {num_hands}")

        cv.imshow("Teste Reconhecimento", image)
        if cv.waitKey(1) & 255 == 27:
            break

capture.release()
cv.destroyAllWindows()
