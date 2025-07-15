import cv2
import time
import winsound

COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open('coco.names.txt', 'r') as f:
    class_names = [cname.strip() for cname in f.readlines()]

cap = cv2.VideoCapture(0)

net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

# Define a função para reproduzir o som

def play_sound(frequencia, duracao):
    winsound.Beep(frequencia, duracao)

frequencia = 900  
duracao = 1000  


while True:
    _, frame = cap.read()
    start = time.time()
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)
    end = time.time()

    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = f"{class_names[int(classid)]} : {score}"
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Se um celular for detectado (você pode ajustar o índice para o label correto do celular)
        if class_names[int(classid)] == 'cell phone':
            play_sound(frequencia, duracao)

    

    cv2.imshow("teste", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
