from ultralytics import YOLO
import cv2

#detecção de objetos 

model = YOLO('yolov8n.pt')

resultados = model.predict(source="0", show=True)

print(resultados)

