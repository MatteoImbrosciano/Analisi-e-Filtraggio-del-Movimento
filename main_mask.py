import cv2
import torch
import numpy as np
from tracker_2 import ObjectTracker
from metrics import calculate_video_metrics

# Definizione della funzione per applicare una maschera al frame
def apply_mask(frame):
    mask = np.zeros(frame.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (500, 340), (800, 720), 255, -1)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    return masked_frame

# Carica il modello YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Creazione dell'oggetto tracker
tracker = ObjectTracker()

# Apertura del video
#video_path = "C:\\Users\\matte\\OneDrive\\Desktop\\Multimedia\\esame\\video_1.mp4"
#video_path = "C:\\Users\\matte\\OneDrive\\Desktop\\Multimedia\\esame\\video_2.mp4"
video_path = "C:\\Users\\matte\\OneDrive\\Desktop\\Multimedia\\esame\\video_3.mp4"
cap = cv2.VideoCapture(video_path)

# ID della classe per le persone (generalmente 0 per i dataset come COCO)
class_id_for_person = 0

# Contatore totale delle persone rilevate
total_persons_detected = 0
detected_ids = set()

frames = []  # Lista per raccogliere i frame per il calcolo delle metriche

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Applica la maschera al frame
    frame = apply_mask(frame)

    # Rilevamento degli oggetti con YOLOv5 sul frame mascherato
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    # Prepara la lista dei rettangoli per il tracker
    rects = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = int(det[0]), int(det[1]), int(det[2]), int(det[3]), det[4], int(det[5])
        if conf > 0.3 and cls == class_id_for_person:
            rects.append([x1, y1, x2 - x1, y2 - y1])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Person {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    frames.append(frame.copy())  # Aggiungi il frame modificato alla lista

    # Aggiorna il tracker con i nuovi rettangoli
    objects = tracker.update(rects)
    
    # Conta le persone rilevate in questo frame
    for obj in objects:
        obj_id = obj[-1]
        if obj_id not in detected_ids:
            detected_ids.add(obj_id)
            total_persons_detected += 1

    # Mostra il conteggio delle persone nel frame e totali
    cv2.putText(frame, f'Persons in this frame: {len(objects)}', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Total persons detected: {total_persons_detected}', (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Visualizzazione del frame con i rilevamenti
    cv2.imshow('YOLOv5 Masked Frame', frame)

    key = cv2.waitKey(30)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

# Calcolo delle metriche dopo l'elaborazione di tutti i frame
avg_mse, avg_psnr, avg_ssim = calculate_video_metrics(frames)
if avg_mse is not None:
    print(f"Media MSE: {avg_mse}")
    print(f"Media PSNR: {avg_psnr}")
    print(f"Media SSIM: {avg_ssim}")
else:
    print("Impossibile calcolare le metriche. Assicurati di avere almeno due frame nel video.")
