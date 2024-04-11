import cv2
import numpy as np
from metrics import calculate_video_metrics

video_path = "C:\\Users\\matte\\OneDrive\\Desktop\\Multimedia\\esame\\video_1.mp4"
#video_path = "C:\\Users\\matte\\OneDrive\\Desktop\\Multimedia\\esame\\video_2.mp4"
#video_path = "C:\\Users\\matte\\OneDrive\\Desktop\\Multimedia\\esame\\video_3.mp4"

cap = cv2.VideoCapture(video_path)

prev_frame = None
prev_roi = None
frames = []

# Lettura e memorizzazione dei frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

# Calcolo del PSNR e MSE medi
avg_mse, avg_psnr, avg_ssim = calculate_video_metrics(frames)

# Verifica se le metriche sono state calcolate con successo
if avg_mse is not None and avg_psnr is not None:
    # Visualizzazione dei risultati
    print("Media MSE:", avg_mse)
    print("Media PSNR:", avg_psnr)
    print("Media SSIM:", avg_ssim)
else:
    print("Impossibile calcolare le metriche. Assicurati di avere almeno due frame nel video.")
    
# Rilascio delle risorse
cap.release()
cv2.destroyAllWindows()