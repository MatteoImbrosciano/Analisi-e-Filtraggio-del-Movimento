import cv2
import torch
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

def calculate_video_metrics(frames):
    if len(frames) < 2:
        return None, None, None
    
    total_mse, total_psnr, total_ssim = 0, 0, 0
    frame_count = len(frames)
    
    for i in range(frame_count - 1):
        frame1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frames[i+1], cv2.COLOR_BGR2GRAY)
        
        mse = np.mean((frame1 - frame2) ** 2)
        if mse == 0:
            psnr = 100
        else:
            PIXEL_MAX = 255.0
            psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
            
        ssim = compare_ssim(frame1, frame2)
        
        total_mse += mse
        total_psnr += psnr
        total_ssim += ssim
    
    avg_mse = total_mse / (frame_count - 1)
    avg_psnr = total_psnr / (frame_count - 1)
    avg_ssim = total_ssim / (frame_count - 1)
    
    return avg_mse, avg_psnr, avg_ssim
