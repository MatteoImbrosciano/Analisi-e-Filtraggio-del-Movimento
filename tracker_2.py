import cv2
import torch
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class ObjectTracker:
    def __init__(self, max_distance=50, max_missed_frames=10):
        self.center_points = {}
        self.id_count = 0
        self.kalman_filters = {}
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.missed_frames = {}

    def update(self, objects_rect):
        objects_bbs_ids = []
        new_center_points = {}

        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + w / 2)
            cy = (y + h / 2)
            min_dist = self.max_distance
            best_match_id = None

            for obj_id, kf in self.kalman_filters.items():
                kf.predict()
                predicted = kf.x[:2].reshape((2,))
                dist = np.linalg.norm([cx, cy] - predicted)

                if dist < min_dist:
                    min_dist = dist
                    best_match_id = obj_id

            if best_match_id is not None:
                kf = self.kalman_filters[best_match_id]
                kf.update(np.array([[cx], [cy]]))
                self.center_points[best_match_id] = kf.x[:2].reshape((2,)).astype(int)
                objects_bbs_ids.append([x, y, w, h, best_match_id])
                new_center_points[best_match_id] = True
                self.missed_frames[best_match_id] = 0
            else:
                kf = self._create_kalman_filter(cx, cy)
                self.kalman_filters[self.id_count] = kf
                self.center_points[self.id_count] = np.array([cx, cy]).astype(int)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                new_center_points[self.id_count] = True
                self.missed_frames[self.id_count] = 0
                self.id_count += 1

        self._cleanup_objects(new_center_points)
        return objects_bbs_ids

    def _create_kalman_filter(self, cx, cy):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x = np.array([[cx], [cy], [0], [0]])  # Stato iniziale
        kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])  # Transizione di stato
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # Misurazione
        kf.R = np.eye(2) * 10  # Errore di misurazione
        kf.P *= 1000  # Stima errore
        kf.Q = Q_discrete_white_noise(dim=4, dt=1, var=0.01)  # Process noise
        return kf

    def _cleanup_objects(self, new_center_points):
        for obj_id in list(self.missed_frames.keys()):
            if obj_id not in new_center_points:
                self.missed_frames[obj_id] += 1
                if self.missed_frames[obj_id] > self.max_missed_frames:
                    del self.kalman_filters[obj_id]
                    del self.center_points[obj_id]
                    del self.missed_frames[obj_id]

# Segue il resto del codice per applicare il filtro mediano, rilevare le persone e tracciarle.
# Nota: il codice sotto questa linea deve essere integrato con la parte superiore.

# Assicurati di sostituire il path del video con il percorso corretto dove è salvato il tuo video di input.
