#iniziato giorno: 15/03/2025
#completato giorno: 17/03/2025 -- 22:16

#import di varie librerie
import cv2
from matplotlib import pyplot as plt
import numpy as np

from util import get_parking_spots_bboxes
from util import empty_or_not

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

#definizione percorso della maschera da usare nel singolo video
#maschere:
#C:/Users/drj4k/OneDrive/Desktop/AI_projects_Fantini/progettoParcheggi/videoParcheggioEsimili/maschere/mask_1920_1080.png
#C:/Users/drj4k/OneDrive/Desktop/AI_projects_Fantini/progettoParcheggi/videoParcheggioEsimili/maschere/mask_crop.png
mask = './ParkingProjectAI/progettoParcheggi/videoParcheggioEsimili/maschere/mask_1920_1080.png'

#definizione percorso del singolo video da prendere come esempio
#video:
#C:/Users/drj4k/OneDrive/Desktop/AI_projects_Fantini/progettoParcheggi/videoParcheggioEsimili/videoELoop/parking_1920_1080.mp4
#C:/Users/drj4k/OneDrive/Desktop/AI_projects_Fantini/progettoParcheggi/videoParcheggioEsimili/videoELoop/parking_1920_1080_loop.mp4
#C:/Users/drj4k/OneDrive/Desktop/AI_projects_Fantini/progettoParcheggi/videoParcheggioEsimili/videoELoop/parking_crop.mp4
#C:/Users/drj4k/OneDrive/Desktop/AI_projects_Fantini/progettoParcheggi/videoParcheggioEsimili/videoELoop/parking_crop_loop.mp4
video_path = './ParkingProjectAI/progettoParcheggi/videoParcheggioEsimili/videoELoop/parking_1920_1080_loop.mp4'

mask = cv2.imread(mask, 0)
cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

spots = get_parking_spots_bboxes(connected_components)

spots_status = [None for j in spots]
diffs = [None for j in spots]

previous_frame = None

ret = True
step = 30
frame_num = 0

while ret:
    ret, frame = cap.read()
    
    if frame_num % step == 0 and previous_frame is not None:
        
        for spot_index, spot in enumerate(spots):
            x1, y1, w, h = spot
            
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            
            diffs[spot_index] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])
        
        print([diffs[j] for j in np.argsort(diffs)][::-1])
        
    if frame_num % step == 0:
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
            
        for spot_index in arr_:
            spot = spots[spot_index]
            x1, y1, w, h = spot
            
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_index] = spot_status
            
    if frame_num % step == 0:
        previous_frame = frame.copy()
    
    for spot_index, spot in enumerate(spots):
        spot_status = spots_status[spot_index]
        
        x1, y1, w, h = spots[spot_index]
        
        if spot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
    
    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, 'Posti disponibili {} / {}'.format(str(sum(spots_status)), str(len(spots_status))), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
    frame_num += 1

cap.release()
cv2.destroyAllWindows()