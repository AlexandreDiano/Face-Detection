import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('../videos/10.mp4')
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()
while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            score = int(detection.score[0]*100)
            cv2.putText(img, f'{score}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)






            # cv2.putText(img, f'{score}%', (5, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            #if using mask
            # if score:
            #     if (score > 92):
            #         cv2.putText(img, f'Nao Esta Usando Mascara', (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 10), 2)
            #     elif (score > 85) and (score <= 90) :
            #         cv2.putText(img, f'Mascara Errada', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            #     else:
            #         cv2.putText(img, f'Usando Mascara, OK', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    # cv2.putText(img, "teste", (5, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow('Image', img)
    cv2.waitKey(10)
