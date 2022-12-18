import cv2
import time
import numpy as np
import mediapipe as mp
from config import gesturelist, Config, Vector
from PIL import ImageFont, ImageDraw, Image

# 데이터 변수 설정
file = np.genfromtxt(
    Config.trained_data_path,
    delimiter=",",
)

font = ImageFont.truetype(Config.font_path, 25)  # 텍스트 표출시 폰트 설정
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)

# K-NN 알고리즘 객체 생성
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

# image process
_sum = []  # 텍스트 전체 저장 리스트
cap = cv2.VideoCapture(1)

def gen_frames():
    StartTime = time.time()
    with mp_hands.Hands(
        max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7
    ) as hands:
        # 무한루프 문으로 웹캠 생성
        while cap.isOpened():
            (grabbed, frame) = cap.read()
            flag = []

            prev_index = 0
            sentence = ""
            recognizeDelay = 0.1

            image = frame
            video = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            video = cv2.flip(video, 1)

            video.flags.writeable = False

            result = hands.process(video)

            video.flags.writeable = True
            video = cv2.cvtColor(video, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                # 손 인식 했을 경우
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 3))
                    # 21개 손가락 마디 부분 좌표 (x, y, z)를 joint 변수에 저장
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z]
                    # 벡터 값 계산
                    v1 = joint[
                        Vector.joint_v1,
                        :,
                    ]
                    v2 = joint[
                        Vector.joint_v2,
                        :,
                    ]
                    v = v2 - v1

                    # 벡터 길이 계산
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                    angle = np.arccos(
                        np.einsum(
                            "nt,nt->n",
                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :],
                        )
                    )

                    angle = np.degrees(angle)
                    data = np.array([angle], dtype=np.float32)
                    ret, results, neighbours, dist = knn.findNearest(data, 3)
                    index = int(results[0][0])

                    if index in gesturelist.keys():
                        if time.time() - StartTime > 3:
                            StartTime = time.time()
                            if index == 6:
                                _sum.clear()
                            else:
                                _sum.append(gesturelist[index])
                                # 인식 성공한 단어는 리스트에 추가

                    mp_drawing.draw_landmarks(video, res, mp_hands.HAND_CONNECTIONS)
                    # 손가락 인식 시각화
            video = Image.fromarray(video)
            draw = ImageDraw.Draw(video)
            for i in _sum:
                if i in sentence:
                    pass
                else:
                    sentence += " "
                    sentence += i

            draw.text(xy=(20, 440), text=sentence, font=font, fill=(255, 255, 255))
            video = np.array(video)
            _, jpeg = cv2.imencode(".jpg", video)
            frame = jpeg.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
