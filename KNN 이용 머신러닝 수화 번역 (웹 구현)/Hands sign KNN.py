import cv2
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import time
import threading

sum = []  # 텍스트 전체 저장 리스트

font = ImageFont.truetype("./SCDream9.otf", 25)  # 텍스트 표출시 폰트 설정

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 리스트 생성
gesturelist = {
    0: "좋아",
    1: "무엇을 해도",
    2: "예뻐",
    3: "너가",
    4: "나는",
    5: "빈칸2",
    6: "지우기",
    7: "오늘",
    8: "고마워",
    9: "항상",
    10: "보다",
    10: "먹다",
    11: "자다",
    12: "반가워",
    13: "커피",
    14: "학교",
    15: "대회",
    16: "내일",
    17: "노래",
    18: "숙제",
    19: "안경",
    20: "책",
}

# 데이터 변수 설정
file = np.genfromtxt(
    "./worddata.csv",
    delimiter=",",
)
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)

# K-NN 알고리즘 객체 생성
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

# 영상 비디오 변수 생성
cap = cv2.VideoCapture(0)
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
                    [
                        0,
                        1,
                        2,
                        3,
                        0,
                        5,
                        6,
                        7,
                        0,
                        9,
                        10,
                        11,
                        0,
                        13,
                        14,
                        15,
                        0,
                        17,
                        18,
                        19,
                    ],
                    :,
                ]
                v2 = joint[
                    [
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        16,
                        17,
                        18,
                        19,
                        20,
                    ],
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
                            sum.clear()
                        else:
                            sum.append(gesturelist[index])
                            # 인식 성공한 단어는 리스트에 추가

                mp_drawing.draw_landmarks(video, res, mp_hands.HAND_CONNECTIONS)
                # 손가락 인식 시각화
        video = Image.fromarray(video)
        draw = ImageDraw.Draw(video)
        for i in sum:
            if i in sentence:
                pass
            else:
                sentence += " "
                sentence += i

        draw.text(xy=(20, 440), text=sentence, font=font, fill=(255, 255, 255))
        # 영상에 한글 텍스트 입력

        video = np.array(video)

        _, jpeg = cv2.imencode(".jpg", video)

        cv2.imshow("image", video)
        if cv2.waitKey(1) == ord("q"):
            break
f.close()
