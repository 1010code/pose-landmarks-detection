import cv2
import mediapipe as mp

#####################
import numpy as np
def angle_btn_3points(p1,p2,p3):
    p1 = np.array(p1) 
    p2 = np.array(p2) 
    p3 = np.array(p3)
    radians = np.arctan2(p3[1]-p2[1], p3[0]-p2[0]) - np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle >180.0:
        angle = 360-angle
    return angle
####################


mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./video/upstair.mp4')

# 取得原始影片的 FPS
original_fps = cap.get(cv2.CAP_PROP_FPS)
print(f'fps: {original_fps}')
# 設定 VideoWriter
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('walk1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), original_fps, (frame_width, frame_height))

# 啟用姿勢偵測
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, img = cap.read()
        if not ret:
            print("End of stream")
            break
        img = cv2.resize(img, (frame_width, frame_height))              # 縮小尺寸，加快演算速度
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        results = pose.process(img2)                  # 取得姿勢偵測結果
        # 根據姿勢偵測結果，標記身體節點和骨架
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        ##################
        landmarks = results.pose_landmarks.landmark
        kp = mp_pose.PoseLandmark
        # Get coordinates and angle (Left Knee angle)
        p1 = [landmarks[kp.LEFT_HIP.value].x,landmarks[kp.LEFT_HIP.value].y]
        p2 = [landmarks[kp.LEFT_KNEE.value].x,landmarks[kp.LEFT_KNEE.value].y]
        p3 = [landmarks[kp.LEFT_ANKLE.value].x,landmarks[kp.LEFT_ANKLE.value].y]
        DL = angle_btn_3points(p1, p2, p3)  
        # Get coordinates and angle (Right Knee angle)
        p1 = [landmarks[kp.RIGHT_HIP.value].x,landmarks[kp.RIGHT_HIP.value].y]
        p2 = [landmarks[kp.RIGHT_KNEE.value].x,landmarks[kp.RIGHT_KNEE.value].y]
        p3 = [landmarks[kp.RIGHT_ANKLE.value].x,landmarks[kp.RIGHT_ANKLE.value].y]
        DR = angle_btn_3points(p1, p2, p3)
        cv2.putText(img, f'{int(DL)}', (int(landmarks[kp.LEFT_KNEE.value].x*frame_width),int(landmarks[kp.LEFT_KNEE.value].y*frame_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        cv2.putText(img, f'{int(DR)}', (int(landmarks[kp.RIGHT_KNEE.value].x*frame_width),int(landmarks[kp.RIGHT_KNEE.value].y*frame_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        ##################
        out.write(img)  # 將畫面寫入 VideoWriter
        cv2.imshow('Preview', img)
        if cv2.waitKey(5) == ord('q'):
            break     # 按下 q 鍵停止

# 釋放 VideoCapture 和 VideoWriter
cap.release()
out.release()
cv2.destroyAllWindows()