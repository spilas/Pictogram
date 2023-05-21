#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import math
import argparse
import threading
import pickle
import os

import cv2 as cv
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

from utils import CvFpsCalc


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=1280)
    parser.add_argument("--height", help='cap height', type=int, default=720)

    parser.add_argument('--static_image_mode', action='store_true')
    parser.add_argument("--model_complexity",
                        help='model_complexity(0,1(default),2)',
                        type=int,
                        default=1)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    parser.add_argument('--rev_color', action='store_true')

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    static_image_mode = args.static_image_mode
    model_complexity = args.model_complexity
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    rev_color = args.rev_color

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv.CAP_PROP_FPS, 30)

    # モデルロード #############################################################
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # テーマ画像読み込み
    #############################################################
    theme_images = []
    no_theme = 0
    theme_folder_path = 'F:\App\Pictogram\img\_pickle'

    for filename in os.listdir(theme_folder_path):
        file_path = os.path.join(theme_folder_path, filename)
    
        with open(file_path, 'rb') as file:
            image_pk = pickle.load(file)
            theme_images.append(image_pk)
    
    # prev画像読み込み
    #############################################################
    prev_default_path = r'F:\App\Pictogram\img\photo_no_image.jpg.pickle'
    with open(prev_default_path, 'rb') as file:
        prev_default = pickle.load(file)
    prev_default = cv.resize(prev_default,None,fx=0.20,fy=0.20)

    prev_images = [[] * 3 for _ in range(len(theme_images))]
    

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # 色指定
    if rev_color:
        color = (255, 255, 255)
        bg_color = (100, 33, 3)
    else:
        color = (100, 33, 3)
        bg_color = (255, 255, 255)

    while True:
        display_fps = cvFpsCalc.get()

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # ミラー表示
        debug_image01 = copy.deepcopy(image)
        debug_image02 = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        cv.rectangle(debug_image02, 
                     (0, 0), 
                     (image.shape[1], image.shape[0]), 
                     bg_color, 
                     thickness=-1)

        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = pose.process(image)

        # 描画 ################################################################
        if results.pose_landmarks is not None:
            # 描画
            debug_image01 = draw_landmarks(
                debug_image01,
                results.pose_landmarks,
            )
            debug_image02 = draw_stick_figure(
                debug_image02,
                results.pose_landmarks,
                color=color,
                bg_color=bg_color,
            )

        main_palette = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        camera_palette = cv.resize(debug_image01,None,fx=0.20,fy=0.20)
        
        cv.putText(debug_image02, 
                   text="FPS:" + str(display_fps), 
                   org=(10,  image.shape[0]-30),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX, 
                   fontScale=1.0, 
                   color=color, 
                   thickness=2, 
                   lineType=cv.LINE_AA)
        

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            print("Press ESC")
            break
        elif key == 113: # q
            print("Press q, no_theme:",no_theme)
            no_theme = (no_theme + 1) if no_theme < 49 else 0
        elif key == 101: # e
            print("Press e, no_theme:",no_theme)
            no_theme = (no_theme - 1) if no_theme > -49 else 0
               
        elif key == 13:  # ENTER
            print("Press ENTER")            
            #t=threading.Timer(5,print("wait 5 sec"))
            #t.start()
            save_image = cv.resize(debug_image02,None,fx=0.20,fy=0.20)
            
            if len(prev_images[no_theme]) < 3:
                prev_images[no_theme].append(save_image)
            else:
                prev_images[no_theme].pop(0)
                prev_images[no_theme].append(save_image)
            

        '''
            prev_image = cv2_putText(img=prev_image,
                        text = "前の人",
                        org = (300,50),
                        fontFace = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
                        fontScale = 30,
                        color = (0,255,0))
        '''
    

        # main画面合成 #########################################################
        main_palette[0:debug_image02.shape[0], 0:debug_image02.shape[1]] = debug_image02

        margin = 10

        # camera
        main_palette[image.shape[0] - camera_palette.shape[0] - margin:image.shape[0] - margin, image.shape[1] - camera_palette.shape[1] - margin:image.shape[1] - margin] = camera_palette
        # theme
        main_palette[margin:theme_images[no_theme].shape[0] + margin, image.shape[1] - theme_images[no_theme].shape[1] - margin:image.shape[1] - margin] = theme_images[no_theme]
        # prev image
        default_shape = prev_default.shape
        
        for i in range(3):
            if len(prev_images[no_theme]) > i:
                image_shape = prev_images[no_theme][i].shape
                main_palette[(image_shape[0] + margin) * i + margin: 
                             (image_shape[0] + margin) * (i + 1), margin: 
                             image_shape[1] + margin] = prev_images[no_theme][i]
            else:
                main_palette[(default_shape[0] + margin) * i + margin: (default_shape[0] + margin) * (i + 1), margin: default_shape[1] + margin] = prev_default
        

        # 画面反映 #############################################################
        # cv.imshow('Tokyo2020 Debug', debug_image01)
        #cv.imshow('Tokyo2020 Pictogram', debug_image02)
        cv.imshow('Pictogram', main_palette)

    cap.release()
    cv.destroyAllWindows()



# OpenCVで日本語フォントを描写する #############################################################
def pil2cv(imgPIL):
    imgCV_RGB = np.array(imgPIL, dtype = np.uint8)
    imgCV_BGR = np.array(imgPIL)[:, :, ::-1]
    return imgCV_BGR

def cv2pil(imgCV):
    imgCV_RGB = imgCV[:, :, ::-1]
    imgPIL = Image.fromarray(imgCV_RGB)
    return imgPIL

def cv2_putText(img, text, org, fontFace, fontScale, color):
    x, y = org
    b, g, r = color
    colorRGB = (r, g, b)
    imgPIL = cv2pil(img)
    draw = ImageDraw.Draw(imgPIL)
    fontPIL = ImageFont.truetype(font = fontFace, size = fontScale)
    w, h = draw.textsize(text, font = fontPIL)
    draw.text(xy = (x,y-h), text = text, fill = colorRGB, font = fontPIL)
    imgCV = pil2cv(imgPIL)
    return imgCV



def pickle_load(path):
    with open(path, mode='rb') as f:
        return pickle.load(f)



def draw_stick_figure(
        image,
        landmarks,
        color=(100, 33, 3),
        bg_color=(255, 255, 255),
        visibility_th=0.5,
):
    image_width, image_height = image.shape[1], image.shape[0]

    # 各ランドマーク算出
    landmark_point = []
    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append(
            [index, landmark.visibility, (landmark_x, landmark_y), landmark_z])

    # 脚の付け根の位置を腰の中点に修正
    right_leg = landmark_point[23]
    left_leg = landmark_point[24]
    leg_x = int((right_leg[2][0] + left_leg[2][0]) / 2)
    leg_y = int((right_leg[2][1] + left_leg[2][1]) / 2)

    landmark_point[23][2] = (leg_x, leg_y)
    landmark_point[24][2] = (leg_x, leg_y)

    # 距離順にソート
    sorted_landmark_point = sorted(landmark_point,
                                   reverse=True,
                                   key=lambda x: x[3])

    # 各サイズ算出
    (face_x, face_y), face_radius = min_enclosing_face_circle(landmark_point)

    face_x = int(face_x)
    face_y = int(face_y)
    face_radius = int(face_radius * 1.5)

    stick_radius01 = int(face_radius * (4 / 5))
    stick_radius02 = int(stick_radius01 * (3 / 4))
    stick_radius03 = int(stick_radius02 * (3 / 4))

    # 描画対象リスト
    draw_list = [
        11,  # 右腕
        12,  # 左腕
        23,  # 右脚
        24,  # 左脚
    ]

    # 背景色
    cv.rectangle(image, (0, 0), (image_width, image_height),
                 bg_color,
                 thickness=-1)

    # 顔 描画
    cv.circle(image, (face_x, face_y), face_radius, color, -1)

    # 腕/脚 描画
    for landmark_info in sorted_landmark_point:
        index = landmark_info[0]

        if index in draw_list:
            point01 = [p for p in landmark_point if p[0] == index][0]
            point02 = [p for p in landmark_point if p[0] == (index + 2)][0]
            point03 = [p for p in landmark_point if p[0] == (index + 4)][0]

            if point01[1] > visibility_th and point02[1] > visibility_th:
                image = draw_stick(
                    image,
                    point01[2],
                    stick_radius01,
                    point02[2],
                    stick_radius02,
                    color=color,
                    bg_color=bg_color,
                )
            if point02[1] > visibility_th and point03[1] > visibility_th:
                image = draw_stick(
                    image,
                    point02[2],
                    stick_radius02,
                    point03[2],
                    stick_radius03,
                    color=color,
                    bg_color=bg_color,
                )

    return image


def min_enclosing_face_circle(landmark_point):
    landmark_array = np.empty((0, 2), int)

    index_list = [1, 4, 7, 8, 9, 10]
    for index in index_list:
        np_landmark_point = [
            np.array(
                (landmark_point[index][2][0], landmark_point[index][2][1]))
        ]
        landmark_array = np.append(landmark_array, np_landmark_point, axis=0)

    center, radius = cv.minEnclosingCircle(points=landmark_array)

    return center, radius


def draw_stick(
        image,
        point01,
        point01_radius,
        point02,
        point02_radius,
        color=(100, 33, 3),
        bg_color=(255, 255, 255),
):
    cv.circle(image, point01, point01_radius, color, -1)
    cv.circle(image, point02, point02_radius, color, -1)

    draw_list = []
    for index in range(2):
        rad = math.atan2(point02[1] - point01[1], point02[0] - point01[0])

        rad = rad + (math.pi / 2) + (math.pi * index)
        point_x = int(point01_radius * math.cos(rad)) + point01[0]
        point_y = int(point01_radius * math.sin(rad)) + point01[1]

        draw_list.append([point_x, point_y])

        point_x = int(point02_radius * math.cos(rad)) + point02[0]
        point_y = int(point02_radius * math.sin(rad)) + point02[1]

        draw_list.append([point_x, point_y])

    points = np.array((draw_list[0], draw_list[1], draw_list[3], draw_list[2]))
    cv.fillConvexPoly(image, points=points, color=color)

    return image


def draw_landmarks(
    image,
    landmarks,
    # upper_body_only,
    visibility_th=0.5,
):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])

        if landmark.visibility < visibility_th:
            continue

        if index == 0:  # 鼻
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 1:  # 右目：目頭
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2:  # 右目：瞳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 3:  # 右目：目尻
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 4:  # 左目：目頭
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 5:  # 左目：瞳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:  # 左目：目尻
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 7:  # 右耳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 8:  # 左耳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 9:  # 口：左端
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 10:  # 口：左端
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 11:  # 右肩
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 12:  # 左肩
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 13:  # 右肘
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:  # 左肘
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 15:  # 右手首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 16:  # 左手首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 17:  # 右手1(外側端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 18:  # 左手1(外側端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 19:  # 右手2(先端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 20:  # 左手2(先端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 21:  # 右手3(内側端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 22:  # 左手3(内側端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 23:  # 腰(右側)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 24:  # 腰(左側)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 25:  # 右ひざ
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 26:  # 左ひざ
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 27:  # 右足首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 28:  # 左足首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 29:  # 右かかと
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 30:  # 左かかと
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 31:  # 右つま先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 32:  # 左つま先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)

        # if not upper_body_only:
        if True:
            cv.putText(image, "z:" + str(round(landmark_z, 3)),
                       (landmark_x - 10, landmark_y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                       cv.LINE_AA)

    # 右目
    if landmark_point[1][0] > visibility_th and landmark_point[2][
            0] > visibility_th:
        cv.line(image, landmark_point[1][1], landmark_point[2][1],
                (0, 255, 0), 2)
    if landmark_point[2][0] > visibility_th and landmark_point[3][
            0] > visibility_th:
        cv.line(image, landmark_point[2][1], landmark_point[3][1],
                (0, 255, 0), 2)

    # 左目
    if landmark_point[4][0] > visibility_th and landmark_point[5][
            0] > visibility_th:
        cv.line(image, landmark_point[4][1], landmark_point[5][1],
                (0, 255, 0), 2)
    if landmark_point[5][0] > visibility_th and landmark_point[6][
            0] > visibility_th:
        cv.line(image, landmark_point[5][1], landmark_point[6][1],
                (0, 255, 0), 2)

    # 口
    if landmark_point[9][0] > visibility_th and landmark_point[10][
            0] > visibility_th:
        cv.line(image, landmark_point[9][1], landmark_point[10][1],
                (0, 255, 0), 2)

    # 肩
    if landmark_point[11][0] > visibility_th and landmark_point[12][
            0] > visibility_th:
        cv.line(image, landmark_point[11][1], landmark_point[12][1],
                (0, 255, 0), 2)

    # 右腕
    if landmark_point[11][0] > visibility_th and landmark_point[13][
            0] > visibility_th:
        cv.line(image, landmark_point[11][1], landmark_point[13][1],
                (0, 255, 0), 2)
    if landmark_point[13][0] > visibility_th and landmark_point[15][
            0] > visibility_th:
        cv.line(image, landmark_point[13][1], landmark_point[15][1],
                (0, 255, 0), 2)

    # 左腕
    if landmark_point[12][0] > visibility_th and landmark_point[14][
            0] > visibility_th:
        cv.line(image, landmark_point[12][1], landmark_point[14][1],
                (0, 255, 0), 2)
    if landmark_point[14][0] > visibility_th and landmark_point[16][
            0] > visibility_th:
        cv.line(image, landmark_point[14][1], landmark_point[16][1],
                (0, 255, 0), 2)

    # 右手
    if landmark_point[15][0] > visibility_th and landmark_point[17][
            0] > visibility_th:
        cv.line(image, landmark_point[15][1], landmark_point[17][1],
                (0, 255, 0), 2)
    if landmark_point[17][0] > visibility_th and landmark_point[19][
            0] > visibility_th:
        cv.line(image, landmark_point[17][1], landmark_point[19][1],
                (0, 255, 0), 2)
    if landmark_point[19][0] > visibility_th and landmark_point[21][
            0] > visibility_th:
        cv.line(image, landmark_point[19][1], landmark_point[21][1],
                (0, 255, 0), 2)
    if landmark_point[21][0] > visibility_th and landmark_point[15][
            0] > visibility_th:
        cv.line(image, landmark_point[21][1], landmark_point[15][1],
                (0, 255, 0), 2)

    # 左手
    if landmark_point[16][0] > visibility_th and landmark_point[18][
            0] > visibility_th:
        cv.line(image, landmark_point[16][1], landmark_point[18][1],
                (0, 255, 0), 2)
    if landmark_point[18][0] > visibility_th and landmark_point[20][
            0] > visibility_th:
        cv.line(image, landmark_point[18][1], landmark_point[20][1],
                (0, 255, 0), 2)
    if landmark_point[20][0] > visibility_th and landmark_point[22][
            0] > visibility_th:
        cv.line(image, landmark_point[20][1], landmark_point[22][1],
                (0, 255, 0), 2)
    if landmark_point[22][0] > visibility_th and landmark_point[16][
            0] > visibility_th:
        cv.line(image, landmark_point[22][1], landmark_point[16][1],
                (0, 255, 0), 2)

    # 胴体
    if landmark_point[11][0] > visibility_th and landmark_point[23][
            0] > visibility_th:
        cv.line(image, landmark_point[11][1], landmark_point[23][1],
                (0, 255, 0), 2)
    if landmark_point[12][0] > visibility_th and landmark_point[24][
            0] > visibility_th:
        cv.line(image, landmark_point[12][1], landmark_point[24][1],
                (0, 255, 0), 2)
    if landmark_point[23][0] > visibility_th and landmark_point[24][
            0] > visibility_th:
        cv.line(image, landmark_point[23][1], landmark_point[24][1],
                (0, 255, 0), 2)

    if len(landmark_point) > 25:
        # 右足
        if landmark_point[23][0] > visibility_th and landmark_point[25][
                0] > visibility_th:
            cv.line(image, landmark_point[23][1], landmark_point[25][1],
                    (0, 255, 0), 2)
        if landmark_point[25][0] > visibility_th and landmark_point[27][
                0] > visibility_th:
            cv.line(image, landmark_point[25][1], landmark_point[27][1],
                    (0, 255, 0), 2)
        if landmark_point[27][0] > visibility_th and landmark_point[29][
                0] > visibility_th:
            cv.line(image, landmark_point[27][1], landmark_point[29][1],
                    (0, 255, 0), 2)
        if landmark_point[29][0] > visibility_th and landmark_point[31][
                0] > visibility_th:
            cv.line(image, landmark_point[29][1], landmark_point[31][1],
                    (0, 255, 0), 2)

        # 左足
        if landmark_point[24][0] > visibility_th and landmark_point[26][
                0] > visibility_th:
            cv.line(image, landmark_point[24][1], landmark_point[26][1],
                    (0, 255, 0), 2)
        if landmark_point[26][0] > visibility_th and landmark_point[28][
                0] > visibility_th:
            cv.line(image, landmark_point[26][1], landmark_point[28][1],
                    (0, 255, 0), 2)
        if landmark_point[28][0] > visibility_th and landmark_point[30][
                0] > visibility_th:
            cv.line(image, landmark_point[28][1], landmark_point[30][1],
                    (0, 255, 0), 2)
        if landmark_point[30][0] > visibility_th and landmark_point[32][
                0] > visibility_th:
            cv.line(image, landmark_point[30][1], landmark_point[32][1],
                    (0, 255, 0), 2)
    return image

if __name__ == '__main__':
    main()
