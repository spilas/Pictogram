#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import math
import argparse
import threading
import pickle
import os
import time
import logging
import datetime
import pandas as pd
from keras.models import load_model

import cv2 as cv
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

from utils import CvFpsCalc

POSEON = False


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
    # テーマ切り替え
    ###########################################################################
    ch_time = 8

    # フォント
    ###########################################################################
    font_path = "font/hiragino-kaku-gothic-std-w8.otf"
    font_path_w6 = "font/ヒラギノ角ゴ ProN W6.otf"

    # ログ設定
    ###########################################################################
    log_file = "log/"+str(datetime.date.today())+".txt"
    if not log_file:
        f = open(log_file, 'w')
        f.write('')
        f.close()
    logging.basicConfig(filename = log_file,
                        level    = logging.DEBUG,
                        format   = " %(asctime)s - %(levelname)s - %(message)s "
    )

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

    CLmodel = load_model('model/mlpose.h5', compile=True)

    landmark_names = [
        'nose',
        'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear',
        'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky_1', 'right_pinky_1',
        'left_index_1', 'right_index_1',
        'left_thumb_2', 'right_thumb_2',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index',
    ]
    class_names = ["karate", "soccer", "volley", "fencing", "boxing", "run", "baseball", "lifting"]
    col_names = []
    for i in range(33):
        name = mp_pose.PoseLandmark(i).name
        name_x = name + '_X'
        name_y = name + '_Y'
        name_z = name + '_Z'
        name_v = name + '_V'
        col_names.append(name_x)
        col_names.append(name_y)
        col_names.append(name_z)
        col_names.append(name_v)

    # テーマ画像読み込み
    #############################################################
    theme_images = []
    no_theme = 0

    theme_folder_path = 'img/_pickle_poses'
    list_theme_path = os.listdir(theme_folder_path)
    list_theme_path.sort()

    list_theme_name = ["　　柔道　　","　サッカー　","バレーボール","フェンシング","ボクシング　","　短距離走　","　　野球　　","リフティング"]

    for filename in list_theme_path:
        file_path = os.path.join(theme_folder_path, filename)
    
        with open(file_path, 'rb') as file:
            image_pk = pickle.load(file)
            image_pk_edge = draw_white_edge(img=image_pk, 
                                    band_width= 5, 
                                    color= [0, 0, 0])
            image_pk_edge = cv.resize(image_pk_edge,None,fx=0.6,fy=0.6)
            theme_images.append(image_pk_edge)
    
    good_image = cv.imread("img/good.jpg")
    good_image = cv.resize(good_image,None,fx=0.45,fy=0.45)

    pose_image = cv.imread("img/poses/pose01.jpg")
    pose_image = cv.resize(pose_image,None,fx=0.6,fy=0.6)

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    tmp_time = time.time()

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
        image[0:image.shape[0],0:300] = (122,122,0)
        image[0:image.shape[0],image.shape[1]-300:image.shape[1]] = (122,122,0)
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

            if POSEON:
                predict_name = predict_pose(
                    results.pose_landmarks,
                    landmark_names,
                    class_names,
                    col_names,
                    model=CLmodel,
                    torso_size_multiplier=2.5,
                    threshold=0.75,
                )        

        main_palette = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        camera_palette = cv.resize(debug_image01,None,fx=0.30,fy=0.30)
        
        cv.putText(debug_image02, 
                   text="FPS:" + str(display_fps), 
                   org=(10,  image.shape[0]-30),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX, 
                   fontScale=0.5, 
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
            logging.debug("Press Q")
            no_theme = (no_theme + 1) if no_theme < 7 else 0
        elif key == 101: # e
            print("Press e, no_theme:",no_theme)
            logging.debug("Press E")
            no_theme = (no_theme - 1) if no_theme > -7 else 0
        elif key == 114: # r
            print("Press r")
            cv.imwrite('saves/'+str(int(time.time()))+'_'+str(no_theme)+'.jpg', cv.cvtColor(image, cv.COLOR_RGB2BGR))

        
        real_time = time.time()

        if real_time - tmp_time > ch_time:
            tmp_time = real_time
            no_theme = (no_theme + 1) if no_theme < 7 else 0
        
        camera_palette[0:camera_palette.shape[0],0:90] = (255,255,255)
        camera_palette[0:camera_palette.shape[0],camera_palette.shape[1]-90:camera_palette.shape[1]] = (255,255,255)

        # main画面合成 #########################################################
        main_palette[0:debug_image02.shape[0], 0:debug_image02.shape[1]] = debug_image02

        margin = 10

        # camera
        main_palette[image.shape[0] - camera_palette.shape[0] - margin:image.shape[0] - margin, image.shape[1] - camera_palette.shape[1] + 70:image.shape[1] - 20] = camera_palette[0:camera_palette.shape[0],0:camera_palette.shape[1]-90]
        # theme
        main_palette[110:theme_images[no_theme].shape[0] + 110, image.shape[1] - theme_images[no_theme].shape[1] - margin - 20:image.shape[1] - margin - 20] = theme_images[no_theme]

        
        main_palette = cv2_putText(main_palette,
                text = "「テーマ」",
                org = (image.shape[1] - theme_images[no_theme].shape[1] - margin,50),
                # fontFace = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
                fontFace = font_path_w6,
                fontScale = 30,
                color = (105,105,105))
        
        main_palette = cv2_putText(main_palette,
                text = "「テーマ」のポーズをしてみよう！",
                org = (image.shape[1]//2 - 280,90),
                # fontFace = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
                fontFace = font_path_w6,
                fontScale = 35,
                color = (0, 100, 0))
        
        lf_time = round(ch_time - (round(time.time(),1) - tmp_time),1)
        main_palette = cv2_putText(main_palette,
                text = str(lf_time),
                org = (image.shape[1]//2 - 50,image.shape[0]-100),
                # fontFace = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
                fontFace = font_path_w6,
                fontScale = 50,
                color = (105,105,105))

        
        main_palette = cv2_putText(main_palette,
                text = list_theme_name[no_theme],
                org = (image.shape[1] - theme_images[no_theme].shape[1] - 20,90),
                # fontFace = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
                fontFace = font_path_w6,
                fontScale = 30,
                color = (105,105,105))
        
        if POSEON:
            main_palette = cv2_putText(main_palette,
                    text = predict_name,
                    org = (50,50),
                    # fontFace = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
                    fontFace = font_path_w6,
                    fontScale = 30,
                    color = (105,105,105))
        
        len_seek_bar = int((time.time() - tmp_time)*image.shape[1] // ch_time)
        if len_seek_bar > image.shape[1]:
            len_seek_bar = image.shape[1]

        
        main_palette[image.shape[0]-8:image.shape[0],len_seek_bar:image.shape[1]] = (122,122,0)

        # main_palette[margin:pose_image.shape[0] + margin, image.shape[1] - pose_image.shape[1] - margin - 20:image.shape[1] - margin - 20] = pose_image

        if key == 32: # SPACE
            print("Press SPACE") 
            cv.imwrite("img/saved/"+str(int(time.time()))+".jpg",main_palette)

        # 画面反映 #############################################################
        # cv.imshow('Tokyo2020 Debug', debug_image01)
        #cv.imshow('Tokyo2020 Pictogram', debug_image02)
        cv.imshow('Pictogram', main_palette)

    cap.release()
    cv.destroyAllWindows()

def save_pose_image(debug_image02,prev_images,no_theme,per_resize):
    print("5 sec later") 
    save_image = cv.resize(debug_image02,None,fx=per_resize,fy=per_resize)
    save_image_edge = draw_white_edge(img=save_image, 
                                    band_width= 5, 
                                    color= [0, 0, 0])
    
    if len(prev_images[no_theme]) < 3:
        prev_images[no_theme].append(save_image_edge)
    else:
        prev_images[no_theme].pop(0)
        prev_images[no_theme].append(save_image_edge)
    


def draw_white_edge(img, band_width=10, color = [0, 0, 0]):
    # Image is aligned in BGR order.
    # color or Gray Scale
    if len(img.shape) < 3:  # gray scale
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    else:
        pass

    height, width, _ = img.shape[0], img.shape[1], img.shape[2]

    img_test = img.copy()

    img_test[height - band_width:height, :, :] = color
    img_test[0: band_width, :, :] = color
    img_test[:, 0:band_width, :] = color
    img_test[:, width - band_width:width, :] = color

    return img_test


# OpenCVで日本語フォントを描写する #############################################################
def pil2cv(imgPIL):
    imgCV_RGB = np.array(imgPIL, dtype = np.uint8)
    imgCV_BGR = np.array(imgPIL)[:, :, ::-1]
    return imgCV_BGR


def cv2pil(imgCV):
    imgCV_RGB = imgCV[:, :, ::-1]
    imgPIL = Image.fromarray(imgCV_RGB)
    return imgPIL


def cv2_putText(imgs, text, org, fontFace, fontScale, color):
    x, y = org
    b, g, r = color
    colorRGB = (r, g, b)
    imgPIL = cv2pil(imgs)
    draw = ImageDraw.Draw(imgPIL)
    fontPIL = ImageFont.truetype(font = fontFace, size = fontScale)
    text_box = draw.textbbox((0, 0), text, font=fontPIL)
    w, h = text_box[2] - text_box[0], text_box[3] - text_box[1]
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
        # print(landmark.z)
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

def predict_pose(
        pose_landmarks,
        landmark_names,
        class_names,
        col_names,
        model,
        torso_size_multiplier=2.5,
        threshold=0.75):
    lm_list = [landmarks for landmarks in pose_landmarks.landmark]
    
    center_x = (lm_list[landmark_names.index('right_hip')].x +
                lm_list[landmark_names.index('left_hip')].x)*0.5
    center_y = (lm_list[landmark_names.index('right_hip')].y +
                lm_list[landmark_names.index('left_hip')].y)*0.5

    shoulders_x = (lm_list[landmark_names.index('right_shoulder')].x +
                    lm_list[landmark_names.index('left_shoulder')].x)*0.5
    shoulders_y = (lm_list[landmark_names.index('right_shoulder')].y +
                    lm_list[landmark_names.index('left_shoulder')].y)*0.5
    
    max_distance = max(
        math.sqrt((lm.x - center_x)**2 + (lm.y - center_y)**2)
        for lm in lm_list
    )

    torso_size = math.sqrt((shoulders_x - center_x) ** 2 + (shoulders_y - center_y)**2)
    max_distance = max(torso_size*torso_size_multiplier, max_distance)

    pre_lm = list(np.array([[(landmark.x-center_x)/max_distance, 
                             (landmark.y-center_y)/max_distance,
                             landmark.z/max_distance, landmark.visibility] 
                             for landmark in lm_list]).flatten())

    data = pd.DataFrame([pre_lm], columns=col_names)
    predict = model.predict(data)[0]
    if max(predict) > threshold:
        pose_class = class_names[predict.argmax()]
    else:
        pose_class = 'Unknown Pose'

    return pose_class

if __name__ == '__main__':
    main()
