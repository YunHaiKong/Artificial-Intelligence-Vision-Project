import sys
import cv2
import numpy as np
import time

tic = time.time()  # 开始时间
numMatchedFrames = 0  # 记录匹配的帧数
MIN_COUNT = 10  # 阈值


def show_frame(frame):
    cv2.imshow('frame', frame)
    cv2.waitKey(1)


if __name__ == '__main__':
    try:
        video_src = sys.argv[1]
        logo_src = sys.argv[2]
    except:
        print("请输入视频文件路径和logo图片路径！")
        exit(1)

    # # 针对sift
    # detector = cv2.SIFT_create()
    # FLANN_INDEX_KDTREE = 1
    # flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(check=50)
    # matcher = cv2.FlannBasedMatcher(flann_params, search_params)

    # 针对ORB
    detector = cv2.ORB_create()
    FLANN_INDEX_KDTREE = 1
    FLANN_INDEX_LSH = 6
    flann_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number = 6,
                        key_size = 12,
                        multi_probe_level = 1)
    search_params = dict()
    matcher = cv2.FlannBasedMatcher(flann_params, search_params)
    ####

    logo = cv2.imread(logo_src, 1)
    # 用检测器检测图像的特征点（关键点）；并且计算特征点的描述符向量。
    keypoints_logo, desc_logo = detector.detectAndCompute(logo, None)  # Query Image

    min_x = logo.shape[1]
    min_y = logo.shape[0]
    max_x = 0
    max_y = 0
    for i in range(len(keypoints_logo)):
        min_x = np.min((min_x, keypoints_logo[i].pt[0]))  # 两个值要放到 tuple 里面
        min_y = np.min((min_y, keypoints_logo[i].pt[1]))
        max_x = np.max((max_x, keypoints_logo[i].pt[0]))
        max_y = np.max((max_y, keypoints_logo[i].pt[1]))
    logo_rect = (min_x, min_y, max_x, max_y)

    cap = cv2.VideoCapture(video_src)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, None, fx=0.35, fy=0.35)  # Modify1:将视频帧进行缩放

        frame_cur = frame.copy()
        vis = frame.copy()

        kps_frame, desc_frame = detector.detectAndCompute(frame_cur, None)  # Train image
        if len(kps_frame) <= 0:
            show_frame(vis)  # Modify2:在每个判断语句中都显示图片
            continue

        # 特征点匹配
        matches = matcher.knnMatch(desc_logo, desc_frame, k=2)
        good_matches_first = []  # 保存好的匹配（且是第一名，即只保留最好的那个匹配）
        matchesMask = [[0, 0] for i in range(len(matches))]  # 一个布尔数组，指定哪些匹配应该被绘制。数组中为 True（即1） 的匹配才会被绘制。
        for i, m in enumerate(matches):
            if len(m) < 2:
                continue
            if m[0].distance < 0.7 * m[1].distance:
                matchesMask[i] = [1, 0]   # 只有最佳的匹配才会被绘制
                good_matches_first.append(m[0])
        print('通过Lowe比率测试的数量：', len(good_matches_first))
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask, flags=0)
        img3 = cv2.drawMatchesKnn(logo, keypoints_logo, frame_cur, kps_frame, matches, None, **draw_params)
        cv2.imshow('img3', img3)

        if len(kps_frame) < MIN_COUNT:
            show_frame(vis)
            continue

        if len(good_matches_first) < MIN_COUNT:
            show_frame(vis)
            continue

        # 可视化这种匹配
        p0 = [keypoints_logo[m.queryIdx].pt for m in good_matches_first]  # 来自query图(可乐标志图）的点集0的坐标
        p1 = [kps_frame[n.trainIdx].pt for n in good_matches_first]  # 来自Train图（视频帧）的点集1的坐标
        p0, p1 = np.float32((p0, p1))

        # 通过计算单应性 来求得 从p0集 到 p1集的变换矩阵H
        H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)  # 找单应性（从p0到p1的变换矩阵）
        status_prime = status.ravel()  # 从二维拉直到一维（拉平）
        print(status_prime)
        good_points = status_prime != 0
        if good_points.sum() < MIN_COUNT:
            show_frame(vis)
            continue

        # 可视化
        x0, y0, x1, y1 = logo_rect
        quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])  # 顺时针组织可乐标志包围矩形的四点
        quad_prime = cv2.perspectiveTransform(quad.reshape(1, 4, 2), H).reshape(4, 2)  # 变换到视频帧
        cv2.polylines(vis, [np.int32(quad_prime)], True, (255, 255, 0), 2)
        show_frame(vis)

        numMatchedFrames += 1  # 匹配帧数增1

print("numMatchedFrames ", numMatchedFrames)
toc = time.time()
shijian = toc - tic
print("total time ", np.round(shijian, 1), "秒")

cap.release()
if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()
