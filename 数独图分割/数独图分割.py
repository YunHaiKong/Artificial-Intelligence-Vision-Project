import cv2
import operator
import numpy as np
import os
from datetime import datetime

def show_image(img, win='image'):
    """显示图片，直到按下任意键继续"""
    cv2.imshow(win, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_digits(digits, color=255, withBorder=True, grid_num=1):
    """将提取并处理过的81个单元格图片构成的列表显示为二维9*9大图"""
    rows = []
    if withBorder:
        with_border = [cv2.copyMakeBorder(digit, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, color) for digit in digits]
    for i in range(9):
        if withBorder:
            row = np.concatenate(with_border[i * 9: (i + 1) * 9], axis=1)
        else:
            row = np.concatenate(digits[i * 9: (i + 1) * 9], axis=1)
        rows.append(row)
    bigImage = np.concatenate(rows, axis=0)
    
    # 在大图上添加网格编号
    if grid_num > 1:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(bigImage, f"Grid {grid_num}", (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    show_image(bigImage, f'bigImage - Grid {grid_num}')
    
    # 生成唯一文件名并保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"segmentedBigImg_grid{grid_num}_{timestamp}.jpg"
    cv2.imwrite(filename, bigImage)
    print(f"已保存数独网格 {grid_num} 到: {filename}")

def convert_with_color(color, img):
    """如果color是元组且img是灰度图，则动态地转换img为彩图"""
    if len(color) == 3 and (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1)):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def pre_process_gray(gray, skip_dilate=False):
    """使用高斯模糊、自适应阈值分割和/或膨胀来暴露图像的主特征"""
    proc = cv2.GaussianBlur(gray.copy(), (9, 9), 0)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    if not skip_dilate:
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
        proc = cv2.dilate(proc, kernel)
    return proc

def display_points(in_img, points, radius=5, color=(0, 0, 255)):
    """在图像上绘制彩色圆点，原图像可能是灰度图"""
    img = in_img.copy()
    img = convert_with_color(color, img)
    for point in points:
        cv2.circle(img, tuple(int(x) for x in point), radius, color, -1)
    return img

def find_corners_of_largest_polygon(bin_img):
    """找出图像中面积最大轮廓的4个角点。"""
    contours, h = cv2.findContours(bin_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]

    bottom_right_idx, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left_idx, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left_idx, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right_idx, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    points = [polygon[top_left_idx][0], polygon[top_right_idx][0],
              polygon[bottom_right_idx][0], polygon[bottom_left_idx][0]]
    show_image(display_points(bin_img, points), '4-points')
    return points

def distance_between(p1, p2):
    """返回两点之间的标量距离"""
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))

def crop_and_warp(gray, crop_rect):
    """将灰度图像中由4角点围成的四边形区域裁剪出来，并将其扭曲为类似大小的正方形"""
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    side = max([
        distance_between(bottom_right, top_right),
        distance_between(top_left, bottom_left),
        distance_between(bottom_right, bottom_left),
        distance_between(top_left, top_right)
    ])

    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    m = cv2.getPerspectiveTransform(src, dst)
    cropped = cv2.warpPerspective(gray, m, (int(side), int(side)))
    
    # 生成唯一文件名并保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cropped_grid_{timestamp}.png"
    cv2.imwrite(filename, cropped)
    print(f"已保存裁剪图像到: {filename}")
    
    show_image(cropped, 'cropped')
    return cropped

def infer_grid(square_gray):
    """从正方形灰度图像推断其内部81个单元网格的位置（以等分方式）。"""
    squares = []
    side = square_gray.shape[:1][0] / 9
    for j in range(9):
        for i in range(9):
            p1 = (i * side, j * side)
            p2 = ((i + 1) * side, (j + 1) * side)
            squares.append((p1, p2))
    return squares

def cut_from_rect(img, rect):
    """从图像中切出一个矩形ROI区域。"""
    return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]

def scale_and_centre(img, size, margin=0, background=0):
    """把单元格图片img经缩放且加边距，置于边长为size的新背景正方形图像中"""
    h, w = img.shape[:2]

    def centre_pad(length):
        padAll = size - length
        if padAll % 2 == 0:
            pad1 = int(padAll / 2)
            pad2 = pad1
        else:
            pad1 = int(padAll / 2)
            pad2 = pad1 + 1
        return pad1, pad2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    if margin % 2 != 0:
        img = cv2.resize(img, (size, size))
    return img

def find_largest_feature(inp_img, scan_tl, scan_br):
    """利用floodFill函数返回它所填充区域的边界框的事实，找到图像中的主特征，将此结构填充为白色，其余部分降为黑色。"""
    img = inp_img.copy()
    h, w = img.shape[:2]
    max_area = 0
    seed_point = (None, None)

    for x in range(scan_tl[0], scan_br[0]):
        for y in range(scan_tl[1], scan_br[1]):
            if img.item(y, x) == 255 and x < w and y < h:
                area = cv2.floodFill(img, None, (x, y), 64)
                if area[0] > max_area:
                    max_area = area[0]
                    seed_point = (x, y)
    for x in range(w):
        for y in range(h):
            if img.item(y, x) == 255 and x < w and y < h:
                cv2.floodFill(img, None, (x, y), 64)

    if all([p is not None for p in seed_point]):
        cv2.floodFill(img, None, seed_point, 255)
    top, bottom, left, right = h, 0, w, 0
    for x in range(w):
        for y in range(h):
            if img.item(y, x) == 64:
                cv2.floodFill(img, None, (x, y), 0)
            if img.item(y, x) == 255:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right

    bbox = [[left, top], [right, bottom]]
    return img, np.array(bbox, dtype='float32'), seed_point

def extract_digit(bin_img, rect, size):
    """从预处理后的二值方形大格子图中提取由rect指定的小单元格数字图"""
    digit = cut_from_rect(bin_img, rect)
    h, w = digit.shape[:2]
    margin = int(np.mean([h, w]) / 2.5)
    flooded, bbox, seed = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])

    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]

    if w > 0 and h > 0 and (w * h) > 200:
        digit = cut_from_rect(flooded, bbox)
        return scale_and_centre(digit, size, 4)
    else:
        return np.zeros((size, size), np.uint8)

def get_digits(square_gray, squares, size):
    """提取小单元格数字，组织成数组形式"""
    digits = []
    square_bin = pre_process_gray(square_gray.copy(), skip_dilate=True)

    color = convert_with_color((0, 0, 255), square_bin)
    h, w = color.shape[:2]
    for i in range(10):
        cv2.line(color, (0, int(i * h / 9)), (w - 1, int(i * h / 9)), (0, 0, 255))
        cv2.line(color, (int(i * w / 9), 0), (int(i * w / 9), h - 1), (0, 0, 255))
    show_image(color, 'drawRedLine')

    for square in squares:
        digits.append(extract_digit(square_bin, square, size))
    return digits

def find_sudoku_grids(image_path):
    """定位图像中的所有数独图"""
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    processed = pre_process_gray(original)

    # 查找轮廓
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 筛选可能是数独图的轮廓（假设数独图是较大的矩形）
    sudoku_grids = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:  # 矩形
            sudoku_grids.append(approx)

    return sudoku_grids

def parse_multiple_grids(image_path):
    """处理包含多个数独图的图像"""
    sudoku_grids = find_sudoku_grids(image_path)
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    for i, grid in enumerate(sudoku_grids):
        print(f"Processing grid {i + 1}")
        corners = grid.reshape(4, 2)  # 四个角点
        # 确保角点顺序正确（左上、右上、右下、左下）
        corners = order_points(corners)
        cropped = crop_and_warp(original, corners)
        squares = infer_grid(cropped)
        digits = get_digits(cropped, squares, 58)
        show_digits(digits, withBorder=True, grid_num=i+1)

def order_points(pts):
    """将四个角点按左上、右上、右下、左下顺序排列"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    return rect

if __name__ == '__main__':
    image_path = 'sudoku2.png'
    parse_multiple_grids(image_path)