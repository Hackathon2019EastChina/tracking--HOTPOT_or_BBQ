import cv2


# 绘制一个框
def draw_rect(img, rect, rgb=(0, 255, 0)):
    x, y, w, h = rect
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    # 颜色在这里是BGR，要转换一下
    return cv2.rectangle(img, (x, y), (x + w, y + h), (rgb[2], rgb[1], rgb[0]), 2)


# 绘制一条线
def draw_line(img, p0=(), p1=(), rgb=(0, 255, 0)):
    return cv2.line(img, p0, p1, (rgb[2], rgb[1], rgb[0]), 2, 4)


# 绘制许多直线，line [[(p0x,p0y),(p1x,p1y)],[...]]
def draw_lines(img, lines=[], rgb=(0, 255, 0)):
    for line in lines:
        img = draw_line(img, line[0], line[1], rgb)
    return img


# 绘制多个框
def draw_rects(img, rects, rgb=(0, 255, 0)):
    for rect in rects:
        img = draw_rect(img, rect, rgb)
    return img


def draw_point(img, point, rgb=(241, 158, 194)):
    """
    绘制一个点
    :param img: cv2 image
    :param point: singal point type: list
    :param rgb: the color of the point ,default 死亡芭比粉
    :return: cv2 image
    """
    a, b = point
    a = int(a)
    b = int(b)
    point_size = 1
    point_color = rgb
    # positive value means the breadth of circle's broder 0,4,8 is available
    thickness = -1
    img = cv2.circle(img, (a, b), point_size, point_color, thickness)
    return img


def draw_points(img, points, rgb=(241, 158, 194)):
    """
        绘制多个点
        :param img: cv2 image
        :param point: points type: list
        :param rgb: the color of the point ,default 死亡芭比粉
        :return: cv2 image
        """
    for point in points:
        img = draw_point(img, point, rgb)
    return img
