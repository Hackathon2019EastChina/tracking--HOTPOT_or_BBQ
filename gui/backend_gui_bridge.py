import cv2
import gui.util.cv_rectutil as cvrect
import gui.extra_function.area_monitor as amonitor
import gui.extra_function.structs as structs
from motion_detect import motion_detector as motdet

white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
black = (0, 0, 0)


class BackendGuiBridge:
    def __init__(self, video_path, lines=[], areas=[], draw_trace=True):
        self._video_cap = cv2.VideoCapture(video_path)
        self._area_monitor = amonitor.AreaMonitor()
        self._draw_trace = draw_trace
        self._current_frame_index = 0
        self._motion_detector = motdet.MotionDetector()
        self._log_list = []
        self._rect_touched_area = 0
        self._rect_touched_line = 0
        self._all_frame_rect_centers = []
        for l in lines:
            sline = structs.Line()
            sline.from_cv_line(l)
            self._area_monitor.add_line(sline)
        for a in areas:
            sarea = structs.Rectangle()
            sarea.from_cv_rect(a)
            self._area_monitor.add_rect(sarea)

    def __del__(self):
        self._video_cap.release()

    # 添加绊线
    def add_line(self, line: structs.Line):
        self._area_monitor.add_line(line)

    # 添加禁区
    def add_area(self, area: structs.Rectangle):
        sarea = structs.Rectangle()
        sarea.from_cv_rect(area)
        self._area_monitor.add_rect(sarea)

    # 打印一个绊线触碰日志
    def _print_line_log(self, rect: structs.Rectangle, line: structs.Line):
        r = rect.to_cv_rect()
        p0 = list(line.a_point.to_cv_point())
        p1 = list(line.b_point.to_cv_point())
        self._log_list.append(
            '绊线警告：目标“({0},{1},{2},{3})”触碰了绊线 “[({4},{5}),({6},{7})]”'.format(int(r[0]), int(r[1]), int(r[2]), int(r[3]),
                                                                             int(p0[0]), int(p0[1]), int(p1[0]),
                                                                             int(p1[1])))

    # 打印一个禁区侵入日志
    def _print_area_log(self, rect: structs.Rectangle, area: structs.Rectangle):
        r = rect.to_cv_rect()
        a = area.to_cv_rect()
        self._log_list.append(
            '禁区警告：目标“({0},{1},{2},{3})”进入了禁区“({4},{5},{6},{7})”'.format(int(r[0]), int(r[1]), int(r[2]), int(r[3]),
                                                                        int(a[0]), int(a[1]), int(a[2]), int(a[3])))

    # 画出禁区
    def _draw_area(self, frame):
        for line in self._area_monitor.get_lines():
            cvrect.draw_line(frame, line.a_point.to_cv_point(), line.b_point.to_cv_point(), blue)
        for area in self._area_monitor.get_rects():
            cvrect.draw_rect(frame, area.to_cv_rect(), blue)
        return frame

    # 绘制跟踪框，并且根据是否在禁区决定框的颜色
    def _draw_tracking_rects(self, frame, rect_problems=[]):
        # 清除上一次的log
        self._log_list.clear()
        # 检查出错的地方
        for rect_problem in rect_problems:
            who = rect_problem.which_rect
            # 碰线警告
            for line in rect_problem.line_touched:
                self._print_line_log(who, line)
            # 区域入侵警告
            for area in rect_problem.rect_trespassed:
                self._print_area_log(who, area)
            # 有触碰绊线或者区域则显示为红色框，否则为绿色
            if len(rect_problem.rect_trespassed) > 0 or len(rect_problem.line_touched) > 0:
                cvrect.draw_rect(frame, who.to_cv_rect(), red)
            else:
                cvrect.draw_rect(frame, who.to_cv_rect(), green)
        return frame

    # 获取一帧的图像（已经标记好框的）
    def get_frame(self):
        success, frame = self._video_cap.read()
        if success:
            self._current_frame_index = self._current_frame_index + 1
            rects = self._motion_detector.detect(frame)
            # 添加矩阵中心到点集中
            if self._draw_trace:
                if self._get_rect_centers(rects):
                    self._all_frame_rect_centers.append(self._get_rect_centers(rects))
                    self._draw_trace_draw(frame)
            rect_problems = self._area_monitor.check_problems(rects)
            frame = self._draw_area(frame)
            frame = self._draw_tracking_rects(frame, rect_problems)
            return success, frame
        else:
            return success, None

    # 获取视频检查信息
    def get_logs(self):
        return self._log_list

    # 内部函数，获取矩形的中心点集
    def _get_rect_centers(self, rects=[]):
        current_frame_center = []
        for x, y, w, h in rects:
            current_frame_center = [x + w / 2, y + h / 2]
            # error tag
            # current_frame_center.append(center)
        return current_frame_center

    # 在这里实现画轨迹功能，frame是当前帧的opencv图像（numpy array），返回一个opencv图像（numpy array）
    # 此函数内只允许读取self._all_frame_rect_centers，否则不和规
    # 此函数每有一帧都会调用一次
    def _draw_trace_draw(self, frame):
        cvrect.draw_points(frame, self._all_frame_rect_centers)
        # although what's the necessity of return the frame, I do it for the last one modified the code
        return frame
