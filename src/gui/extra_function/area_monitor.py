import gui.extra_function.structs as models
import gui.extra_function.intersection_check as icheck


class AreaMonitor:
    def __init__(self):
        self._lines = []
        self._rects = []

    def add_line(self, new_line: models.Line):
        self._lines.append(new_line)

    def get_lines(self):
        return self._lines

    def add_rect(self, new_rect: models.Rectangle):
        self._rects.append(new_rect)

    def get_rects(self):
        return self._rects

    def clear_monitoring_area(self):
        self._lines = []
        self._rects = []

    def check_problems(self, tracking_rects=[]):
        result = []
        # 一开始都没问题
        for tracking_rect in tracking_rects:
            rect_problem = models.RectProblem()
            rect_problem.which_rect.from_cv_rect(tracking_rect)
            result.append(rect_problem)
        # 问题检测
        for i in range(len(tracking_rects)):
            # 绊线检查
            for line in self._lines:
                if icheck.is_line_rect_intersection(line=line, rect=result[i].which_rect):
                    result[i].line_touched.append(line)
            # 区域入侵检查，使用矩阵的中心
            for area in self._rects:
                center = result[i].which_rect.center()
                if icheck.is_point_in_rect(center, area):
                    result[i].rect_trespassed.append(area)
        return result
