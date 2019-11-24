# CNN特征的时候用这个
# from motion_detect.tracker.cnn import multi_kcftracker as mkcf
# NON-CNN特征的时候用这个
# from motion_detect.tracker import multi_kcftracker as mkcf
from motion_detect.tracker import multi_kcftracker as mkcf
from motion_detect.bma.mog2 import Mog2Detect as bma_algorithm


class MotionDetector:
    def __init__(self, reinit_interval=90):
        # 背景建模算法
        self._bgs_algorithm = bma_algorithm(100)
        self._tracker = mkcf.KCFMultiTracker(True, True)
        self._frame = None
        self._is_need_reinit = True
        self._current_frame_index = int(0)
        # 每90帧进行一次重新建模，防止漏掉新加入视频边界的目标
        self._reinit_interval = reinit_interval
        self._rois = []

    # 初始化/重新初始化
    def _reinitialize(self):
        boxes_temp = self._bgs_algorithm.get_object_boxes(self._frame)
        # 有至少一个目标，初始化完成，更新boxes，初始化才算完成
        if len(boxes_temp) > 0:
            self._rois = boxes_temp
            self._is_need_reinit = False
            self._tracker.init(self._rois, self._frame)

    def _update(self):
        success, boxes_temp = self._tracker.update(self._frame)
        self._rois = boxes_temp
        if not success:
            self._is_need_reinit = True

    def detect(self, frame):
        self._frame = frame
        # 记录当前帧的索引
        self._current_frame_index = self._current_frame_index + 1
        if self._is_need_reinit:
            self._reinitialize()
        else:
            self._bgs_algorithm.train_only(frame)
            self._update()
        # 超过最短重置时间，重新建模
        if self._current_frame_index % self._reinit_interval == 0:
            self._is_need_reinit = True
        return self._rois
