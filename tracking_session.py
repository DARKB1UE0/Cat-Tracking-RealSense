"""Mode/reference lifecycle. Invalidates slow model loads and old detections."""
import threading


class TrackingSession:
    def __init__(self, markers, follow, factory):
        self.markers, self.follow, self.factory = markers, follow, factory
        self.lock = threading.RLock()
        self.mode = 'cat'
        self.generation = 0
        self.epoch = 0
        self.tracker = None
        self.loading = False
        self.reference_paths = set()
        self.error = ''

    def snapshot(self):
        with self.lock:
            return dict(mode=self.mode, generation=self.generation, tracking_active=self.tracker is not None,
                        tracking_loading=self.loading, tracking_error=self.error,
                        has_tracker=self.tracker is not None)

    def _check(self, mode, generation):
        if mode != self.mode or type(generation) is not int or generation != self.generation:
            raise ValueError('识别模式已变化，请重新上传参考照片')

    def set_mode(self, mode):
        if not isinstance(mode, str) or mode not in ('cat', 'person'):
            raise ValueError('mode 必须为 cat 或 person')
        with self.lock:
            if self.mode != mode:
                self.epoch += 1; self.generation += 1
                self.tracker = None; self.loading = False; self.error = ''
                self.reference_paths.clear()
                self.mode = mode
                self.markers.set_target_kind(mode)
                self.follow.stop('识别模式已切换，自动追踪结束')
            return self.snapshot()

    def register_reference(self, path, mode, generation):
        with self.lock:
            self._check(mode, generation)
            if self.tracker or self.loading:
                raise ValueError('请先停止识别，再上传参考照片')
            self.reference_paths.add(path)

    def start(self, path, mode, generation):
        with self.lock:
            self._check(mode, generation)
            if not isinstance(path, str) or path not in self.reference_paths:
                raise ValueError('请上传当前模式的参考照片')
            if self.tracker or self.loading:
                raise ValueError('识别正在运行或初始化中')
            self.epoch += 1
            ticket = self.epoch
            self.loading = True; self.error = ''
            self.markers.clear('等待目标识别')
            self.follow.stop('切换识别目标，自动追踪结束')
        try:
            tracker = self.factory(path, mode)
        except Exception as exc:
            with self.lock:
                if ticket == self.epoch:
                    self.loading = False; self.error = str(exc)
            raise ValueError(str(exc)) from exc
        with self.lock:
            if ticket != self.epoch:
                raise ValueError('初始化期间识别已停止或模式已切换，请重新启动')
            self.tracker = tracker; self.loading = False

    def stop(self):
        with self.lock:
            self.epoch += 1
            self.tracker = None; self.loading = False; self.error = ''
            self.markers.clear('识别已停止')
            self.follow.stop('识别已停止，自动追踪结束')

    def capture(self, feedback):
        with self.lock:
            return (self.tracker, self.epoch, self.mode,
                    self.markers.capture(feedback) if self.tracker else None)

    def observe(self, sample, point, reason='未确认目标，暂不标注'):
        tracker, epoch, _, capture = sample
        with self.lock:
            if tracker is not self.tracker or epoch != self.epoch or tracker is None:
                return
            if point is None:
                self.markers.clear(reason)
            else:
                self.markers.observe(point, capture)
