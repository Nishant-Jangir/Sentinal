"""
PipelineManager: spins up one thread per camera feed. Threads are fine here
(not multiprocessing) because:
  - cv2 and torch release the GIL during their actual compute
  - all GPU work funnels through a shared semaphore anyway (see config.
    MAX_CONCURRENT_HEAVY_JOBS), so true parallel GPU execution isn't the goal --
    we're bounding GPU concurrency deliberately, not maximizing CPU parallelism.
  - much simpler to share the loaded models (Zero-DCE, Real-ESRGAN) across
    threads than to pickle/reload them per process.
"""

import threading
import time
import cv2

import config
from core.camera_worker import CameraWorker


class PipelineManager:
    def __init__(self, camera_sources: list, zero_dce, clahe, real_esrgan, on_result=None):
        """
        camera_sources: list of cv2.VideoCapture-compatible sources
                        (RTSP URLs, video files, or webcam indices)
        on_result: optional callback(dict) called with each worker's output
                   dict (see CameraWorker.process_frame) -- wire this up to
                   your UI/display/logging layer.
        """
        self.camera_sources = camera_sources
        self.on_result = on_result or (lambda result: None)
        self.heavy_semaphore = threading.Semaphore(config.MAX_CONCURRENT_HEAVY_JOBS)

        self.workers = [
            CameraWorker(cam_id, zero_dce, clahe, real_esrgan, self.heavy_semaphore)
            for cam_id in range(len(camera_sources))
        ]
        self._stop_event = threading.Event()
        self._threads = []

    def _run_camera_loop(self, cam_id: int):
        source = self.camera_sources[cam_id]
        worker = self.workers[cam_id]
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"[Camera {cam_id}] ERROR: could not open source {source}")
            return

        print(f"[Camera {cam_id}] started")
        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                # For video files this means EOF -- loop for demo purposes.
                # For RTSP this means dropped connection -- add reconnect logic here.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            result = worker.process_frame(frame)
            self.on_result(result)

        cap.release()
        print(f"[Camera {cam_id}] stopped")

    def start(self):
        for cam_id in range(len(self.camera_sources)):
            t = threading.Thread(target=self._run_camera_loop, args=(cam_id,), daemon=True)
            t.start()
            self._threads.append(t)
        print(f"[Manager] {len(self._threads)} camera threads started "
              f"(max {config.MAX_CONCURRENT_HEAVY_JOBS} concurrent heavy-tier jobs)")

    def request_inspect(self, cam_id: int):
        """Wire this to a UI button: 'zoom and enhance' on a specific camera."""
        if 0 <= cam_id < len(self.workers):
            self.workers[cam_id].request_inspect()

    def stop(self):
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=2)
