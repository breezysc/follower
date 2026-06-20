import cv2
import numpy as np
import os

class VisionDebugger:
    def __init__(self, window_name='HSV Controls', window_size=(600, 200),
                 initial_values=None, config_path=None, enable_persistence=False):
        """
        HSV 参数调试器

        Args:
            window_name: 窗口名称
            window_size: 窗口尺寸 (width, height)
            initial_values: 初始HSV值字典，格式为 {'h_low': 90, 'h_high': 130, ...}
            config_path: 配置文件路径，如果提供则启用配置持久化
            enable_persistence: 是否启用配置持久化
        """
        defaults = {
            'h_low': 90, 'h_high': 130,
            's_low': 100, 's_high': 255,
            'v_low': 100, 'v_high': 255
        }

        values = initial_values if initial_values else defaults
        self.h_low = values.get('h_low', defaults['h_low'])
        self.h_high = values.get('h_high', defaults['h_high'])
        self.s_low = values.get('s_low', defaults['s_low'])
        self.s_high = values.get('s_high', defaults['s_high'])
        self.v_low = values.get('v_low', defaults['v_low'])
        self.v_high = values.get('v_high', defaults['v_high'])

        self.trackbar_window = window_name
        self.window_size = window_size
        self.config_path = config_path
        self.enable_persistence = enable_persistence and config_path is not None

        if self.enable_persistence:
            self.load_config()

    def load_config(self):
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 6:
                        self.h_low = int(lines[0].strip())
                        self.h_high = int(lines[1].strip())
                        self.s_low = int(lines[2].strip())
                        self.s_high = int(lines[3].strip())
                        self.v_low = int(lines[4].strip())
                        self.v_high = int(lines[5].strip())
                        print(f"Loaded HSV config: H[{self.h_low}-{self.h_high}] S[{self.s_low}-{self.s_high}] V[{self.v_low}-{self.v_high}]")
            except Exception as e:
                print(f"Failed to load config: {e}")

    def save_config(self):
        if not self.enable_persistence or not self.config_path:
            return
        try:
            with open(self.config_path, 'w') as f:
                f.write(f"{self.h_low}\n{self.h_high}\n{self.s_low}\n{self.s_high}\n{self.v_low}\n{self.v_high}\n")
            print(f"Saved HSV config: H[{self.h_low}-{self.h_high}] S[{self.s_low}-{self.s_high}] V[{self.v_low}-{self.v_high}]")
        except Exception as e:
            print(f"Failed to save config: {e}")

    def init_trackbars(self):
        cv2.namedWindow(self.trackbar_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.trackbar_window, *self.window_size)

        cv2.createTrackbar('H_low', self.trackbar_window, self.h_low, 179, self.on_trackbar)
        cv2.createTrackbar('H_high', self.trackbar_window, self.h_high, 179, self.on_trackbar)
        cv2.createTrackbar('S_low', self.trackbar_window, self.s_low, 255, self.on_trackbar)
        cv2.createTrackbar('S_high', self.trackbar_window, self.s_high, 255, self.on_trackbar)
        cv2.createTrackbar('V_low', self.trackbar_window, self.v_low, 255, self.on_trackbar)
        cv2.createTrackbar('V_high', self.trackbar_window, self.v_high, 255, self.on_trackbar)

    def on_trackbar(self, val):
        try:
            self.h_low = cv2.getTrackbarPos('H_low', self.trackbar_window)
            self.h_high = cv2.getTrackbarPos('H_high', self.trackbar_window)
            self.s_low = cv2.getTrackbarPos('S_low', self.trackbar_window)
            self.s_high = cv2.getTrackbarPos('S_high', self.trackbar_window)
            self.v_low = cv2.getTrackbarPos('V_low', self.trackbar_window)
            self.v_high = cv2.getTrackbarPos('V_high', self.trackbar_window)

            if self.enable_persistence:
                self.save_config()
        except Exception:
            pass

    def get_hsv_bounds(self):
        return (self.h_low, self.s_low, self.v_low), (self.h_high, self.s_high, self.v_high)

    def show_windows(self, frame, mask, result_frame, roi_frame=None):
        cv2.imshow('1. Original', frame)
        cv2.imshow('2. HSV Mask', mask)
        cv2.imshow('3. Detection Result', result_frame)
        if roi_frame is not None:
            cv2.imshow('4. ROI Region', roi_frame)