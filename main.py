import time
import cv2
import math
import numpy as np
import os
from datetime import datetime
from capture.screen import ScreenCapture
from capture.window_capture import WindowCapture
from vision.leader_detector import LeaderDetector
from vision.debugger import VisionDebugger as _VisionDebugger
from vision.roi_manager import ROIManager
from vision.ocr_engine import OCREngine
from logic.tracker import Tracker, TrackerState
from logic.rule_engine import RuleEngine
from logic.event_manager import EventManager
from control.keyboard import KeyboardController
from control.mouse_controller import MouseController
from pynput.keyboard import Key
import config

class TemplateCapture:
    def __init__(self):
        self.is_capturing = False
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0
        self.temp_frame = None
        self.window_name = 'Template Capture'
    
    def capture_template(self, frame):
        self.temp_frame = frame.copy()
        self.is_capturing = True
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        print("=== Template Capture Mode ===")
        print("Drag mouse to select the leader marker region")
        print("Press ENTER to save the template")
        print("Press ESC to cancel")
        
        while self.is_capturing:
            display_frame = self.temp_frame.copy()
            
            if self.start_x != 0 or self.start_y != 0:
                cv2.rectangle(display_frame, 
                             (self.start_x, self.start_y), 
                             (self.end_x, self.end_y), 
                             (0, 255, 0), 2)
            
            cv2.imshow(self.window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:
                self.is_capturing = False
                cv2.destroyWindow(self.window_name)
                print("Template capture cancelled")
                return None
            elif key == 13:
                if self.start_x != 0 and self.end_x != 0:
                    x1 = min(self.start_x, self.end_x)
                    y1 = min(self.start_y, self.end_y)
                    x2 = max(self.start_x, self.end_x)
                    y2 = max(self.start_y, self.end_y)
                    
                    template = self.temp_frame[y1:y2, x1:x2]
                    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'leader_template.png')
                    
                    os.makedirs(os.path.dirname(template_path), exist_ok=True)
                    cv2.imwrite(template_path, template)
                    
                    cv2.destroyWindow(self.window_name)
                    self.is_capturing = False
                    print(f"Template saved to: {template_path}")
                    print(f"Template size: {template.shape[1]}x{template.shape[0]}")
                    return template
        
        return None
    
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_x = x
            self.start_y = y
            self.end_x = x
            self.end_y = y
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            self.end_x = x
            self.end_y = y

class OCRDebugger:
    def __init__(self):
        self.ocr = OCREngine()
        self.rule_engine = RuleEngine()
        self.event_manager = EventManager()
        self.roi_manager = ROIManager()
        self.template_matcher = None
        self.template_saved = False
        
        # ROI_C的模板（单独保存）
        self.template_c2_saved = False
        self.template_c2_image = None
        self.template_c2_path = os.path.join(os.path.dirname(__file__), 'templates', 'target_c2_template.png')
        
        self.current_ocr_text = ""
        self.current_confidence = 0.0
        self.last_match_result = None
        
        self.screenshot_mode = False
        self.screenshot_start = None
        self.screenshot_end = None
        self.screenshot_frame = None
        self.screenshot_mode_for_c2 = False
        
        # OCR区域在combined窗口中的偏移量
        self.ocr_offset_x = 0
        # OCR区域的缩放比例
        self.ocr_scale_x = 1.0
        self.ocr_scale_y = 1.0
        
        self.roi_manager.load_config()
    
    def set_ocr_params(self, offset_x, scale_x, scale_y):
        """设置OCR区域的显示参数"""
        self.ocr_offset_x = offset_x
        self.ocr_scale_x = scale_x
        self.ocr_scale_y = scale_y
    
    def init_windows(self):
        cv2.namedWindow('OCR Debug Panel', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('OCR Debug Panel', 1400, 900)
        self.roi_manager.set_mode('select')
        cv2.setMouseCallback('OCR Debug Panel', self.mouse_callback)
    
    def mouse_callback(self, event, x, y, flags, param):
        # 坐标转换：鼠标在combined窗口的坐标 -> OCR原始帧的坐标
        # 需要考虑X偏移和缩放比例
        is_in_ocr_area = x >= self.ocr_offset_x if hasattr(self, 'ocr_offset_x') else False
        
        if self.screenshot_mode:
            # 在OCR区域内拖拽选择模板
            if is_in_ocr_area:
                # 将显示坐标转换为原始帧坐标
                roi_x = int((x - self.ocr_offset_x) / self.ocr_scale_x)
                roi_y = int(y / self.ocr_scale_y)
                
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.screenshot_start = (roi_x, roi_y)
                    self.screenshot_end = None
                elif event == cv2.EVENT_MOUSEMOVE and self.screenshot_start:
                    self.screenshot_end = (roi_x, roi_y)
                elif event == cv2.EVENT_LBUTTONUP and self.screenshot_start and self.screenshot_end:
                    x1 = min(self.screenshot_start[0], self.screenshot_end[0])
                    y1 = min(self.screenshot_start[1], self.screenshot_end[1])
                    x2 = max(self.screenshot_start[0], self.screenshot_end[0])
                    y2 = max(self.screenshot_start[1], self.screenshot_end[1])
                    
                    if self.screenshot_frame is not None and x2 > x1 and y2 > y1:
                        template_image = self.screenshot_frame[y1:y2, x1:x2]
                        if template_image.size > 0 and template_image.shape[0] > 10 and template_image.shape[1] > 10:
                            # 根据标志决定保存到哪个模板
                            if hasattr(self, 'screenshot_mode_for_c2') and self.screenshot_mode_for_c2:
                                self.save_target_c2_template(template_image)
                                print(f"[SCREENSHOT] Saved ROI_C template from selection")
                            else:
                                self.save_target_c_template(template_image)
                                print(f"[SCREENSHOT] Saved template from selection")
                    
                    self.screenshot_mode = False
                    self.screenshot_start = None
                    self.screenshot_end = None
                    self.screenshot_mode_for_c2 = False
            # ESC键取消screenshot mode（在主循环中处理）
        else:
            # ROI管理操作（使用显示坐标）
            if is_in_ocr_area:
                roi_x = int((x - self.ocr_offset_x) / self.ocr_scale_x)
                roi_y = int(y / self.ocr_scale_y)
                self.roi_manager.mouse_callback(event, roi_x, roi_y, flags, param)
            elif event == cv2.EVENT_LBUTTONDOWN:
                # 点击左侧区域，清除选择
                self.roi_manager.selected_roi = None
    
    def start_screenshot_mode(self, frame):
        self.screenshot_mode = True
        self.screenshot_frame = frame.copy()
        print("[SCREENSHOT] Entered screenshot mode. Drag to select.")
    
    def save_target_c_template(self, image):
        if image is not None and image.size > 0:
            path = os.path.join('templates', 'target_c_template.png')
            os.makedirs('templates', exist_ok=True)
            
            # 转换为灰度图（模板匹配用灰度图）
            if len(image.shape) == 3:
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) == 4:
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                image_gray = image
            
            cv2.imwrite(path, image_gray)
            print(f"[TEMPLATE] Saved: {path} ({image_gray.shape[1]}x{image_gray.shape[0]})")
            
            # 初始化模板（保存灰度图）
            self.template_image = image_gray
            self.template_saved = True
            return path
        return None
    
    def save_target_c2_template(self, image):
        """保存ROI_C的目标C2模板"""
        if image is not None and image.size > 0:
            path = os.path.join('templates', 'target_c2_template.png')
            os.makedirs('templates', exist_ok=True)
            
            # 转换为灰度图
            if len(image.shape) == 3:
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) == 4:
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                image_gray = image
            
            cv2.imwrite(path, image_gray)
            print(f"[TEMPLATE2] Saved: {path} ({image_gray.shape[1]}x{image_gray.shape[0]})")
            
            # 初始化模板
            self.template_c2_image = image_gray
            self.template_c2_saved = True
            return path
        return None
    
    def find_template_in_roi_c(self, roi_image):
        """在ROI_C区域搜索C2模板（多尺度匹配）"""
        if roi_image is None or self.template_c2_image is None:
            return None, 0.0, None
        
        # 转换为灰度图
        if len(roi_image.shape) == 3:
            roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        elif len(roi_image.shape) == 4:
            roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGRA2GRAY)
        else:
            roi_gray = roi_image
        
        if len(self.template_c2_image.shape) == 3:
            template_gray = cv2.cvtColor(self.template_c2_image, cv2.COLOR_BGR2GRAY)
        elif len(self.template_c2_image.shape) == 4:
            template_gray = cv2.cvtColor(self.template_c2_image, cv2.COLOR_BGRA2GRAY)
        else:
            template_gray = self.template_c2_image
        
        # 检查尺寸兼容性
        if roi_gray.shape[0] < template_gray.shape[0] or roi_gray.shape[1] < template_gray.shape[1]:
            return None, 0.0, None
        
        try:
            # 多尺度匹配
            scales = [0.8, 0.9, 1.0, 1.1, 1.2]
            best_match_val = 0.0
            best_match_loc = None
            
            for scale in scales:
                # 调整模板尺寸
                scaled_template = cv2.resize(template_gray, 
                                            (int(template_gray.shape[1] * scale), 
                                             int(template_gray.shape[0] * scale)),
                                            interpolation=cv2.INTER_CUBIC)
                
                # 检查尺寸
                if roi_gray.shape[0] < scaled_template.shape[0] or roi_gray.shape[1] < scaled_template.shape[1]:
                    continue
                
                result = cv2.matchTemplate(roi_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_match_val:
                    best_match_val = max_val
                    best_match_loc = max_loc
            
            if best_match_loc is None:
                return None, 0.0, None
            
            template_h, template_w = template_gray.shape
            center_offset = (template_w // 2, template_h // 2)
            
            return best_match_loc, best_match_val, center_offset
        except Exception as e:
            print(f"[TEMPLATE MATCH ERROR] {e}")
            return None, 0.0, None
    
    def find_template_in_roi(self, roi_image):
        if roi_image is None or self.template_image is None:
            return None, 0.0, None
        
        # 转换为灰度图（确保类型一致）
        if len(roi_image.shape) == 3:
            roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        elif len(roi_image.shape) == 4:
            roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGRA2GRAY)
        else:
            roi_gray = roi_image
        
        if len(self.template_image.shape) == 3:
            template_gray = cv2.cvtColor(self.template_image, cv2.COLOR_BGR2GRAY)
        elif len(self.template_image.shape) == 4:
            template_gray = cv2.cvtColor(self.template_image, cv2.COLOR_BGRA2GRAY)
        else:
            template_gray = self.template_image
        
        # 确保图像不为空
        if roi_gray.size == 0 or template_gray.size == 0:
            return None, 0.0, None
        
        # 确保尺寸兼容
        if roi_gray.shape[0] < template_gray.shape[0] or roi_gray.shape[1] < template_gray.shape[1]:
            return None, 0.0, None
        
        result = cv2.matchTemplate(roi_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        # 计算模板中心点
        template_h, template_w = template_gray.shape
        center_offset = (template_w // 2, template_h // 2)
        
        # 降低阈值以提高匹配成功率
        if max_val >= 0.5:
            return max_loc, max_val, center_offset
        return None, max_val, center_offset

class VisionDebugger(_VisionDebugger):
    """main.py 专用的 VisionDebugger，继承统一实现并添加 create_debug_panel 功能"""

    def __init__(self):
        super().__init__(
            window_name='Debug Panel',
            window_size=(1200, 800),
            initial_values={
                'h_low': config.BLUE_LOWER[0],
                'h_high': config.BLUE_UPPER[0],
                's_low': config.BLUE_LOWER[1],
                's_high': config.BLUE_UPPER[1],
                'v_low': config.BLUE_LOWER[2],
                'v_high': config.BLUE_UPPER[2]
            },
            config_path=os.path.join(os.path.dirname(__file__), 'hsv_config.txt'),
            enable_persistence=True
        )

    def create_debug_panel(self, original, mask, leader_pos, center_x, center_y, metrics, candidates):
        h, w = original.shape[:2]

        if original.shape[2] == 4:
            original = cv2.cvtColor(original, cv2.COLOR_BGRA2BGR)

        scale = 0.6
        small_w = int(w * scale)
        small_h = int(h * scale)

        original_small = cv2.resize(original, (small_w, small_h))

        if mask.shape[:2] != (small_h, small_w):
            mask_small = cv2.resize(mask, (small_w, small_h))
        else:
            mask_small = mask

        if len(mask_small.shape) == 2:
            mask_small = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        elif mask_small.shape[2] == 4:
            mask_small = cv2.cvtColor(mask_small, cv2.COLOR_BGRA2BGR)

        cv2.putText(original_small, 'Original', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(mask_small, 'HSV Mask', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        info_panel = np.zeros((small_h, 300, 3), dtype=np.uint8)
        info_y = 30
        spacing = 22

        cv2.putText(info_panel, '=== HSV Parameters ===', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        info_y += spacing
        cv2.putText(info_panel, f'H: {self.h_low}-{self.h_high}', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        info_y += spacing
        cv2.putText(info_panel, f'S: {self.s_low}-{self.s_high}', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        info_y += spacing
        cv2.putText(info_panel, f'V: {self.v_low}-{self.v_high}', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        info_y += spacing + 5

        cv2.putText(info_panel, '=== Candidates ===', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        info_y += spacing
        cv2.putText(info_panel, f'Total: {len(candidates)}', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        info_y += spacing

        for i, cand in enumerate(candidates[:5]):
            cv2.putText(info_panel, f'#{cand["id"]} pos=({cand["center"][0]},{cand["center"][1]})',
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1)
            info_y += 16
            cv2.putText(info_panel, f'    area={cand["area"]} w={cand["bbox"][2]} h={cand["bbox"][3]} ar={cand["aspect_ratio"]:.1f}',
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 200), 1)
            info_y += spacing

        if len(candidates) > 5:
            cv2.putText(info_panel, f'... and {len(candidates) - 5} more', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1)
            info_y += spacing

        info_y += 5
        cv2.putText(info_panel, '=== Selected Target ===', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        info_y += spacing

        if leader_pos:
            cv2.putText(info_panel, f'Leader: ({leader_pos[0]},{leader_pos[1]})', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.putText(info_panel, 'Leader: NOT FOUND', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        info_y += spacing

        cv2.putText(info_panel, f'Confidence: {metrics["confidence"]:.2f}', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        info_y += spacing
        cv2.putText(info_panel, 'Press S to Save', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        top_row = np.hstack([original_small, mask_small])
        bottom_row = np.hstack([info_panel, np.zeros((small_h, small_w * 2 - 300, 3), dtype=np.uint8)])

        panel = np.vstack([top_row, bottom_row])

        return panel

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def log_frame(leader_pos, dx, dy, distance, state, keys_pressed, fps, metrics, candidates):
    detection = "SUCCESS" if leader_pos else "FAILED"
    coord_str = f"({leader_pos[0]},{leader_pos[1]})" if leader_pos else "N/A"

    dirs = []
    if dx < -config.DEAD_ZONE:
        dirs.append("LEFT")
    elif dx > config.DEAD_ZONE:
        dirs.append("RIGHT")
    if dy < -config.DEAD_ZONE:
        dirs.append("UP")
    elif dy > config.DEAD_ZONE:
        dirs.append("DOWN")
    direction = "+ ".join(dirs) if dirs else "CENTERED"

    decision = "STOP" if state == TrackerState.STOP else "FOLLOW" if state == TrackerState.FOLLOW else "SEARCH"

    candidate_info = ""
    if candidates:
        selected_id = metrics.get('selected_id', '?')
        candidate_info = f" | Candidates: {len(candidates)} | Selected: #{selected_id}"

    log_str = (
        f"[FRAME] Detection: {detection:7s} | "
        f"Leader: {coord_str:12s} | "
        f"dx:{dx:4d} dy:{dy:4d} | "
        f"Distance: {int(distance):4d}px | "
        f"Direction: {direction:12s} | "
        f"Decision: {decision:6s} | "
        f"Keys: {','.join(keys_pressed) if keys_pressed else 'NONE'} | "
        f"FPS: {fps:.1f}{candidate_info}"
    )
    print(log_str)

def main():
    # 优先尝试 Path of Exile 2
    window_capture = WindowCapture("Path of Exile 2")
    
    if not window_capture.hwnd:
        # 如果没找到，尝试 Path of Exile
        window_capture = WindowCapture("path of exile")
    
    if window_capture.hwnd:
        screen_capture = window_capture
        window_rect = window_capture.get_window_rect()
        if window_rect and window_rect['width'] > 0 and window_rect['height'] > 0:
            center_x = window_rect['width'] // 2
            center_y = window_rect['height'] // 2
        else:
            print("[WARN] Window rect unavailable at startup - using fallback center.")
            center_x = config.CENTER_X
            center_y = config.CENTER_Y
        # 保存窗口捕获器，用于动态获取窗口位置
        active_window_capture = window_capture
        print(f"Capturing from: {window_capture.get_window_title()}")
        if window_rect:
            print(f"Window size: {window_rect['width']}x{window_rect['height']}")
    else:
        screen_capture = ScreenCapture(config.MAP_REGION)
        center_x = config.CENTER_X
        center_y = config.CENTER_Y
        active_window_capture = None
        print("Path of Exile not found, capturing from full screen")
    
    debugger = VisionDebugger()
    debugger.init_trackbars()
    
    ocr_debugger = OCRDebugger()
    
    # 初始化窗口（只在开始时创建一次）
    cv2.namedWindow('POE2 Debug Panel', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('POE2 Debug Panel', ocr_debugger.mouse_callback)
    
    tracker = Tracker(center_x, center_y, config.DEAD_ZONE, config.FOLLOW_DISTANCE_MIN, config.FOLLOW_DISTANCE_MAX)
    keyboard = KeyboardController()
    mouse = MouseController()
    template_capture = TemplateCapture()
    
    # 预加载LeaderDetector（减少每帧加载日志）
    leader_template_path = os.path.join(os.path.dirname(__file__), 'templates', 'leader_template.png')
    leader_detector = LeaderDetector(config.BLUE_LOWER, config.BLUE_UPPER, leader_template_path)
    
    # OCR默认规则：包含111
    ocr_debugger.rule_engine.add_rule('contains_111', 'contains', '111')
    
    # 加载目标C模板
    target_c_path = os.path.join('templates', 'target_c_template.png')
    if os.path.exists(target_c_path):
        template_img = cv2.imread(target_c_path, cv2.IMREAD_GRAYSCALE)
        if template_img is not None:
            ocr_debugger.template_image = template_img
            ocr_debugger.template_saved = True
            print(f"Loaded target C template: {target_c_path}")
    
    # 加载目标C2模板（ROI_C用）
    target_c2_path = os.path.join('templates', 'target_c2_template.png')
    if os.path.exists(target_c2_path):
        template_img2 = cv2.imread(target_c2_path, cv2.IMREAD_GRAYSCALE)
        if template_img2 is not None:
            ocr_debugger.template_c2_image = template_img2
            ocr_debugger.template_c2_saved = True
            print(f"Loaded target C2 template: {target_c2_path}")

    mode_names = ["VISUAL_DEBUG", "LIVE_MODE"]
    print(f"\n{'='*70}")
    print("POE2 Auto Follower + OCR System")
    print(f"Mode: {mode_names[config.DEBUG_MODE]}")
    print(f"{'='*70}")
    print("\nControls:")
    print("  ESC = Quit")
    print("  M = Toggle LIVE_MODE")
    print("  S = Save HSV config")
    print("  T = Capture leader template")
    print("  G = Capture target C template")
    print("  H = Capture ROI_C template (for second click)")
    print("  O = Test OCR on ROI_B (debug)")
    print("  1 = Select ROI mode")
    print("  2 = Draw ROI mode")
    print("  3 = Delete ROI")
    print("  4 = Save ROI config")
    print("  SPACE = Click target C manually")
    print(f"\nOCR Rule: contains('111')")
    print(f"{'='*70}\n")

    frame_count = 0
    fps_start_time = time.time()
    fps = 0.0
    
    # 传送状态机
    waitingForTrigger = True
    waitingForClear = False
    CLEAR_COOLDOWN = 60
    clear_cooldown = 0
    match_cooldown = 0
    
    # 窗口偏移量（用于鼠标点击）
    window_offset_x = 0
    window_offset_y = 0

    try:
        while True:
            frame_start = time.time()
            
            # 重置调试标志（只在特定帧输出，减少日志频率）
            if frame_count % 30 == 0:  # 每30帧输出一次
                pass  # 允许输出
            else:
                # 跳过调试输出
                ocr_debugger._last_match_logged = True
                ocr_debugger._last_roi_c_logged = True
            
            if hasattr(ocr_debugger, '_last_match_logged'):
                ocr_debugger._last_match_logged = False
            if hasattr(ocr_debugger, '_last_roi_c_logged'):
                ocr_debugger._last_roi_c_logged = False
            
            if clear_cooldown > 0:
                clear_cooldown -= 1
            if match_cooldown > 0:
                match_cooldown -= 1

            frame = screen_capture.capture()
            if frame is None:
                print(f"[SKIP] Capture returned empty frame (window may be minimized or invisible) - retrying...")
                time.sleep(0.1)
                continue

            # 更新ROI尺寸（基于当前捕获的帧大小）
            h, w = frame.shape[:2]
            ocr_debugger.roi_manager.set_current_size(w, h)
            
            # === WASD 跟随检测 ===
            lower_bound, upper_bound = debugger.get_hsv_bounds()
            leader_detector.lower_bound = lower_bound
            leader_detector.upper_bound = upper_bound
            
            roi_region = None
            if config.USE_ROI:
                roi_region = leader_detector.get_roi_region(center_x, center_y, config.ROI_SIZE)
            
            leader_pos = leader_detector.detect(frame, roi_region)
            metrics = leader_detector.get_detection_metrics()
            candidates = leader_detector.get_candidates()

            (dx, dy), state, avoid_info = tracker.update(leader_pos)
            distance = calculate_distance(center_x, center_y, leader_pos[0], leader_pos[1]) if leader_pos else 9999
            keys_pressed = keyboard.simulate_movement(dx, dy)

            if config.DEBUG_MODE == 1:
                if avoid_info.get('needs_avoid', False):
                    # 躲避模式：按下空格键
                    if avoid_info.get('press_space', False):
                        keyboard.update_movement(dx, dy, space=True)
                        direction_str = "LEFT" if avoid_info['avoid_direction'] < 0 else "RIGHT"
                        if avoid_info['phase'] == 1:
                            print(f"[AVOID] Phase 1: Dodging {direction_str} (angle={avoid_info['angle']}°)")
                        elif avoid_info['phase'] == 2:
                            print(f"[AVOID] Phase 2: Dodging OPPOSITE {direction_str} (angle={avoid_info['angle']}°)")
                        elif avoid_info['phase'] == 3:
                            print(f"[AVOID] Phase 3: Continuing {direction_str} (angle={avoid_info['angle']}°)")
                    else:
                        keyboard.update_movement(dx, dy)
                elif state == TrackerState.FOLLOW:
                    keyboard.update_movement(dx, dy)
                else:
                    keyboard.release_all()

            # === OCR 识别 ===
            roi_b_info = ocr_debugger.roi_manager.get_roi('ROI_B')
            roi_b_image = ocr_debugger.roi_manager.extract_roi(frame, 'ROI_B')
            
            target_c_pos = None
            target_c_confidence = 0.0
            
            # 目标C模板匹配（在ROI_A区域搜索目标）
            roi_a_info = ocr_debugger.roi_manager.get_roi('ROI_A')
            roi_a_image = ocr_debugger.roi_manager.extract_roi(frame, 'ROI_A')
            
            if roi_a_image is not None and roi_a_image.size > 0:
                pos_in_roi, confidence, center_offset = ocr_debugger.find_template_in_roi(roi_a_image)
                target_c_confidence = confidence
                
                if pos_in_roi and roi_a_info and center_offset:
                    # 点击模板左上角位置（更精确）
                    click_x = roi_a_info['x'] + pos_in_roi[0]
                    click_y = roi_a_info['y'] + pos_in_roi[1]
                    target_c_pos = (click_x, click_y)
                else:
                    target_c_pos = None
            
            # 目标C_2模板匹配（在ROI_C区域搜索）
            roi_c_info = ocr_debugger.roi_manager.get_roi('ROI_C')
            roi_c_image = ocr_debugger.roi_manager.extract_roi(frame, 'ROI_C')
            
            target_c2_pos = None
            target_c2_confidence = 0.0
            
            # 获取ROI_C的比例坐标（调试用）
            roi_c_ratio = ocr_debugger.roi_manager.rois.get('ROI_C')
            
            if roi_c_image is not None and roi_c_image.size > 0 and ocr_debugger.template_c2_saved:
                pos_in_roi_c, confidence_c, center_offset_c = ocr_debugger.find_template_in_roi_c(roi_c_image)
                target_c2_confidence = confidence_c
                
                if pos_in_roi_c and roi_c_info and center_offset_c:
                    # 点击模板左上角位置
                    click_x_c = roi_c_info['x'] + pos_in_roi_c[0]
                    click_y_c = roi_c_info['y'] + pos_in_roi_c[1]
                    target_c2_pos = (click_x_c, click_y_c)
            
            # 初始化 match_result（避免未定义错误）
            match_result = {'all_matched': False}
            
            # 检查 ROI_B 是否设置
            roi_b_info = ocr_debugger.roi_manager.get_roi('ROI_B')
            roi_b_image = ocr_debugger.roi_manager.extract_roi(frame, 'ROI_B')
            
            # 每30帧输出ROI_B状态（调试）
            if frame_count % 30 == 0:
                if roi_b_info:
                    print(f"[ROI_B] x={roi_b_info['x']} y={roi_b_info['y']} w={roi_b_info['w']} h={roi_b_info['h']}")
                else:
                    print(f"[ROI_B] NOT SET - press 2 to draw ROI_B over chat area")
            
            # 调试信息（只在检测到111时输出一次）
            if match_result and match_result['all_matched'] and waitingForTrigger and match_cooldown <= 0:
                if roi_c_ratio:
                    print(f"[DEBUG] ROI_C ratio: x={roi_c_ratio.get('x_ratio', 0):.4f}, y={roi_c_ratio.get('y_ratio', 0):.4f}, w={roi_c_ratio.get('width_ratio', 0):.4f}, h={roi_c_ratio.get('height_ratio', 0):.4f}")
                if roi_c_info:
                    print(f"[DEBUG] ROI_C abs: x={roi_c_info['x']}, y={roi_c_info['y']}, w={roi_c_info['w']}, h={roi_c_info['h']}")
                if roi_c_image is not None and roi_c_image.size > 0:
                    print(f"[DEBUG] ROI_C image: {roi_c_image.shape}, template_c2_saved={ocr_debugger.template_c2_saved}")
                else:
                    print(f"[DEBUG] ROI_C image: None or empty")
                if target_c2_confidence > 0:
                    print(f"[DEBUG] ROI_C match: pos={pos_in_roi_c}, conf={target_c2_confidence:.2f}")
                elif not ocr_debugger.template_c2_saved:
                    print(f"[DEBUG] ROI_C template not saved - press H to capture")
            
            # OCR识别
            if roi_b_image is not None and roi_b_image.size > 0:
                ocr_text, ocr_conf = ocr_debugger.ocr.recognize(roi_b_image)
                ocr_debugger.current_ocr_text = ocr_text
                ocr_debugger.current_confidence = ocr_conf
                
                match_result = ocr_debugger.rule_engine.evaluate(ocr_text)
                ocr_debugger.last_match_result = match_result
                
                # 每30帧输出一次OCR状态（调试）
                if frame_count % 30 == 0:
                    print(f"[OCR] text='{ocr_text}' conf={ocr_conf:.2f} matched={match_result['all_matched']} waiting={waitingForTrigger} cooldown={match_cooldown}")
                
                # 检测到111触发
                if match_result['all_matched'] and waitingForTrigger and match_cooldown <= 0:
                    print(f"\n[TRIGGER] OCR detected: '{ocr_text}'")
                    
                    # 第1步：点击目标C
                    # 动态获取窗口位置
                    if active_window_capture:
                        current_window_rect = active_window_capture.get_window_rect()
                        window_offset_x = current_window_rect['left']
                        window_offset_y = current_window_rect['top']
                    
                    if target_c_pos and target_c_confidence >= 0.7:
                        screen_x = int(target_c_pos[0]) + window_offset_x
                        screen_y = int(target_c_pos[1]) + window_offset_y
                        mouse.click(screen_x, screen_y)
                        print(f"[CLICK1] Window offset: ({window_offset_x}, {window_offset_y})")
                        print(f"[CLICK1] Clicked Target C at: ({screen_x}, {screen_y})")
                    elif roi_a_info:
                        # ROI坐标是窗口内的相对坐标，需要加上窗口偏移得到屏幕坐标
                        screen_x = roi_a_info['x'] + window_offset_x
                        screen_y = roi_a_info['y'] + window_offset_y
                        mouse.click(screen_x, screen_y)
                        print(f"[CLICK1] Window offset: ({window_offset_x}, {window_offset_y})")
                        print(f"[CLICK1] Clicked ROI_A at: ({screen_x}, {screen_y})")
                    
                    # 第2步：等待0.8秒后检查ROI_C
                    time.sleep(0.8)
                    
                    # 第3步：判断是否识别到ROI_C（优先使用模板匹配，失败则回退到区域中心点）
                    print(f"[DEBUG] ROI_C template: saved={ocr_debugger.template_c2_saved}, match_conf={target_c2_confidence:.2f}")

                    # 如果有ROI_C区域，无论模板匹配是否成功都点击中心点
                    if roi_c_info:
                        print(f"[DEBUG] ROI_C abs: x={roi_c_info['x']}, y={roi_c_info['y']}, w={roi_c_info['w']}, h={roi_c_info['h']}, window_offset=({window_offset_x}, {window_offset_y})")
                        roi_c_center_x = roi_c_info['x'] + roi_c_info['w'] // 2
                        roi_c_center_y = roi_c_info['y'] + roi_c_info['h'] // 2
                        screen_x_c = roi_c_center_x + window_offset_x
                        screen_y_c = roi_c_center_y + window_offset_y
                        mouse.click(screen_x_c, screen_y_c)

                        if target_c2_confidence >= 0.2:
                            print(f"[CLICK2] Clicked ROI_C center at: ({screen_x_c}, {screen_y_c}) (confidence: {target_c2_confidence:.2f})")
                        else:
                            print(f"[CLICK2] Template match failed (conf: {target_c2_confidence:.2f}), clicking ROI_C center at: ({screen_x_c}, {screen_y_c})")
                    else:
                        print(f"[SKIP] ROI_C not configured - please set ROI_C region")
                    
                    # 第4步：等待3秒
                    print(f"[WAIT] Waiting 3 seconds before checking...")
                    time.sleep(3)
                    
                    # 第5步：持续检查5秒
                    print(f"[CHECK] Starting 5-second continuous check for Target C...")
                    target_found = False
                    check_interval = 0.5  # 每0.5秒检查一次
                    max_checks = int(5 / check_interval)  # 最多10次
                    
                    for i in range(max_checks):
                        frame_check = screen_capture.capture()
                        if frame_check is None:
                            print(f"[SKIP] Empty frame during C check {i+1}/{max_checks}")
                            time.sleep(check_interval)
                            continue
                        roi_a_image_check = ocr_debugger.roi_manager.extract_roi(frame_check, 'ROI_A')
                        target_c_check_pos, target_c_check_conf, _ = ocr_debugger.find_template_in_roi(roi_a_image_check)
                        
                        if target_c_check_pos and target_c_check_conf >= 0.5:
                            target_found = True
                            print(f"[FOUND] Target C detected (conf={target_c_check_conf:.2f}) at check {i+1}/{max_checks}")
                            break
                        else:
                            time.sleep(check_interval)
                    
                    # 根据检查结果决定是否发送/clear
                    if target_found:
                        print(f"[CONFIRM] Sending /clear command")
                        keyboard.press_key(Key.enter)
                        time.sleep(0.1)
                        keyboard.release_key(Key.enter)
                        time.sleep(0.2)
                        keyboard.type_text('/clear')
                        time.sleep(0.1)
                        keyboard.press_key(Key.enter)
                        time.sleep(0.1)
                        keyboard.release_key(Key.enter)
                    else:
                        print(f"[SKIP] Target C not found in 5 seconds, skipping /clear")
                    
                    # 重置状态
                    waitingForTrigger = False
                    waitingForClear = False
                    match_cooldown = 30
                    print(f"[STATE] Reset to waiting for trigger")

            # === 日志输出（减少频率）===
            frame_count += 1
            if frame_count % 30 == 0:  # 改为30帧输出一次
                elapsed = time.time() - fps_start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                frame_count = 0
                fps_start_time = time.time()
                
                log_frame(leader_pos, dx, dy, distance, state, keys_pressed, fps, metrics, candidates)
            
            mask = leader_detector.get_mask()
            if mask is not None:
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            else:
                mask_3ch = np.zeros_like(frame)

            # 主调试面板（Original + HSV Mask）
            debug_panel = debugger.create_debug_panel(frame, mask_3ch, leader_pos, center_x, center_y, metrics, candidates)
            
            # OCR调试面板（ROI绘制）
            ocr_display = ocr_debugger.roi_manager.draw_rois(frame)
            if len(ocr_display.shape) == 3 and ocr_display.shape[2] == 4:
                ocr_display = cv2.cvtColor(ocr_display, cv2.COLOR_BGRA2BGR)
            
            # Screenshot模式预览（坐标需要转换为显示坐标）
            if ocr_debugger.screenshot_mode:
                cv2.putText(ocr_display, "SCREENSHOT MODE - Drag to select", 
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if ocr_debugger.screenshot_start and ocr_debugger.screenshot_end:
                    # 将原始帧坐标转换为显示坐标
                    x1 = int(ocr_debugger.screenshot_start[0] / scale_x)
                    y1 = int(ocr_debugger.screenshot_start[1] / scale_y)
                    x2 = int(ocr_debugger.screenshot_end[0] / scale_x)
                    y2 = int(ocr_debugger.screenshot_end[1] / scale_y)
                    cv2.rectangle(ocr_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(ocr_display, f"Selected: {abs(x2-x1)}x{abs(y2-y1)}", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 调整高度一致
            target_h = min(debug_panel.shape[0], ocr_display.shape[0])
            debug_panel_resized = cv2.resize(debug_panel, (debug_panel.shape[1], target_h))
            ocr_display_resized = cv2.resize(ocr_display, (ocr_display.shape[1], target_h))
            
            # 计算OCR区域的缩放比例
            if ocr_display.shape[0] > 0 and target_h > 0:
                scale_y = ocr_display.shape[0] / target_h
            else:
                scale_y = 1.0
            if ocr_display.shape[1] > 0 and ocr_display_resized.shape[1] > 0:
                scale_x = ocr_display.shape[1] / ocr_display_resized.shape[1]
            else:
                scale_x = 1.0
            
            # 设置OCR区域参数（用于鼠标坐标转换）
            ocr_debugger.set_ocr_params(debug_panel_resized.shape[1], scale_x, scale_y)
            
            # 左右拼接两个窗口
            combined = np.hstack([debug_panel_resized, ocr_display_resized])
            
            cv2.imshow('POE2 Debug Panel', combined)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                # ESC键：如果是screenshot模式则取消，否则退出程序
                if ocr_debugger.screenshot_mode:
                    ocr_debugger.screenshot_mode = False
                    ocr_debugger.screenshot_start = None
                    ocr_debugger.screenshot_end = None
                    print("[SCREENSHOT] Cancelled")
                else:
                    break
            elif key == ord("m") or key == ord("M"):
                config.DEBUG_MODE = 1 if config.DEBUG_MODE == 0 else 0
                print(f"Mode changed to: {mode_names[config.DEBUG_MODE]}")
            elif key == ord("s") or key == ord("S"):
                debugger.save_config()
            elif key == ord("t") or key == ord("T"):
                cv2.destroyWindow('POE2 Debug Panel')
                template_capture.capture_template(frame)
                debugger.init_trackbars()
                combined = np.hstack([debug_panel, ocr_display])
                cv2.imshow('POE2 Debug Panel', combined)
            elif key == ord("g") or key == ord("G"):
                ocr_debugger.start_screenshot_mode(frame)
            elif key == ord("h") or key == ord("H"):
                ocr_debugger.start_screenshot_mode(frame)
                ocr_debugger.screenshot_mode_for_c2 = True
                print("[SCREENSHOT] Capturing ROI_C template (H)")
            elif key == ord("1"):
                ocr_debugger.roi_manager.set_mode('select')
                print("ROI Mode: Select (click to select ROI)")
            elif key == ord("2"):
                ocr_debugger.roi_manager.set_mode('draw')
                print("ROI Mode: Draw (drag mouse to draw ROI)")
            elif key == ord("3"):
                if ocr_debugger.roi_manager.selected_roi:
                    name = ocr_debugger.roi_manager.selected_roi
                    ocr_debugger.roi_manager.delete_roi(name)
                    print(f"Deleted ROI: {name}")
            elif key == ord("4"):
                ocr_debugger.roi_manager.save_config()
                print("ROI config saved")
            elif key == ord("o") or key == ord("O"):
                # 测试 ROI_B 区域的 OCR 识别
                roi_b_image = ocr_debugger.roi_manager.extract_roi(frame, 'ROI_B')
                if roi_b_image is not None and roi_b_image.size > 0:
                    text_result, conf_result = ocr_debugger.ocr.recognize(roi_b_image)
                    print(f"[OCR TEST] ROI_B text: '{text_result}' confidence: {conf_result:.2f}")
                    if text_result:
                        # 检查是否包含触发关键词
                        if '111' in text_result:
                            print(f"[OCR TEST] Found trigger '111' in text!")
                        else:
                            print(f"[OCR TEST] No '111' found")
                    else:
                        print(f"[OCR TEST] No text detected")
                else:
                    print(f"[OCR TEST] ROI_B not set or empty - press 2 to draw ROI_B")
            elif key == ord(" "):
                # 动态获取窗口位置
                if active_window_capture:
                    current_window_rect = active_window_capture.get_window_rect()
                    window_offset_x = current_window_rect['left']
                    window_offset_y = current_window_rect['top']
                
                if target_c_pos and target_c_confidence >= 0.7:
                    screen_x = int(target_c_pos[0]) + window_offset_x
                    screen_y = int(target_c_pos[1]) + window_offset_y
                    mouse.click(screen_x, screen_y)
                    print(f"[CLICK] Window offset: ({window_offset_x}, {window_offset_y})")
                    print(f"[CLICK] Clicked at: ({screen_x}, {screen_y}) confidence={target_c_confidence:.2f}")
                else:
                    print(f"[CLICK] No target - pos={target_c_pos}, conf={target_c_confidence:.2f}")

            elapsed = time.time() - frame_start
            sleep_time = max(0, (config.TICK_INTERVAL / 1000.0) - elapsed)
            time.sleep(sleep_time)

    finally:
        debugger.save_config()
        keyboard.release_all()
        screen_capture.release()
        cv2.destroyAllWindows()
        print("POE2 Auto Follower stopped.")

if __name__ == "__main__":
    main()