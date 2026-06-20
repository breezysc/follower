import time
import cv2
import math
import numpy as np
import os
from datetime import datetime

from capture.screen import ScreenCapture
from capture.window_capture import WindowCapture
from vision.leader_detector import LeaderDetector
from vision.roi_manager import ROIManager
from vision.ocr_engine import OCREngine
from logic.tracker import Tracker, TrackerState
from logic.rule_engine import RuleEngine
from logic.event_manager import EventManager
from pynput.keyboard import Key
from control.keyboard import KeyboardController
from control.mouse_controller import MouseController
import config

class TemplateMatcher:
    def __init__(self, template_path=None, threshold=0.7):
        self.template = None
        self.template_path = template_path
        self.threshold = threshold
        self.best_match_pos = None
        self.best_match_confidence = 0.0
        
        if template_path and os.path.exists(template_path):
            self.load_template(template_path)
    
    def load_template(self, path):
        self.template = cv2.imread(path, cv2.IMREAD_COLOR)
        if self.template is not None:
            print(f"Loaded template: {path} (size: {self.template.shape})")
    
    def save_template(self, image, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, image)
        self.template = image
        print(f"Saved template: {path}")
    
    def match(self, search_image):
        if self.template is None or search_image is None:
            self.best_match_pos = None
            self.best_match_confidence = 0.0
            return None, 0.0
        
        result = cv2.matchTemplate(search_image, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        self.best_match_confidence = max_val
        self.best_match_pos = None
        
        if max_val >= self.threshold:
            h, w = self.template.shape[:2]
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            self.best_match_pos = (center_x, center_y)
            return self.best_match_pos, max_val
        
        return None, max_val
    
    def get_last_match(self):
        return self.best_match_pos, self.best_match_confidence

class RegionCapture:
    def __init__(self, name, save_dir='captures'):
        self.name = name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def capture_and_save(self, image, suffix=""):
        if image is None or image.size == 0:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name}_{timestamp}{suffix}.png"
        filepath = os.path.join(self.save_dir, filename)
        cv2.imwrite(filepath, image)
        return filepath

class OCRDebugger:
    def __init__(self):
        self.ocr = OCREngine()
        self.rule_engine = RuleEngine()
        self.event_manager = EventManager()
        
        self.roi_manager = ROIManager()
        
        self.region_a = RegionCapture('ROI_A')
        self.region_b = RegionCapture('ROI_B')
        self.template_c = RegionCapture('Template_C')
        
        self.template_matcher = TemplateMatcher(threshold=0.7)
        
        self.lower_bound = (config.BLUE_LOWER[0], config.BLUE_LOWER[1], config.BLUE_LOWER[2])
        self.upper_bound = (config.BLUE_UPPER[0], config.BLUE_UPPER[1], config.BLUE_UPPER[2])
        self.leader_detector = LeaderDetector(self.lower_bound, self.upper_bound)
        
        self.current_ocr_text = ""
        self.current_confidence = 0.0
        self.last_match_result = None
        self.last_target_pos = None
        self.last_match_confidence = 0.0
        self.template_saved = False
        
        self.screenshot_mode = False
        self.screenshot_start = None
        self.screenshot_end = None
        self.screenshot_frame = None
        
    def init_windows(self):
        cv2.namedWindow('OCR Debug Panel', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('OCR Debug Panel', 1400, 900)
        
        self.roi_manager.set_mode('select')
        cv2.setMouseCallback('OCR Debug Panel', self.mouse_callback)
        
    def mouse_callback(self, event, x, y, flags, param):
        if self.screenshot_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.screenshot_start = (x, y)
                self.screenshot_end = None
            elif event == cv2.EVENT_MOUSEMOVE and self.screenshot_start:
                self.screenshot_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and self.screenshot_start and self.screenshot_end:
                x1 = min(self.screenshot_start[0], self.screenshot_end[0])
                y1 = min(self.screenshot_start[1], self.screenshot_end[1])
                x2 = max(self.screenshot_start[0], self.screenshot_end[0])
                y2 = max(self.screenshot_start[1], self.screenshot_end[1])
                
                if self.screenshot_frame is not None:
                    template_image = self.screenshot_frame[y1:y2, x1:x2]
                    if template_image.size > 0 and template_image.shape[0] > 10 and template_image.shape[1] > 10:
                        if template_image.shape[2] == 4:
                            template_image = cv2.cvtColor(template_image, cv2.COLOR_BGRA2BGR)
                        saved_path = self.save_target_c_template(template_image)
                        print(f"[SCREENSHOT] Saved template from selection: {saved_path}")
                        print(f"Template size: {template_image.shape[1]}x{template_image.shape[0]}")
                    else:
                        print("[SCREENSHOT] Selection too small, please select a larger area")
                
                self.screenshot_mode = False
                self.screenshot_start = None
                self.screenshot_end = None
        else:
            self.roi_manager.mouse_callback(event, x, y, flags, param)
        
    def start_screenshot_mode(self, frame):
        self.screenshot_mode = True
        self.screenshot_frame = frame.copy()
        print("[SCREENSHOT] Entered screenshot mode. Drag mouse to select area, release to save.")
        
    def save_target_c_template(self, image):
        if image is not None and image.size > 0:
            path = os.path.join('templates', 'target_c_template.png')
            self.template_matcher.save_template(image, path)
            self.template_saved = True
            return path
        return None
    
    def create_debug_panel(self, original, leader_pos, leader_detected, metrics, center_x, center_y, target_pos=None, match_confidence=0.0):
        display = original.copy()
        
        if display.shape[2] == 4:
            display = cv2.cvtColor(display, cv2.COLOR_BGRA2BGR)
        
        display = self.roi_manager.draw_rois(display)
        
        if target_pos:
            cv2.circle(display, target_pos, 10, (0, 0, 255), -1)
            cv2.circle(display, target_pos, 15, (0, 0, 255), 2)
            cv2.putText(display, f'TARGET C ({match_confidence:.2f})', 
                       (target_pos[0] + 15, target_pos[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if leader_pos:
            cv2.circle(display, leader_pos, 8, (0, 255, 0), -1)
            cv2.circle(display, leader_pos, 12, (0, 255, 0), 2)
            cv2.putText(display, 'LEADER', (leader_pos[0] + 15, leader_pos[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        h, w = display.shape[:2]
        
        panel_w = 400
        panel_h = h
        
        info_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        
        info_y = 30
        spacing = 25
        
        cv2.putText(info_panel, '=== OCR Results (Region B) ===', (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        info_y += spacing
        
        cv2.putText(info_panel, f'Text: {self.current_ocr_text or "N/A"}', (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        info_y += spacing
        
        conf_color = (0, 255, 0) if self.current_confidence >= 80 else (0, 255, 255) if self.current_confidence >= 60 else (0, 0, 255)
        cv2.putText(info_panel, f'Confidence: {self.current_confidence:.1f}%', (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, conf_color, 1)
        info_y += spacing + 5
        
        cv2.putText(info_panel, '=== Rule Status ===', (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        info_y += spacing
        
        if self.last_match_result:
            all_matched = self.last_match_result['all_matched']
            status_color = (0, 255, 0) if all_matched else (0, 0, 255)
            status_text = "MATCHED" if all_matched else "NO MATCH"
            cv2.putText(info_panel, f'Status: {status_text}', (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            info_y += spacing
            
            for rule_name, result in self.last_match_result['results'].items():
                match_color = (0, 255, 0) if result['match'] else (0, 0, 255)
                cv2.putText(info_panel, f'{rule_name}: {result["match"]}', (20, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, match_color, 1)
                info_y += 18
        else:
            cv2.putText(info_panel, 'Status: No evaluation', (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 1)
            info_y += spacing
            
        info_y += 5
        cv2.putText(info_panel, '=== Target C (Template Match) ===', (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        info_y += spacing
        
        cv2.putText(info_panel, f'Template Saved: {"YES" if self.template_saved else "NO"}', (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0) if self.template_saved else (0, 0, 255), 1)
        info_y += 18
        
        if target_pos:
            cv2.putText(info_panel, f'Detected: YES', (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            info_y += 18
            cv2.putText(info_panel, f'Position: ({target_pos[0]}, {target_pos[1]})', (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            info_y += 18
            conf_color = (0, 255, 0) if match_confidence >= 0.8 else (0, 255, 255) if match_confidence >= 0.6 else (0, 0, 255)
            cv2.putText(info_panel, f'Match Confidence: {match_confidence:.3f}', (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, conf_color, 1)
        else:
            cv2.putText(info_panel, f'Detected: NO', (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            info_y += 18
            cv2.putText(info_panel, f'Best Confidence: {match_confidence:.3f}', (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        
        info_y += spacing + 5
        cv2.putText(info_panel, '=== Leader Detection (Full Screen) ===', (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        info_y += spacing
        
        detect_color = (0, 255, 0) if leader_detected else (0, 0, 255)
        cv2.putText(info_panel, f'Leader: {"DETECTED" if leader_detected else "LOST"}', (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, detect_color, 1)
        info_y += 18
        
        if leader_pos:
            cv2.putText(info_panel, f'Position: ({leader_pos[0]}, {leader_pos[1]})', (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            info_y += 18
        
        cv2.putText(info_panel, f'Candidates: {metrics.get("candidate_count", 0)}', (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        info_y += spacing
        
        info_y += 5
        cv2.putText(info_panel, '=== Event Log ===', (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        info_y += spacing
        
        event_status = self.event_manager.get_status()
        cv2.putText(info_panel, f'Match Count: {event_status["match_count"]}', (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        info_y += 18
        
        if event_status['last_match_time']:
            time_str = event_status['last_match_time'].strftime('%H:%M:%S')
            cv2.putText(info_panel, f'Last Match: {time_str}', (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        info_y += spacing + 10
        cv2.putText(info_panel, '=== Controls ===', (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        info_y += spacing
        
        controls = [
            '1=Select ROI  2=Draw ROI',
            '3=Delete ROI  4=Save',
            '5=Clear Rules  6=Add Rule',
            'G/T=Manual Screenshot (drag)',
            'SPACE=Click Target C',
            'ESC=Exit'
        ]
        for ctrl in controls:
            cv2.putText(info_panel, ctrl, (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            info_y += 18
        
        main_with_panel = np.hstack([display, info_panel])
        
        return main_with_panel

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def main():
    print("POE2 OCR Event Detection System")
    print("="*70)
    print("Architecture:")
    print("  ROI_A: Teleport area (search for Target C)")
    print("  ROI_B: OCR trigger area (password/match condition)")
    print("  Target C: Teleport template (saved reference image)")
    print("  Leader: Full screen detection for WASD follow")
    print("="*70)
    
    window_capture = WindowCapture("path of exile")
    
    if window_capture.hwnd:
        screen_capture = window_capture
        window_rect = window_capture.get_window_rect()
        center_x = window_rect['width'] // 2
        center_y = window_rect['height'] // 2
        print(f"Capturing from Path of Exile window")
        print(f"Window size: {window_rect['width']}x{window_rect['height']}")
    else:
        screen_capture = ScreenCapture(config.MAP_REGION)
        center_x = config.CENTER_X
        center_y = config.CENTER_Y
        print("Path of Exile not found, capturing from full screen")
    
    debugger = OCRDebugger()
    debugger.init_windows()
    
    tracker = Tracker(center_x, center_y, config.DEAD_ZONE, config.FOLLOW_DISTANCE_MIN, config.FOLLOW_DISTANCE_MAX)
    keyboard = KeyboardController()
    mouse = MouseController()
    
    debugger.rule_engine.add_rule('contains_111', 'contains', '111')
    
    template_path = os.path.join('templates', 'target_c_template.png')
    if os.path.exists(template_path):
        debugger.template_matcher.load_template(template_path)
        debugger.template_saved = True
        print(f"Loaded existing template: {template_path}")
    
    print("\nDefault rule: contains('111')")
    print("\nControls:")
    print("  1 = Select ROI mode")
    print("  2 = Draw ROI mode")
    print("  3 = Delete selected ROI")
    print("  4 = Save ROI config")
    print("  5 = Clear rules")
    print("  6 = Add custom rule")
    print("  G/T = Manual Screenshot (drag to select template)")
    print("  SPACE = Click detected Target C")
    print("  ESC = Exit")
    print("\n" + "="*70)
    
    frame_count = 0
    fps_start_time = time.time()
    fps = 0.0
    
    match_cooldown = 0
    
    # 状态机变量
    waitingForTrigger = True   # 等待111触发
    waitingForClear = False    # 等待清理聊天记录
    waitingForClearStartTime = None  # 进入waitingForClear的时间
    CLEAR_MIN_WAIT = 2.5  # 最小等待时间（秒）
    CLEAR_COOLDOWN = 60  # Clear操作后的冷却帧数
    
    clear_cooldown = 0  # Clear操作后的冷却计数器
    
    try:
        while True:
            frame_start = time.time()
            
            # 更新冷却计数器
            if clear_cooldown > 0:
                clear_cooldown -= 1
            
            frame = screen_capture.capture()
            
            roi_a_info = debugger.roi_manager.get_roi('ROI_A')
            roi_b_info = debugger.roi_manager.get_roi('ROI_B')
            
            roi_a_image = debugger.roi_manager.extract_roi(frame, 'ROI_A')
            roi_b_image = debugger.roi_manager.extract_roi(frame, 'ROI_B')
            
            target_c_pos = None
            target_c_confidence = 0.0
            
            if roi_a_image is not None and roi_a_image.size > 0:
                debugger.region_a.capture_and_save(roi_a_image)
                
                if debugger.template_matcher.template is not None:
                    if roi_a_image.shape[2] == 4:
                        search_image = cv2.cvtColor(roi_a_image, cv2.COLOR_BGRA2BGR)
                    else:
                        search_image = roi_a_image
                    
                    pos_in_roi, confidence = debugger.template_matcher.match(search_image)
                    target_c_confidence = confidence
                    
                    if pos_in_roi and roi_a_info:
                        target_c_pos = (
                            roi_a_info['x'] + pos_in_roi[0],
                            roi_a_info['y'] + pos_in_roi[1]
                        )
            
            leader_pos = debugger.leader_detector.detect(frame)
            metrics = debugger.leader_detector.get_detection_metrics()

            (dx, dy), state = tracker.update(leader_pos)

            distance = calculate_distance(center_x, center_y, leader_pos[0], leader_pos[1]) if leader_pos else 9999

            keys_pressed = keyboard.simulate_movement(dx, dy)

            if config.DEBUG_MODE == 1:
                if state == TrackerState.FOLLOW:
                    keyboard.update_movement(dx, dy)
                else:
                    keyboard.release_all()
            
            ocr_match = False
            if roi_b_image is not None and roi_b_image.size > 0:
                debugger.region_b.capture_and_save(roi_b_image)
                
                ocr_text, ocr_conf = debugger.ocr.recognize(roi_b_image)
                debugger.current_ocr_text = ocr_text
                debugger.current_confidence = ocr_conf
                
                # 调试：每10帧打印一次OCR识别结果
                if frame_count % 10 == 0 and ocr_text:
                    print(f"[OCR] Text: '{ocr_text}' | Conf: {ocr_conf:.1f}%")
                
                match_result = debugger.rule_engine.evaluate(ocr_text)
                debugger.last_match_result = match_result
                
                if match_result['all_matched']:
                    ocr_match = True
                    debugger.event_manager.on_match(ocr_text, match_result['matched_rules'])
                    print(f"\n[TRIGGER] OCR matched: '{ocr_text}' | State: Trigger={waitingForTrigger}, Clear={waitingForClear}")
                    
                    if waitingForTrigger and match_cooldown <= 0:
                        # 执行传送
                        if target_c_pos and target_c_confidence >= 0.7:
                            mouse.click(target_c_pos[0], target_c_pos[1])
                            print(f"[CLICK] Clicked Target C at: {target_c_pos} (confidence: {target_c_confidence:.3f})")
                            debugger.template_c.capture_and_save(roi_a_image, "_clicked")
                            
                            roi_c_info = debugger.roi_manager.get_roi('ROI_C')
                            if roi_c_info:
                                time.sleep(0.8)
                                cx = roi_c_info['x'] + roi_c_info['w'] // 2
                                cy = roi_c_info['y'] + roi_c_info['h'] // 2
                                mouse.click(cx, cy)
                                print(f"[CLICK] Clicked ROI_C at: ({cx}, {cy})")
                        elif roi_a_info:
                            cx = roi_a_info['x'] + roi_a_info['w'] // 2
                            cy = roi_a_info['y'] + roi_a_info['h'] // 2
                            mouse.click(cx, cy)
                            print(f"[CLICK] Clicked ROI_A center at: ({cx}, {cy})")
                        
                        # 切换状态：进入waitingForClear
                        waitingForTrigger = False
                        waitingForClear = True
                        waitingForClearStartTime = time.time()
                        match_cooldown = 30
                        print(f"[STATE] Switched to waitingForClear, min wait: {CLEAR_MIN_WAIT}s")
                        
                    elif waitingForClear and clear_cooldown <= 0:
                        # 检查是否满足最小等待时间
                        elapsed = time.time() - waitingForClearStartTime if waitingForClearStartTime else 0
                        if elapsed >= CLEAR_MIN_WAIT:
                            # 执行clear命令
                            print(f"[CLEAR] Sending /clear command (waited {elapsed:.1f}s)")
                            
                            # 按回车打开聊天框
                            keyboard.press_key(Key.enter)
                            time.sleep(0.1)
                            keyboard.release_key(Key.enter)
                            time.sleep(0.2)
                            
                            # 输入 /clear
                            keyboard.type_text('/clear')
                            time.sleep(0.1)
                            
                            # 按回车发送
                            keyboard.press_key(Key.enter)
                            time.sleep(0.1)
                            keyboard.release_key(Key.enter)
                            
                            # 切换状态：回到waitingForTrigger
                            waitingForTrigger = True
                            waitingForClear = False
                            waitingForClearStartTime = None
                            clear_cooldown = CLEAR_COOLDOWN
                            print(f"[STATE] Switched to waitingForTrigger, clear cooldown: {CLEAR_COOLDOWN} frames")
                        else:
                            print(f"[CLEAR] Waiting... {elapsed:.1f}s / {CLEAR_MIN_WAIT}s")
                
                else:
                    debugger.event_manager.on_no_match(ocr_text)
            
            if match_cooldown > 0:
                match_cooldown -= 1

            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.time() - fps_start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                frame_count = 0
                fps_start_time = time.time()

            debug_panel = debugger.create_debug_panel(
                frame, leader_pos, leader_pos is not None, 
                metrics, center_x, center_y, target_c_pos, target_c_confidence
            )
            
            cv2.imshow('OCR Debug Panel', debug_panel)

            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:
                break
            elif key == ord("1"):
                debugger.roi_manager.set_mode('select')
                print("Mode: Select ROI")
            elif key == ord("2"):
                debugger.roi_manager.set_mode('draw')
                print("Mode: Draw ROI")
            elif key == ord("3"):
                if debugger.roi_manager.selected_roi:
                    name = debugger.roi_manager.selected_roi
                    debugger.roi_manager.delete_roi(name)
                    debugger.roi_manager.selected_roi = None
                    print(f"Deleted ROI: {name}")
            elif key == ord("4"):
                debugger.roi_manager.save_config()
                print("ROI config saved")
            elif key == ord("5"):
                debugger.rule_engine.clear_rules()
                print("Rules cleared")
            elif key == ord("6"):
                try:
                    rule_type = input("Enter rule type (contains/starts_with/regex): ").strip()
                    rule_value = input("Enter rule value: ").strip()
                    if rule_type and rule_value:
                        debugger.rule_engine.add_rule(f'custom_{len(debugger.rule_engine.rules)}', rule_type, rule_value)
                except:
                    pass
            elif key == ord("g") or key == ord("G"):
                debugger.start_screenshot_mode(frame)
            elif key == ord("t") or key == ord("T"):
                debugger.start_screenshot_mode(frame)
                print("[TEMPLATE] Entered screenshot mode. Drag to select template area.")
            elif key == ord(" "):
                if target_c_pos and target_c_confidence >= 0.7:
                    mouse.click(target_c_pos[0], target_c_pos[1])
                    print(f"[MANUAL] Clicked Target C at: {target_c_pos} (confidence: {target_c_confidence:.3f})")
                    
                    roi_c_info = debugger.roi_manager.get_roi('ROI_C')
                    if roi_c_info:
                        time.sleep(0.8)
                        cx = roi_c_info['x'] + roi_c_info['w'] // 2
                        cy = roi_c_info['y'] + roi_c_info['h'] // 2
                        mouse.click(cx, cy)
                        print(f"[MANUAL] Clicked ROI_C at: ({cx}, {cy})")
                    else:
                        print("[MANUAL] ROI_C not found! Please draw a third ROI.")
                elif roi_a_info:
                    cx = roi_a_info['x'] + roi_a_info['w'] // 2
                    cy = roi_a_info['y'] + roi_a_info['h'] // 2
                    mouse.click(cx, cy)
                    print(f"[MANUAL] Clicked ROI_A center at: ({cx}, {cy})")
                else:
                    print("ROI_A not found! Please draw ROI_A first.")

            elapsed = time.time() - frame_start
            sleep_time = max(0, (config.TICK_INTERVAL / 1000.0) - elapsed)
            time.sleep(sleep_time)

    finally:
        keyboard.release_all()
        screen_capture.release()
        cv2.destroyAllWindows()
        debugger.event_manager.export_log('ocr_event_log.txt')
        print("\nPOE2 OCR Event Detection System stopped.")

if __name__ == "__main__":
    main()