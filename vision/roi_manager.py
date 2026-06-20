import numpy as np
import cv2
import os
import json

class ROIManager:
    def __init__(self, config_path=None):
        self.rois = {}
        self.config_path = config_path or 'roi_config.txt'
        self.current_width = 1920
        self.current_height = 1080
        self.load_config()
        
        self.current_mode = 'select'
        self.temp_roi = None
        self.selected_roi = None
        
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
    def set_display_params(self, scale, offset_x=0, offset_y=0):
        self.scale = scale
        self.offset_x = offset_x
        self.offset_y = offset_y
    
    def set_current_size(self, width, height):
        self.current_width = width
        self.current_height = height
        
    def screen_to_display(self, x, y):
        return int(x * self.scale + self.offset_x), int(y * self.scale + self.offset_y)
    
    def display_to_screen(self, x, y):
        return int((x - self.offset_x) / self.scale), int((y - self.offset_y) / self.scale)
        
    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    self.rois = data.get('rois', {})
                    
                    if 'base_width' in data:
                        self.current_width = data['base_width']
                    if 'base_height' in data:
                        self.current_height = data['base_height']
                        
                print(f"Loaded {len(self.rois)} ROIs (base: {self.current_width}x{self.current_height})")
            except Exception as e:
                print(f"Failed to load ROI config (JSON): {e}")
                print("Trying to load old format...")
                self._load_old_format()
    
    def _load_old_format(self):
        try:
            with open(self.config_path, 'r') as f:
                lines = f.readlines()
                i = 0
                while i < len(lines):
                    name = lines[i].strip()
                    if name.startswith('[') and name.endswith(']'):
                        name = name[1:-1]
                        i += 1
                        if i < len(lines):
                            coords = lines[i].strip()
                            parts = coords.split(',')
                            if len(parts) == 4:
                                x, y, w, h = map(int, parts)
                                # 转换为比例坐标（基于基准尺寸1920x1080）
                                self.rois[name] = {
                                    'x_ratio': x / 1920,
                                    'y_ratio': y / 1080,
                                    'width_ratio': w / 1920,
                                    'height_ratio': h / 1080,
                                    'enabled': True
                                }
                    i += 1
            print(f"Loaded {len(self.rois)} ROIs from old format")
        except Exception as e:
            print(f"Failed to load old ROI config: {e}")
    
    def save_config(self):
        try:
            data = {
                'base_width': self.current_width,
                'base_height': self.current_height,
                'rois': self.rois
            }
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(self.rois)} ROIs")
        except Exception as e:
            print(f"Failed to save ROI config: {e}")
    
    def get_absolute_roi(self, name):
        roi = self.rois.get(name)
        if roi is None:
            return None
        
        x_ratio = roi.get('x_ratio', 0)
        y_ratio = roi.get('y_ratio', 0)
        w_ratio = roi.get('width_ratio', 0)
        h_ratio = roi.get('height_ratio', 0)
        
        x = int(x_ratio * self.current_width)
        y = int(y_ratio * self.current_height)
        w = int(w_ratio * self.current_width)
        h = int(h_ratio * self.current_height)
        
        return {'x': x, 'y': y, 'w': w, 'h': h, 'enabled': roi.get('enabled', True)}
    
    def _save_ratio_roi(self, name, x, y, w, h):
        if w <= 0 or h <= 0:
            return
            
        x_ratio = x / self.current_width if self.current_width > 0 else 0
        y_ratio = y / self.current_height if self.current_height > 0 else 0
        w_ratio = w / self.current_width if self.current_width > 0 else 0
        h_ratio = h / self.current_height if self.current_height > 0 else 0
        
        self.rois[name] = {
            'x_ratio': x_ratio,
            'y_ratio': y_ratio,
            'width_ratio': w_ratio,
            'height_ratio': h_ratio,
            'enabled': True
        }
    
    def mouse_callback(self, event, x, y, flags, param):
        sx, sy = self.display_to_screen(x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_mode == 'select':
                for name, roi_data in self.rois.items():
                    roi_abs = self.get_absolute_roi(name)
                    if roi_abs:
                        rx, ry, rw, rh = roi_abs['x'], roi_abs['y'], roi_abs['w'], roi_abs['h']
                        if rx <= sx <= rx + rw and ry <= sy <= ry + rh:
                            self.selected_roi = name
                            self.temp_roi = None
                            return
                self.selected_roi = None
                
            elif self.current_mode == 'draw':
                self.temp_roi = {'x': sx, 'y': sy, 'w': 0, 'h': 0}
                
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.temp_roi is not None:
                self.temp_roi['w'] = sx - self.temp_roi['x']
                self.temp_roi['h'] = sy - self.temp_roi['y']
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.temp_roi is not None:
                x, y = self.temp_roi['x'], self.temp_roi['y']
                w, h = self.temp_roi['w'], self.temp_roi['h']
                
                if w < 0:
                    x, w = x + w, -w
                if h < 0:
                    y, h = y + h, -h
                
                if w > 10 and h > 10:
                    if self.selected_roi:
                        self._save_ratio_roi(self.selected_roi, x, y, w, h)
                        print(f"Updated {self.selected_roi} at ratio ({x/self.current_width:.3f}, {y/self.current_height:.3f})")
                    else:
                        target_names = ['ROI_A', 'ROI_B', 'ROI_C']
                        for name in target_names:
                            if name not in self.rois:
                                self._save_ratio_roi(name, x, y, w, h)
                                print(f"Created {name} at ratio ({x/self.current_width:.3f}, {y/self.current_height:.3f})")
                                break
                        else:
                            name = f"ROI_{len(self.rois)}"
                            self._save_ratio_roi(name, x, y, w, h)
                            print(f"Created {name} at ratio ({x/self.current_width:.3f}, {y/self.current_height:.3f})")
                self.temp_roi = None
    
    def set_mode(self, mode):
        self.current_mode = mode
        self.temp_roi = None
        
    def get_roi(self, name):
        return self.get_absolute_roi(name)
    
    def get_all_rois(self):
        result = {}
        for name in self.rois.keys():
            result[name] = self.get_absolute_roi(name)
        return result
    
    def enable_roi(self, name, enabled=True):
        if name in self.rois:
            self.rois[name]['enabled'] = enabled
            
    def delete_roi(self, name):
        if name in self.rois:
            del self.rois[name]
            
    def draw_rois(self, frame):
        display = frame.copy()
        
        colors = {
            'ROI_A': (0, 255, 0),
            'ROI_B': (255, 0, 0),
            'ROI_C': (0, 255, 255),
            'ROI_D': (255, 0, 255)
        }
        
        for i, (name, roi_data) in enumerate(self.rois.items()):
            if not roi_data.get('enabled', True):
                continue
            
            roi_abs = self.get_absolute_roi(name)
            if not roi_abs:
                continue
                
            x, y = self.screen_to_display(roi_abs['x'], roi_abs['y'])
            w = int(roi_abs['w'] * self.scale)
            h = int(roi_abs['h'] * self.scale)
            
            color = colors.get(name, (0, 128, 255))
            
            if name == self.selected_roi:
                color = (0, 255, 255)
                thickness = 3
            else:
                thickness = 2
                
            cv2.rectangle(display, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(display, name, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            x_ratio = roi_data.get('x_ratio', 0)
            y_ratio = roi_data.get('y_ratio', 0)
            label_x = x + w + 5
            label_y = y + 20
            cv2.putText(display, f"({x_ratio:.2f},{y_ratio:.2f})", (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
        if self.temp_roi:
            x, y = self.screen_to_display(self.temp_roi['x'], self.temp_roi['y'])
            w = int(self.temp_roi['w'] * self.scale)
            h = int(self.temp_roi['h'] * self.scale)
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, "Drawing...", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        mode_text = f"Mode: {self.current_mode.upper()} | Scale: {self.scale:.2f} | Size: {self.current_width}x{self.current_height}"
        cv2.putText(display, mode_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        instructions = "Keys: 1=Select 2=Draw 3=Delete 4=Save 5=Clear ESC=Exit"
        cv2.putText(display, instructions, (10, display.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return display
    
    def extract_roi(self, frame, name):
        roi_abs = self.get_absolute_roi(name)
        if roi_abs is None:
            return None
        
        x, y, w, h = roi_abs['x'], roi_abs['y'], roi_abs['w'], roi_abs['h']
        
        if x < 0: x = 0
        if y < 0: y = 0
        if x + w > frame.shape[1]: w = frame.shape[1] - x
        if y + h > frame.shape[0]: h = frame.shape[0] - y
        
        if w <= 0 or h <= 0:
            return None
            
        return frame[y:y+h, x:x+w]