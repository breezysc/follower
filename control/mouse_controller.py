import ctypes
from ctypes import wintypes
import time

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010

class MouseController:
    def __init__(self):
        self.enabled = True
        self.last_click_pos = None
        self.last_click_time = None
        
    def click(self, x, y, button='left'):
        if not self.enabled:
            return False
            
        try:
            old_pos = self.get_cursor_pos()
            
            self.set_cursor_pos(x, y)
            time.sleep(0.01)
            
            if button == 'left':
                self.mouse_event(MOUSEEVENTF_LEFTDOWN, x, y)
                time.sleep(0.05)
                self.mouse_event(MOUSEEVENTF_LEFTUP, x, y)
            elif button == 'right':
                self.mouse_event(MOUSEEVENTF_RIGHTDOWN, x, y)
                time.sleep(0.05)
                self.mouse_event(MOUSEEVENTF_RIGHTUP, x, y)
                
            time.sleep(0.01)
            self.set_cursor_pos(*old_pos)
            
            self.last_click_pos = (x, y)
            self.last_click_time = time.time()
            
            return True
            
        except Exception as e:
            print(f"Click error: {e}")
            return False
    
    def click_relative(self, base_x, base_y, offset_x, offset_y, button='left'):
        return self.click(base_x + offset_x, base_y + offset_y, button)
    
    def double_click(self, x, y, button='left'):
        if not self.enabled:
            return False
            
        self.click(x, y, button)
        time.sleep(0.1)
        self.click(x, y, button)
        return True
    
    def right_click(self, x, y):
        return self.click(x, y, 'right')
    
    def get_cursor_pos(self):
        try:
            point = wintypes.POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
            return (point.x, point.y)
        except:
            return (0, 0)
    
    def set_cursor_pos(self, x, y):
        try:
            ctypes.windll.user32.SetCursorPos(x, y)
            return True
        except:
            return False
    
    def mouse_event(self, event, x, y):
        try:
            x_input = int(x * 65535 / ctypes.windll.user32.GetSystemMetrics(0))
            y_input = int(y * 65535 / ctypes.windll.user32.GetSystemMetrics(1))
            
            extra = ctypes.c_ulong(0)
            ii_ = wintypes.INPUT_union()
            ii_.mi = wintypes.MOUSEINPUT(x_input, y_input, 0, event, 0, None)
            x = wintypes.INPUT(ctypes.c_ulong(0), ii_)
            ctypes.windll.user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))
        except:
            try:
                ctypes.windll.user32.mouse_event(event, x, y, 0, 0)
            except:
                pass
    
    def enable(self):
        self.enabled = True
        
    def disable(self):
        self.enabled = False
        
    def is_enabled(self):
        return self.enabled