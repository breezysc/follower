import ctypes
from ctypes import wintypes
import numpy as np
import mss

user32 = ctypes.windll.user32

class WindowCapture:
    def __init__(self, window_name=None):
        self.hwnd = None
        self.window_name = window_name

        if window_name:
            self.find_window(window_name)
    
    def find_window(self, name):
        """查找包含指定名称的窗口"""
        name_lower = name.lower()
        
        def enum_callback(hwnd, lparam):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buffer = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buffer, length + 1)
                    title = buffer.value
                    if name_lower in title.lower():
                        self.hwnd = hwnd
                        return False
            return True
        
        EnumWindowsProc = ctypes.WINFUNCTYPE(
            ctypes.c_bool,
            wintypes.HWND,
            wintypes.LPARAM
        )
        user32.EnumWindows(EnumWindowsProc(enum_callback), 0)
        
        if self.hwnd:
            print(f"Found window: {self.get_window_title()}")
            return True
        else:
            print(f"Window not found: {name}")
            return False
    
    def get_window_title(self):
        """获取窗口标题"""
        if self.hwnd:
            length = user32.GetWindowTextLengthW(self.hwnd)
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(self.hwnd, buffer, length + 1)
            return buffer.value
        return None
    
    def get_window_rect(self):
        """获取窗口内容区域的位置和大小"""
        if self.hwnd:
            # 如果窗口不可见（例如最小化），直接返回 None，避免得到 0x0 尺寸
            if not user32.IsWindowVisible(self.hwnd):
                return None

            # 获取整个窗口的位置（包括标题栏和边框）
            window_rect = wintypes.RECT()
            user32.GetWindowRect(self.hwnd, ctypes.byref(window_rect))

            # 获取客户区的大小（不包括标题栏和边框）
            client_rect = wintypes.RECT()
            user32.GetClientRect(self.hwnd, ctypes.byref(client_rect))

            # 获取客户区相对于窗口左上角的偏移
            # 需要将客户区坐标转换为屏幕坐标
            client_tl = wintypes.POINT()
            client_tr = wintypes.POINT()
            client_tl.x = 0
            client_tl.y = 0
            client_tr.x = client_rect.right
            client_tr.y = client_rect.bottom
            user32.ClientToScreen(self.hwnd, ctypes.byref(client_tl))
            user32.ClientToScreen(self.hwnd, ctypes.byref(client_tr))

            width = client_tr.x - client_tl.x
            height = client_tr.y - client_tl.y
            if width <= 0 or height <= 0:
                return None

            return {
                'left': client_tl.x,  # 客户区左上角X
                'top': client_tl.y,  # 客户区左上角Y（不包括标题栏）
                'width': width,
                'height': height
            }
        return None

    def capture(self):
        """截取窗口内容"""
        if self.hwnd:
            rect = self.get_window_rect()
            if rect and rect['width'] > 0 and rect['height'] > 0:
                monitor = {
                    'top': rect['top'],
                    'left': rect['left'],
                    'width': rect['width'],
                    'height': rect['height']
                }
                # 每次都创建新的 mss 实例，避免内部线程句柄在重用时损坏
                try:
                    with mss.mss() as sct:
                        img = sct.grab(monitor)
                        return np.array(img)
                except Exception as e:
                    print(f"[CAPTURE ERROR] {e}")
                    return None
        return None

    def release(self):
        """释放资源"""
        # mss 实例现在在 capture() 内部 with 块中创建并自动释放
        pass
