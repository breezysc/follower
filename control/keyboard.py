from pynput.keyboard import Key, Controller
import time

class KeyboardController:
    def __init__(self):
        self.keyboard = Controller()
        self.current_keys = set()
        self.last_press_time = {}
        
    def press_key(self, key):
        try:
            if key not in self.current_keys:
                # 处理空格键
                if key == 'space':
                    self.keyboard.press(Key.space)
                    self.current_keys.add('space')
                else:
                    self.keyboard.press(key)
                    self.current_keys.add(str(key))
                self.last_press_time[str(key)] = time.time()
        except Exception as e:
            print(f"Press key error: {e}")
    
    def release_key(self, key):
        try:
            key_str = str(key)
            if key_str in self.current_keys:
                # 处理空格键
                if key_str == 'space':
                    self.keyboard.release(Key.space)
                else:
                    self.keyboard.release(key)
                self.current_keys.remove(key_str)
        except Exception as e:
            print(f"Release key error: {e}")
    
    def release_all(self):
        keys_to_release = list(self.current_keys)
        for key in keys_to_release:
            self.release_key(key)
    
    def update_movement(self, dx, dy, space=False):
        keys_to_press = set()
        
        if dx < -5:
            keys_to_press.add('a')
        elif dx > 5:
            keys_to_press.add('d')
        
        if dy < -5:
            keys_to_press.add('w')
        elif dy > 5:
            keys_to_press.add('s')
        
        if space:
            keys_to_press.add('space')
        
        # 释放不在keys_to_press中的键
        keys_to_release = [key for key in self.current_keys if key not in keys_to_press]
        for key in keys_to_release:
            self.release_key(key)
        
        # 按下需要按的键
        for key in keys_to_press:
            self.press_key(key)
    
    def simulate_movement(self, dx, dy):
        keys_to_press = []
        
        if dx < -5:
            keys_to_press.append('A')
        elif dx > 5:
            keys_to_press.append('D')
        
        if dy < -5:
            keys_to_press.append('W')
        elif dy > 5:
            keys_to_press.append('S')
        
        return keys_to_press
    
    def type_text(self, text):
        for char in text:
            try:
                self.keyboard.press(char)
                self.keyboard.release(char)
            except:
                pass
            time.sleep(0.01)
