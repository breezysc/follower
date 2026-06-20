import mss
import numpy as np

class ScreenCapture:
    def __init__(self, region=None):
        self.sct = mss.mss()
        self.region = region
    
    def capture(self):
        if self.region:
            monitor = {
                'top': self.region['top'],
                'left': self.region['left'],
                'width': self.region['width'],
                'height': self.region['height']
            }
        else:
            monitor = self.sct.monitors[1]
        
        img = self.sct.grab(monitor)
        return np.array(img)
    
    def release(self):
        self.sct.close()