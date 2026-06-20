import cv2
import numpy as np
import os
import shutil

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    
    if shutil.which('tesseract') is None:
        common_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'D:\Program Files\Tesseract-OCR\tesseract.exe',
        ]
        for path in common_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"Found Tesseract at: {path}")
                break
        else:
            TESSERACT_AVAILABLE = False
            print("Warning: Tesseract not found in PATH or common locations. OCR will use fallback method.")
    else:
        print("Tesseract found in PATH")
        
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not installed. OCR will use fallback method.")

class OCREngine:
    def __init__(self, lang='eng'):
        self.lang = lang
        self.config = '--oem 3 --psm 6'
        self.confidence_threshold = 30
        
        self.last_result = ""
        self.last_confidence = 0.0
        self.last_image = None
        
    def preprocess_for_ocr(self, image):
        if image is None or image.size == 0:
            return None
            
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return binary
    
    def recognize(self, image):
        if image is None or image.size == 0:
            self.last_result = ""
            self.last_confidence = 0.0
            self.last_image = None
            return "", 0.0
        
        self.last_image = image.copy()
        
        if not TESSERACT_AVAILABLE:
            return self._fallback_ocr(image)
        
        processed = self.preprocess_for_ocr(image)
        if processed is None:
            return "", 0.0
        
        try:
            data = pytesseract.image_to_data(
                processed,
                config=self.config,
                output_type=pytesseract.Output.DICT
            )
            
            text_parts = []
            confidences = []
            
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                conf = float(data['conf'][i])
                if conf > self.confidence_threshold:
                    text = data['text'][i].strip()
                    if text:
                        text_parts.append(text)
                        confidences.append(conf)
            
            full_text = ' '.join(text_parts)
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            self.last_result = full_text
            self.last_confidence = avg_confidence
            
            return full_text, avg_confidence
            
        except Exception as e:
            print(f"OCR Error: {e}")
            return self._fallback_ocr(image)
    
    def _fallback_ocr(self, image):
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = ""
        confidences = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(contour)
            
            if area < 20 or area > 5000:
                continue
            
            digit = self._classify_digit(binary[y:y+h, x:x+w])
            if digit:
                result += digit
                confidences.append(70)
        
        if result:
            self.last_result = result
            self.last_confidence = 70.0
            return result, 70.0
        else:
            text = f"<Detected {len(contours)} shapes>"
            self.last_result = text
            self.last_confidence = 30.0
            return text, 30.0
    
    def _classify_digit(self, roi):
        h, w = roi.shape
        
        if h < 10 or w < 5:
            return ""
        
        white_pixels = cv2.countNonZero(roi)
        total_pixels = h * w
        ratio = white_pixels / total_pixels
        
        if ratio < 0.05 or ratio > 0.9:
            return ""
        
        horizontal_lines = []
        vertical_lines = []
        
        for y in range(h):
            row_sum = np.sum(roi[y, :])
            if row_sum > 0:
                horizontal_lines.append(y)
        
        for x in range(w):
            col_sum = np.sum(roi[:, x])
            if col_sum > 0:
                vertical_lines.append(x)
        
        aspect_ratio = w / float(h)
        
        if aspect_ratio > 1.5:
            if len(horizontal_lines) > len(vertical_lines):
                return "1"
            else:
                return "1"
        elif aspect_ratio < 0.7:
            return "1"
        else:
            if len(horizontal_lines) > 2:
                if ratio < 0.35:
                    return "0"
                elif ratio > 0.5:
                    return "8"
                else:
                    return "3"
            else:
                if len(vertical_lines) > len(horizontal_lines):
                    if ratio < 0.4:
                        return "1"
                    else:
                        return "7"
                else:
                    return "0"
    
    def recognize_numbers_only(self, image):
        if not TESSERACT_AVAILABLE:
            return self.recognize(image)
        
        processed = self.preprocess_for_ocr(image)
        if processed is None:
            return "", 0.0
        
        try:
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_data(processed, config=custom_config, output_type=pytesseract.Output.DICT)
            
            text_parts = []
            confidences = []
            
            n_boxes = len(text['text'])
            for i in range(n_boxes):
                conf = float(text['conf'][i])
                if conf > self.confidence_threshold:
                    char = text['text'][i].strip()
                    if char.isdigit():
                        text_parts.append(char)
                        confidences.append(conf)
            
            full_text = ''.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            self.last_result = full_text
            self.last_confidence = avg_confidence
            
            return full_text, avg_confidence
            
        except Exception as e:
            return self._fallback_ocr(image)
    
    def draw_confidence_visualization(self, image, text, confidence):
        display = image.copy()
        
        if display.shape[2] == 4:
            display = cv2.cvtColor(display, cv2.COLOR_BGRA2BGR)
        
        h, w = display.shape[:2]
        
        bar_height = 30
        bar_bg = np.zeros((bar_height, w, 3), dtype=np.uint8)
        
        fill_width = int((confidence / 100.0) * w)
        
        if confidence >= 80:
            bar_color = (0, 255, 0)
        elif confidence >= 60:
            bar_color = (0, 255, 255)
        elif confidence >= 40:
            bar_color = (0, 165, 255)
        else:
            bar_color = (0, 0, 255)
        
        cv2.rectangle(bar_bg, (0, 0), (fill_width, bar_height), bar_color, -1)
        
        label = f"Conf: {confidence:.1f}% | {text}"
        cv2.putText(bar_bg, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        result = np.vstack([display, bar_bg])
        
        return result
    
    def get_last_result(self):
        return self.last_result, self.last_confidence, self.last_image