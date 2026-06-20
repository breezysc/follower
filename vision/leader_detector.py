import cv2
import numpy as np
import os

class LeaderDetector:
    def __init__(self, lower_bound, upper_bound, template_path=None):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mask = None
        self.contours = []
        self.candidates = []
        self.selected_target = None
        self.template = None
        self.template_path = template_path
        self.match_threshold = 0.7
        
        if template_path and os.path.exists(template_path):
            self.load_template(template_path)
    
    def load_template(self, path, silent=False):
        self.template = cv2.imread(path, cv2.IMREAD_COLOR)
        if self.template is not None:
            if not silent:
                print(f"Loaded template from {path}")
        else:
            if not silent:
                print(f"Failed to load template from {path}")
        
    def detect(self, frame, roi_region=None):
        if frame is None:
            self.mask = None
            self.contours = []
            self.candidates = []
            self.selected_target = None
            return None
        
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        if self.template is not None:
            template_result = self._template_match(frame_bgr, roi_region)
            if template_result is not None:
                return template_result
        
        if roi_region is not None:
            x, y, w, h = roi_region
            frame_bgr = frame_bgr[y:y+h, x:x+w]
            self.roi_offset = (x, y)
        else:
            self.roi_offset = (0, 0)
        
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)
        
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        self.mask = mask
        
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = contours
        
        self.candidates = self._analyze_candidates(contours)
        
        self.selected_target = self._select_best_target(self.candidates)
        
        if self.selected_target is not None:
            cx, cy = self.selected_target['center']
            cx += self.roi_offset[0]
            cy += self.roi_offset[1]
            return (cx, cy)
        
        return None
    
    def _analyze_candidates(self, contours):
        candidates = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area < 5 or area > 5000:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            moments = cv2.moments(contour)
            if moments['m00'] > 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
            else:
                continue
            
            aspect_ratio = w / h if h > 0 else 0
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            rect = cv2.minAreaRect(contour)
            rect_area = rect[1][0] * rect[1][1]
            extent = area / rect_area if rect_area > 0 else 0
            
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            
            candidate = {
                'id': i + 1,
                'center': (cx, cy),
                'bbox': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'solidity': solidity,
                'extent': extent,
                'circularity': circularity,
                'perimeter': perimeter,
                'contour': contour
            }
            candidates.append(candidate)
        
        return candidates
    
    def _select_best_target(self, candidates):
        if not candidates:
            return None
        
        scores = []
        
        for candidate in candidates:
            score = 0
            
            area = candidate['area']
            if 20 < area < 500:
                score += 30
            elif 10 < area <= 20 or 500 <= area < 1000:
                score += 15
            
            ar = candidate['aspect_ratio']
            if 0.5 <= ar <= 2.0:
                score += 25
            elif 0.3 <= ar < 0.5 or 2.0 < ar <= 3.0:
                score += 10
            
            solidity = candidate['solidity']
            if solidity > 0.8:
                score += 20
            elif solidity > 0.6:
                score += 10
            
            circularity = candidate['circularity']
            if 0.3 < circularity < 0.8:
                score += 15
            
            extent = candidate['extent']
            if extent > 0.5:
                score += 10
            
            scores.append(score)
        
        max_score_idx = np.argmax(scores)
        
        if scores[max_score_idx] > 0:
            return candidates[max_score_idx]
        
        return candidates[0] if candidates else None
    
    def get_detection_metrics(self):
        if self.mask is None:
            return {
                'mask_pixel_ratio': 0,
                'contour_count': 0,
                'largest_contour_area': 0,
                'confidence': 0.0,
                'candidate_count': 0,
                'selected_id': None,
                'selected_area': 0,
                'selected_aspect_ratio': 0
            }
        
        total_pixels = self.mask.shape[0] * self.mask.shape[1]
        mask_pixels = np.sum(self.mask > 0)
        mask_ratio = mask_pixels / total_pixels if total_pixels > 0 else 0
        
        contour_count = len(self.contours)
        largest_area = max((cv2.contourArea(c) for c in self.contours), default=0)
        
        confidence = 0.0
        if self.selected_target:
            confidence = min(1.0, self.selected_target['area'] / 100)
        
        selected_id = self.selected_target['id'] if self.selected_target else None
        selected_area = self.selected_target['area'] if self.selected_target else 0
        selected_ar = self.selected_target['aspect_ratio'] if self.selected_target else 0
        
        return {
            'mask_pixel_ratio': mask_ratio,
            'contour_count': contour_count,
            'largest_contour_area': largest_area,
            'confidence': confidence,
            'candidate_count': len(self.candidates),
            'selected_id': selected_id,
            'selected_area': selected_area,
            'selected_aspect_ratio': selected_ar
        }
    
    def get_candidates(self):
        return self.candidates
    
    def draw_debug(self, frame, leader_pos, center_x, center_y, cfg):
        debug_frame = frame.copy()
        
        if frame.shape[2] == 4:
            debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_BGRA2BGR)
        
        if cfg.SHOW_ROI:
            roi_half = cfg.ROI_SIZE // 2
            roi_x = max(0, center_x - roi_half)
            roi_y = max(0, center_y - roi_half)
            roi_w = min(frame.shape[1] - roi_x, cfg.ROI_SIZE)
            roi_h = min(frame.shape[0] - roi_y, cfg.ROI_SIZE)
            cv2.rectangle(debug_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), cfg.COLOR_ROI, 2)
            cv2.putText(debug_frame, 'ROI', (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, cfg.HUD_FONT_SIZE, cfg.COLOR_ROI, cfg.HUD_LINE_THICKNESS)
        
        if cfg.SHOW_CENTER:
            cv2.circle(debug_frame, (center_x, center_y), 8, cfg.COLOR_CENTER, -1)
            cv2.circle(debug_frame, (center_x, center_y), 15, cfg.COLOR_CENTER, 2)
            cv2.putText(debug_frame, 'CENTER', (center_x + 20, center_y), cv2.FONT_HERSHEY_SIMPLEX, cfg.HUD_FONT_SIZE, cfg.COLOR_CENTER, cfg.HUD_LINE_THICKNESS)
        
        for candidate in self.candidates:
            cx, cy = candidate['center']
            x, y, w, h = candidate['bbox']
            area = candidate['area']
            ar = candidate['aspect_ratio']
            
            is_selected = (self.selected_target is not None and 
                          candidate['id'] == self.selected_target['id'])
            
            color = cfg.COLOR_LEADER if is_selected else cfg.COLOR_CONTOUR
            thickness = 3 if is_selected else 1
            
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, thickness)
            
            label = f"[{candidate['id']}]"
            cv2.putText(debug_frame, label, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(debug_frame, f"area={area}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            cv2.putText(debug_frame, f"ar={ar:.1f}", (x + w, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            
            cv2.circle(debug_frame, (cx, cy), 4, color, -1)
        
        if leader_pos:
            cv2.circle(debug_frame, leader_pos, 10, cfg.COLOR_LEADER, -1)
            cv2.circle(debug_frame, leader_pos, 18, cfg.COLOR_LEADER, 2)
            cv2.putText(debug_frame, f'LEADER ({leader_pos[0]},{leader_pos[1]})', 
                       (leader_pos[0] + 20, leader_pos[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, cfg.HUD_FONT_SIZE, cfg.COLOR_LEADER, cfg.HUD_LINE_THICKNESS)
            
            if cfg.SHOW_DIRECTION_LINE:
                cv2.arrowedLine(debug_frame, (center_x, center_y), leader_pos, cfg.COLOR_DIRECTION, 3, tipLength=0.1)
        
        return debug_frame
    
    def get_roi_region(self, center_x, center_y, roi_size):
        roi_half = roi_size // 2
        x = max(0, center_x - roi_half)
        y = max(0, center_y - roi_half)
        w = roi_size
        h = roi_size
        
        return (x, y, w, h)
    
    def get_mask(self):
        return self.mask
    
    def get_contours(self):
        return self.contours
    
    def _template_match(self, frame_bgr, roi_region=None):
        if self.template is None:
            return None
        
        search_frame = frame_bgr
        offset = (0, 0)
        
        if roi_region is not None:
            x, y, w, h = roi_region
            search_frame = frame_bgr[y:y+h, x:x+w]
            offset = (x, y)
        
        template_h, template_w = self.template.shape[:2]
        frame_h, frame_w = search_frame.shape[:2]
        
        if template_h > frame_h or template_w > frame_w:
            return None
        
        result = cv2.matchTemplate(search_frame, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= self.match_threshold:
            center_x = max_loc[0] + template_w // 2
            center_y = max_loc[1] + template_h // 2
            center_x += offset[0]
            center_y += offset[1]
            
            aspect_ratio = template_w / template_h if template_h > 0 else 1.0
            
            self.selected_target = {
                'id': 0,
                'center': (center_x - offset[0], center_y - offset[1]),
                'bbox': (max_loc[0], max_loc[1], template_w, template_h),
                'area': template_w * template_h,
                'aspect_ratio': aspect_ratio,
                'confidence': max_val,
                'solidity': 1.0,
                'circularity': 1.0,
                'extent': 1.0
            }
            self.candidates = [self.selected_target]
            
            return (center_x, center_y)
        
        return None