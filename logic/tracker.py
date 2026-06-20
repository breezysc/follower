from enum import Enum
import time
from utils.math_utils import calculate_distance, calculate_direction, apply_dead_zone

class TrackerState(Enum):
    SEARCH = 'search'
    FOLLOW = 'follow'
    STOP = 'stop'
    AVOID = 'avoid'  # 新增：躲避状态

class Tracker:
    def __init__(self, center_x, center_y, dead_zone, follow_min, follow_max):
        self.center_x = center_x
        self.center_y = center_y
        self.dead_zone = dead_zone
        self.follow_min = follow_min
        self.follow_max = follow_max
        
        self.state = TrackerState.SEARCH
        self.last_position = None
        self.last_dx = 0
        self.last_dy = 0
        self.stability_counter = 0
        self.last_update_time = 0
        
        # 卡住检测相关
        self.distance_history = []  # 保存历史距离
        self.history_max_len = 10  # 保存最近10帧的距离
        self.stuck_threshold = 20  # 距离变化小于此值认为卡住
        self.stuck_frames = 0  # 连续卡住的帧数
        self.stuck_required = 5  # 连续5帧卡住才触发躲避
        
        # 躲避相关
        self.avoid_direction = 0  # 躲避方向：-1=左，0=无，1=右
        self.avoid_angle = 0  # 当前躲避角度偏移（度）
        self.avoid_press_space = False  # 是否按下空格
        self.avoid_start_time = 0  # 躲避开始时间
        self.avoid_phase = 0  # 躲避阶段：0=无，1=第一次躲避，2=第二次反向躲避
        self.max_avoid_angle = 45  # 最大躲避角度（度）
        
    def update(self, leader_position):
        current_time = time.time()
        
        if leader_position is None:
            self.state = TrackerState.SEARCH
            self.last_position = None
            return (0, 0), self.state, {}
        
        self.last_position = leader_position
        
        dx, dy = calculate_direction(leader_position[0], leader_position[1], self.center_x, self.center_y)
        dx, dy = apply_dead_zone(dx, dy, self.dead_zone)
        
        distance = calculate_distance(leader_position[0], leader_position[1], self.center_x, self.center_y)
        
        # 记录历史距离用于卡住检测
        self.distance_history.append(distance)
        if len(self.distance_history) > self.history_max_len:
            self.distance_history.pop(0)
        
        # 检查是否卡住
        avoid_info = self._check_stuck(distance)
        
        if distance < self.follow_min:
            self.state = TrackerState.STOP
            dx, dy = 0, 0
        else:
            # 如果需要躲避，应用躲避偏移
            if avoid_info['needs_avoid']:
                dx, dy = self._apply_avoid(dx, dy, avoid_info)
                self.state = TrackerState.AVOID
            else:
                self.state = TrackerState.FOLLOW
            
        direction_changed = abs(dx - self.last_dx) > self.dead_zone or abs(dy - self.last_dy) > self.dead_zone
        
        if direction_changed:
            self.stability_counter = 0
        else:
            self.stability_counter += 1
        
        self.last_dx = dx
        self.last_dy = dy
        self.last_update_time = current_time
        
        return (dx, dy), self.state, avoid_info
    
    def _check_stuck(self, current_distance):
        """检查是否卡住"""
        avoid_info = {
            'needs_avoid': False,
            'avoid_direction': 0,
            'press_space': False,
            'phase': 0
        }
        
        if len(self.distance_history) < 3:
            return avoid_info
        
        # 计算距离变化
        distance_change = self.distance_history[-1] - self.distance_history[0]
        
        # 如果距离没有变近甚至变远，认为卡住
        if distance_change >= -self.stuck_threshold:
            self.stuck_frames += 1
        else:
            # 距离在接近，重置卡住计数
            if self.stuck_frames > 0 and distance_change < -self.stuck_threshold:
                self.stuck_frames = 0
                self.avoid_phase = 0
                self.avoid_direction = 0
        
        # 连续多帧卡住，需要躲避
        if self.stuck_frames >= self.stuck_required:
            avoid_info['needs_avoid'] = True
            
            # 第一阶段躲避（刚开始卡住）
            if self.avoid_phase == 0:
                self.avoid_phase = 1
                self.avoid_direction = -1 if self.last_dx > 0 else 1  # 根据移动方向决定躲避方向
                self.avoid_start_time = time.time()
                self.avoid_angle = 20  # 初始躲避角度
            # 第二阶段躲避（2秒后反向躲避）
            elif self.avoid_phase == 1:
                elapsed = time.time() - self.avoid_start_time
                if elapsed >= 2.0:
                    self.avoid_phase = 2
                    self.avoid_direction = -self.avoid_direction  # 反向
                    self.avoid_start_time = time.time()
                    self.avoid_angle = min(self.avoid_angle + 10, self.max_avoid_angle)  # 增加角度
            # 第三阶段躲避（再2秒后继续反向）
            elif self.avoid_phase == 2:
                elapsed = time.time() - self.avoid_start_time
                if elapsed >= 2.0:
                    self.avoid_phase = 3
                    self.avoid_direction = -self.avoid_direction  # 继续反向
                    self.avoid_start_time = time.time()
                    self.avoid_angle = min(self.avoid_angle + 15, self.max_avoid_angle)
            
            avoid_info['avoid_direction'] = self.avoid_direction
            avoid_info['press_space'] = True  # 躲避时按下空格
            avoid_info['phase'] = self.avoid_phase
            avoid_info['angle'] = self.avoid_angle
        
        return avoid_info
    
    def _apply_avoid(self, dx, dy, avoid_info):
        """应用躲避偏移到移动方向"""
        if avoid_info['avoid_direction'] == 0:
            return dx, dy
        
        # 计算躲避角度（弧度）
        import math
        avoid_rad = math.radians(avoid_info['angle'] * avoid_info['avoid_direction'])
        
        # 原始角度
        original_angle = math.atan2(dy, dx)
        
        # 应用躲避偏移
        new_angle = original_angle + avoid_rad
        
        # 转换为dx, dy
        magnitude = math.sqrt(dx*dx + dy*dy)
        new_dx = magnitude * math.cos(new_angle)
        new_dy = magnitude * math.sin(new_angle)
        
        return int(new_dx), int(new_dy)
    
    def get_state(self):
        return self.state