from datetime import datetime
import os

class EventManager:
    def __init__(self):
        self.events = []
        self.match_count = 0
        self.last_match_time = None
        self.last_match_text = ""
        self.event_callbacks = {}
        
    def add_event(self, event_type, text, details=None):
        event = {
            'timestamp': datetime.now(),
            'type': event_type,
            'text': text,
            'details': details or {}
        }
        self.events.append(event)
        
        if len(self.events) > 100:
            self.events.pop(0)
            
        return event
    
    def on_match(self, ocr_text, matched_rules):
        self.match_count += 1
        self.last_match_time = datetime.now()
        self.last_match_text = ocr_text
        
        event = self.add_event('MATCH', ocr_text, {
            'matched_rules': matched_rules
        })
        
        if 'match' in self.event_callbacks:
            self.event_callbacks['match'](ocr_text, matched_rules)
            
        return event
    
    def on_no_match(self, ocr_text):
        self.add_event('NO_MATCH', ocr_text)
        
    def on_detection(self, leader_pos, confidence):
        self.add_event('DETECTION', f"Leader at {leader_pos}", {
            'position': leader_pos,
            'confidence': confidence
        })
        
    def on_error(self, error_msg):
        self.add_event('ERROR', error_msg)
        
    def register_callback(self, event_type, callback):
        self.event_callbacks[event_type] = callback
        
    def get_status(self):
        return {
            'match_count': self.match_count,
            'last_match_time': self.last_match_time,
            'last_match_text': self.last_match_text,
            'total_events': len(self.events)
        }
    
    def get_recent_events(self, count=10):
        return self.events[-count:]
    
    def clear_events(self):
        self.events.clear()
        self.match_count = 0
        self.last_match_time = None
        self.last_match_text = ""
        
    def export_log(self, filepath):
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("OCR Event Detection Log\n")
                f.write(f"Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n\n")
                
                f.write(f"Total Matches: {self.match_count}\n")
                if self.last_match_time:
                    f.write(f"Last Match: {self.last_match_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n" + "="*60 + "\n\n")
                
                f.write("Event Log:\n")
                f.write("-"*60 + "\n")
                
                for event in self.events:
                    time_str = event['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    f.write(f"[{time_str}] {event['type']}: {event['text']}\n")
                    if event['details']:
                        for key, value in event['details'].items():
                            f.write(f"    {key}: {value}\n")
                
            print(f"Log exported to: {filepath}")
            return True
        except Exception as e:
            print(f"Failed to export log: {e}")
            return False