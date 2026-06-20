import re
from datetime import datetime

class RuleEngine:
    def __init__(self):
        self.rules = {}
        self.history = []
        self.match_count = 0
        
    def add_rule(self, name, condition_type, condition_value):
        self.rules[name] = {
            'type': condition_type,
            'value': condition_value,
            'enabled': True
        }
        print(f"Added rule: {name} = {condition_type}({condition_value})")
        
    def remove_rule(self, name):
        if name in self.rules:
            del self.rules[name]
            print(f"Removed rule: {name}")
            
    def enable_rule(self, name, enabled=True):
        if name in self.rules:
            self.rules[name]['enabled'] = enabled
            
    def clear_rules(self):
        self.rules.clear()
        
    def evaluate(self, ocr_text):
        results = {}
        matched_rules = []
        
        for name, rule in self.rules.items():
            if not rule['enabled']:
                results[name] = {'match': False, 'reason': 'disabled'}
                continue
            
            rule_type = rule['type']
            rule_value = rule['value']
            
            match = False
            reason = ""
            
            if rule_type == 'contains':
                if rule_value.lower() in ocr_text.lower():
                    match = True
                    reason = f"Found '{rule_value}'"
                else:
                    reason = f"'{rule_value}' not in text"
                    
            elif rule_type == 'starts_with':
                if ocr_text.lower().startswith(rule_value.lower()):
                    match = True
                    reason = f"Text starts with '{rule_value}'"
                else:
                    reason = f"Text doesn't start with '{rule_value}'"
                    
            elif rule_type == 'ends_with':
                if ocr_text.lower().endswith(rule_value.lower()):
                    match = True
                    reason = f"Text ends with '{rule_value}'"
                else:
                    reason = f"Text doesn't end with '{rule_value}'"
                    
            elif rule_type == 'regex':
                try:
                    pattern = re.compile(rule_value, re.IGNORECASE)
                    if pattern.search(ocr_text):
                        match = True
                        reason = f"Regex '{rule_value}' matched"
                    else:
                        reason = f"Regex '{rule_value}' not matched"
                except re.error as e:
                    reason = f"Invalid regex: {e}"
                    
            elif rule_type == 'equals':
                if ocr_text.lower() == rule_value.lower():
                    match = True
                    reason = f"Exact match '{rule_value}'"
                else:
                    reason = f"Not equal to '{rule_value}'"
                    
            elif rule_type == 'contains_any':
                keywords = [k.strip() for k in rule_value.split(',')]
                found = [k for k in keywords if k.lower() in ocr_text.lower()]
                if found:
                    match = True
                    reason = f"Found keywords: {', '.join(found)}"
                else:
                    reason = f"None of {keywords} found"
                    
            elif rule_type == 'numeric_compare':
                try:
                    numbers = re.findall(r'\d+', ocr_text)
                    target_num = int(rule_value)
                    for num_str in numbers:
                        num = int(num_str)
                        if num == target_num:
                            match = True
                            reason = f"Found number {target_num}"
                            break
                    if not match:
                        reason = f"Number {target_num} not found"
                except:
                    reason = "Invalid number comparison"
                    
            elif rule_type == 'numeric_greater':
                try:
                    numbers = re.findall(r'\d+', ocr_text)
                    threshold = float(rule_value)
                    for num_str in numbers:
                        num = float(num_str)
                        if num > threshold:
                            match = True
                            reason = f"Found number {num} > {threshold}"
                            break
                    if not match:
                        reason = f"No number > {threshold}"
                except:
                    reason = "Invalid comparison"
                    
            elif rule_type == 'numeric_less':
                try:
                    numbers = re.findall(r'\d+', ocr_text)
                    threshold = float(rule_value)
                    for num_str in numbers:
                        num = float(num_str)
                        if num < threshold:
                            match = True
                            reason = f"Found number {num} < {threshold}"
                            break
                    if not match:
                        reason = f"No number < {threshold}"
                except:
                    reason = "Invalid comparison"
            
            else:
                reason = f"Unknown rule type: {rule_type}"
            
            results[name] = {
                'match': match,
                'reason': reason,
                'type': rule_type,
                'value': rule_value
            }
            
            if match:
                matched_rules.append(name)
        
        all_matched = len(matched_rules) > 0 and all(
            results[r]['match'] for r in matched_rules
        )
        
        if all_matched:
            self.match_count += 1
            self.history.append({
                'timestamp': datetime.now(),
                'ocr_text': ocr_text,
                'matched_rules': matched_rules.copy(),
                'results': results.copy()
            })
        
        return {
            'all_matched': all_matched,
            'matched_rules': matched_rules,
            'results': results
        }
    
    def get_stats(self):
        return {
            'total_rules': len(self.rules),
            'match_count': self.match_count,
            'history_length': len(self.history)
        }
    
    def get_recent_matches(self, count=10):
        return self.history[-count:]
    
    def clear_history(self):
        self.history.clear()
        self.match_count = 0