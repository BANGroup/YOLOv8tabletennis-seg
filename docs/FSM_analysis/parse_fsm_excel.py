#!/usr/bin/env python3
"""
Парсинг Excel таблицы с FSM логикой в JSON
"""

import openpyxl
import json

def parse_fsm_excel(excel_file):
    """Читает Excel и преобразует в JSON"""
    
    wb = openpyxl.load_workbook(excel_file)
    ws = wb['FSM']
    
    fsm_states = []
    
    # Читаем с 2й строки (1я - заголовок)
    for row_idx in range(2, ws.max_row + 1):
        state_name = ws[f'A{row_idx}'].value
        bounce_s = ws[f'C{row_idx}'].value
        net_touch = ws[f'D{row_idx}'].value
        bounce_r = ws[f'E{row_idx}'].value
        timeout = ws[f'F{row_idx}'].value
        score_s = ws[f'H{row_idx}'].value
        score_r = ws[f'I{row_idx}'].value
        comment = ws[f'J{row_idx}'].value
        
        if state_name:
            state = {
                'state': str(state_name),
                'conditions': {
                    'bounce_S': bounce_s,
                    'net_touch': net_touch,
                    'bounce_R': bounce_r,
                    'timeout': str(timeout) if timeout else None
                },
                'actions': {
                    'score_S': score_s,
                    'score_R': score_r
                },
                'comment': str(comment) if comment else None
            }
            fsm_states.append(state)
    
    return fsm_states

if __name__ == "__main__":
    states = parse_fsm_excel("tt_fsm_rules_rev_1.xlsx")
    
    print("\n=== FSM ЛОГИКА ИЗ EXCEL ===\n")
    
    for state in states:
        print(f"Состояние: {state['state']}")
        print(f"  Условия:")
        print(f"    - Отскок S (bounce_S): {state['conditions']['bounce_S']}")
        print(f"    - Сетка (net_touch): {state['conditions']['net_touch']}")
        print(f"    - Отскок R (bounce_R): {state['conditions']['bounce_R']}")
        print(f"    - Таймаут/Смена направления: {state['conditions']['timeout']}")
        print(f"  Действие:")
        print(f"    - Очки S: {state['actions']['score_S']}")
        print(f"    - Очки R: {state['actions']['score_R']}")
        print(f"  Комментарий: {state['comment']}")
        print()
    
    # Сохраняем в JSON
    with open('fsm_logic.json', 'w', encoding='utf-8') as f:
        json.dump(states, f, ensure_ascii=False, indent=2)
    
    print("OK: Логика сохранена в fsm_logic.json")


