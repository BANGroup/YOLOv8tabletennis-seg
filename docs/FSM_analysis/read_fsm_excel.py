#!/usr/bin/env python3
"""
Чтение Excel таблицы с FSM логикой
"""

import openpyxl
import json

def read_fsm_excel(excel_file):
    """Читает Excel файл с FSM логикой"""
    
    print(f"\n=== ЧТЕНИЕ {excel_file} ===\n")
    
    wb = openpyxl.load_workbook(excel_file)
    
    print(f"Листы в файле: {wb.sheetnames}\n")
    
    # Читаем каждый лист
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        print(f"\n{'='*80}")
        print(f"ЛИСТ: {sheet_name}")
        print(f"{'='*80}\n")
        
        # Читаем все ячейки
        for row in ws.iter_rows(min_row=1, max_row=ws.max_row, values_only=False):
            for cell in row:
                if cell.value is not None:
                    print(f"  {cell.coordinate}: {cell.value}")
            print()

if __name__ == "__main__":
    read_fsm_excel("tt_fsm_rules_rev_1.xlsx")


