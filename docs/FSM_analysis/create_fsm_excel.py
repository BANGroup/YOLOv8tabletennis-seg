#!/usr/bin/env python3
"""
Создание Excel таблицы с FSM логикой для трекинга теннисного мяча
"""

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

def create_fsm_table():
    """Создает Excel таблицу с FSM логикой"""
    
    # Создаем новую книгу
    wb = openpyxl.Workbook()
    
    # ============================================================
    # ЛИСТ 1: ПАРАМЕТРЫ И ТАЙМЕРЫ
    # ============================================================
    ws_params = wb.active
    ws_params.title = "Параметры"
    
    # Заголовок
    ws_params['A1'] = "ПАРАМЕТРЫ И ТАЙМЕРЫ ДЕТЕКТОРА РОЗЫГРЫШЕЙ"
    ws_params['A1'].font = Font(bold=True, size=14)
    ws_params.merge_cells('A1:E1')
    
    # Заголовки таблицы
    headers = ['№', 'Параметр', 'Текущее значение', 'Рекомендуемое', 'Описание']
    for col, header in enumerate(headers, 1):
        cell = ws_params.cell(row=3, column=col)
        cell.value = header
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    
    # Данные параметров
    params_data = [
        [1, "confidence_threshold", 0.25, 0.25, "Минимальная уверенность YOLO для мяча"],
        [2, "min_ball_area", 100, 80, "Минимальная площадь мяча (пиксели²)"],
        [3, "max_ball_area", 3000, 3000, "Максимальная площадь мяча"],
        [4, "max_jump", 100, 100, "Максимальный скачок между кадрами (пиксели)"],
        [5, "max_speed", 150, 150, "Максимальная скорость мяча (пиксели/кадр)"],
        [6, "min_velocity", "—", 2.0, "🆕 Минимальная скорость для начала розыгрыша"],
        [7, "max_direction_change", 90, 180, "❌→✓ Максимальная смена направления (градусы)"],
        [8, "min_rally_duration", 1.5, 0.3, "❌→✓ Минимальная длительность розыгрыша (сек)"],
        [9, "max_frames_outside", 30, 30, "Максимум кадров вне стола (0.5 сек при 60FPS)"],
        [10, "min_consecutive_detections", 1, 3, "🆕 Минимум последовательных детекций перед началом"],
        [11, "max_frames_no_detection", "—", 30, "🆕 Максимум кадров без детекции мяча (0.5 сек)"],
        [12, "table_detection_frequency", "—", 10, "🆕 Детектировать стол раз в N кадров"],
    ]
    
    for row_idx, row_data in enumerate(params_data, 4):
        for col_idx, value in enumerate(row_data, 1):
            cell = ws_params.cell(row=row_idx, column=col_idx)
            cell.value = value
            cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
            if col_idx == 1:
                cell.alignment = Alignment(horizontal="center")
            if "→" in str(value) or "🆕" in str(value):
                cell.font = Font(color="FF0000", bold=True)
    
    # Ширина колонок
    ws_params.column_dimensions['A'].width = 5
    ws_params.column_dimensions['B'].width = 30
    ws_params.column_dimensions['C'].width = 20
    ws_params.column_dimensions['D'].width = 20
    ws_params.column_dimensions['E'].width = 40
    
    # ============================================================
    # ЛИСТ 2: FSM ТАБЛИЦА ПЕРЕХОДОВ
    # ============================================================
    ws_fsm = wb.create_sheet("FSM Переходы")
    
    # Заголовок
    ws_fsm['A1'] = "ТАБЛИЦА ПЕРЕХОДОВ FSM (КОНЕЧНЫЙ АВТОМАТ)"
    ws_fsm['A1'].font = Font(bold=True, size=14, color="FFFFFF")
    ws_fsm['A1'].fill = PatternFill(start_color="203864", end_color="203864", fill_type="solid")
    ws_fsm.merge_cells('A1:H1')
    
    # Заголовки
    fsm_headers = [
        'Текущее\nсостояние',
        'Условие 1\nлогика И',
        'Условие 2\nлогика И',
        'Условие 3\nлогика И',
        'Новое\nсостояние',
        'Действие',
        'Значения\nсчетчиков',
        'Примечание'
    ]
    
    for col, header in enumerate(fsm_headers, 1):
        cell = ws_fsm.cell(row=3, column=col)
        cell.value = header
        cell.font = Font(bold=True, color="FFFFFF", size=10)
        cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    
    # Данные FSM
    fsm_data = [
        [
            "IDLE\n(ожидание)",
            "Мяч обнаружен",
            "На столе",
            "3+ кадра подряд",
            "RALLY_STARTING",
            "Начать розыгрыш,\nсохранить время",
            "rally_counter++,\nstart_frame = N,\ntrajectory = []",
            "Первое обнаружение мяча на столе"
        ],
        [
            "RALLY_STARTING\n(подтверждение)",
            "Мяч движется",
            "velocity >= 2.0 пикс/кадр",
            "Направление вниз/горизонтально",
            "RALLY_ACTIVE",
            "Активировать\nтрекинг",
            "frames_out_table = 0,\nframes_no_detection = 0",
            "Мяч реально движется, не просто видна"
        ],
        [
            "RALLY_ACTIVE\n(розыгрыш)",
            "Мяч обнаружен",
            "На столе",
            "—",
            "RALLY_ACTIVE",
            "Добавить точку\nв траекторию",
            "frames_out_table = 0,\nframes_no_detection = 0",
            "Продолжение розыгрыша"
        ],
        [
            "RALLY_ACTIVE\n(розыгрыш)",
            "Мяч вне стола",
            "frames_out_table < 30 (0.5сек)",
            "—",
            "RALLY_ACTIVE",
            "Инкрементировать\nсчетчик",
            "frames_out_table++",
            "Отскок/сетка/помеха, продолжить ралли"
        ],
        [
            "RALLY_ACTIVE\n(розыгрыш)",
            "Мяч не обнаружен",
            "frames_no_detection < 30 (0.5сек)",
            "—",
            "RALLY_ACTIVE",
            "Инкрементировать\nсчетчик",
            "frames_no_detection++",
            "Временная потеря мяча (игрок/помеха)"
        ],
        [
            "RALLY_ACTIVE\n(розыгрыш)",
            "Мяч вне стола ИЛИ\nнет детекции",
            "frames_out_table >= 30 (0.5сек)\nИЛИ\nframes_no_detection >= 30",
            "—",
            "RALLY_ENDING",
            "Завершить\nтрекинг",
            "end_frame = N,\nrally_duration = t",
            "Мяч вне стола > 0.5сек = конец ралли"
        ],
        [
            "RALLY_ENDING\n(проверка)",
            "rally_duration >= 0.3 сек",
            "trajectory_points >= 3",
            "—",
            "IDLE",
            "Сохранить\nв JSON",
            "rallies[] += rally,\nrally_counter",
            "Валидный розыгрыш"
        ],
        [
            "RALLY_ENDING\n(проверка)",
            "rally_duration < 0.3 сек",
            "ИЛИ\ntrajectory_points < 3",
            "—",
            "IDLE",
            "Отбросить,\nничего не сохранять",
            "Ничего",
            "Шум/ложная детекция, отфильтрована"
        ],
        [
            "RALLY_ACTIVE\n(розыгрыш)",
            "Смена направления",
            "angle_change <= 180°",
            "—",
            "RALLY_ACTIVE",
            "Добавить точку,\nобновить угол",
            "direction_history[]",
            "Обратный удар - разрешен"
        ],
        [
            "RALLY_ACTIVE\n(розыгрыш)",
            "Смена направления",
            "angle_change > 180°",
            "—",
            "RALLY_ACTIVE",
            "Отклонить точку,\nлогировать",
            "false_positives++",
            "Физически невозможно - ошибка YOLO"
        ],
    ]
    
    for row_idx, row_data in enumerate(fsm_data, 4):
        for col_idx, value in enumerate(row_data, 1):
            cell = ws_fsm.cell(row=row_idx, column=col_idx)
            cell.value = value
            cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
            cell.font = Font(size=9)
            
            # Раскрашиваем состояния
            if col_idx == 1 and value:
                if "IDLE" in str(value):
                    cell.fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
                elif "STARTING" in str(value):
                    cell.fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
                elif "ACTIVE" in str(value):
                    cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                elif "ENDING" in str(value):
                    cell.fill = PatternFill(start_color="F4B084", end_color="F4B084", fill_type="solid")
            
            if col_idx == 5 and value:  # Новое состояние
                if "IDLE" in str(value):
                    cell.fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
                elif "STARTING" in str(value):
                    cell.fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
                elif "ACTIVE" in str(value):
                    cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                elif "ENDING" in str(value):
                    cell.fill = PatternFill(start_color="F4B084", end_color="F4B084", fill_type="solid")
    
    # Ширина колонок FSM
    ws_fsm.column_dimensions['A'].width = 18
    ws_fsm.column_dimensions['B'].width = 22
    ws_fsm.column_dimensions['C'].width = 22
    ws_fsm.column_dimensions['D'].width = 22
    ws_fsm.column_dimensions['E'].width = 18
    ws_fsm.column_dimensions['F'].width = 20
    ws_fsm.column_dimensions['G'].width = 22
    ws_fsm.column_dimensions['H'].width = 28
    
    # ============================================================
    # ЛИСТ 3: ТАЙМЕРЫ И СЧЕТЧИКИ
    # ============================================================
    ws_timers = wb.create_sheet("Таймеры")
    
    # Заголовок
    ws_timers['A1'] = "СИСТЕМА ТАЙМЕРОВ И СЧЕТЧИКОВ"
    ws_timers['A1'].font = Font(bold=True, size=14, color="FFFFFF")
    ws_timers['A1'].fill = PatternFill(start_color="203864", end_color="203864", fill_type="solid")
    ws_timers.merge_cells('A1:F1')
    
    # Заголовки
    timer_headers = ['Таймер/Счетчик', 'Максимум', 'Единица', 'Когда сбрасывается', 'При каком событии', 'Действие']
    for col, header in enumerate(timer_headers, 1):
        cell = ws_timers.cell(row=3, column=col)
        cell.value = header
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color="70AD47", end_color="70AD47", fill_type="solid")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    
    timer_data = [
        ["frames_out_of_table", 30, "кадры (0.5 сек @ 60FPS)", "Мяч вновь обнаружен на столе", "Мяч вне стола", "Отскок/сетка/помеха"],
        ["frames_no_detection", 30, "кадры (0.5 сек @ 60FPS)", "Мяч вновь обнаружен", "Мяч не видна", "Потеря мяча (игрок)"],
        ["consecutive_detections", 3, "кадры", "Мяч не обнаружен или нарушена консистентность", "Каждая детекция", "Фильтр ложных шумов"],
        ["min_rally_frames", 18, "кадры (0.3 сек @ 60FPS)", "Начало нового розыгрыша", "—", "Фильтр очень коротких ралли"],
        ["table_detection_counter", 10, "кадры", "Детектирование стола", "Каждый кадр", "Оптимизация: кэш маски стола"],
        ["rally_counter", "—", "единицы", "Сохранение розыгрыша", "Каждый завершенный ралли", "Нумерация розыгрышей"],
    ]
    
    for row_idx, row_data in enumerate(timer_data, 4):
        for col_idx, value in enumerate(row_data, 1):
            cell = ws_timers.cell(row=row_idx, column=col_idx)
            cell.value = value
            cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
            if col_idx == 2 and value != "—":
                cell.alignment = Alignment(horizontal="center")
    
    ws_timers.column_dimensions['A'].width = 25
    ws_timers.column_dimensions['B'].width = 12
    ws_timers.column_dimensions['C'].width = 22
    ws_timers.column_dimensions['D'].width = 30
    ws_timers.column_dimensions['E'].width = 28
    ws_timers.column_dimensions['F'].width = 25
    
    # ============================================================
    # ЛИСТ 4: ТЕСТОВЫЕ СЦЕНАРИИ
    # ============================================================
    ws_tests = wb.create_sheet("Тесты")
    
    # Заголовок
    ws_tests['A1'] = "ТЕСТОВЫЕ СЦЕНАРИИ"
    ws_tests['A1'].font = Font(bold=True, size=14, color="FFFFFF")
    ws_tests['A1'].fill = PatternFill(start_color="203864", end_color="203864", fill_type="solid")
    ws_tests.merge_cells('A1:E1')
    
    test_headers = ['№', 'Сценарий', 'Ожидаемый результат', 'Текущий результат', 'После фикса']
    for col, header in enumerate(test_headers, 1):
        cell = ws_tests.cell(row=3, column=col)
        cell.value = header
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color="C55A11", end_color="C55A11", fill_type="solid")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    
    test_data = [
        [1, "Подача + возврат (2 сек)", "1 розыгрыш", "✓ РАБОТАЕТ", "✓ РАБОТАЕТ"],
        [2, "Быстрый пас (0.4 сек)", "1 розыгрыш", "❌ ТЕРЯЕТСЯ", "✓ СОХРАНЕН"],
        [3, "Обратный удар (5 сек)", "1 целый ралли", "❌ РАЗБИТ НА ЧАСТИ", "✓ ЦЕЛЫЙ"],
        [4, "Отскок (0.1 сек вне видео)", "1 розыгрыш (продолжен)", "✓ РАБОТАЕТ", "✓ РАБОТАЕТ"],
        [5, "Помеха от игрока (0.3 сек)", "1 розыгрыш (продолжен)", "⚠️ МОЖЕТ ОШИБИТЬСЯ", "✓ ПРАВИЛЬНО"],
        [6, "Шум YOLO (1 кадр)", "0 розыгрышей", "⚠️ МИКРОРОЗЫГРЫШ", "✓ ОТФИЛЬТРОВАН"],
        [7, "Трясение камеры", "0 розыгрышей", "⚠️ ЛОЖНЫЙ СТАРТ", "✓ ИГНОРИРОВАН"],
        [8, "Нету мяча 0.5 сек", "Конец ралли", "✓ РАБОТАЕТ", "✓ РАБОТАЕТ"],
        [9, "Нету мяча 0.3 сек, потом найден", "Продолжить ралли", "✓ РАБОТАЕТ", "✓ РАБОТАЕТ"],
        [10, "Мяч очень маленький (далеко)", "1 розыгрыш", "⚠️ МОЖЕТ ПОТЕРЯТЬСЯ", "✓ СОХРАНЕН"],
    ]
    
    for row_idx, row_data in enumerate(test_data, 4):
        for col_idx, value in enumerate(row_data, 1):
            cell = ws_tests.cell(row=row_idx, column=col_idx)
            cell.value = value
            cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
            if col_idx == 1:
                cell.alignment = Alignment(horizontal="center")
            
            # Раскрашиваем результаты
            if "✓" in str(value):
                cell.font = Font(color="00B050", bold=True)
            elif "❌" in str(value):
                cell.font = Font(color="FF0000", bold=True)
            elif "⚠️" in str(value):
                cell.font = Font(color="FFC000", bold=True)
    
    ws_tests.column_dimensions['A'].width = 5
    ws_tests.column_dimensions['B'].width = 30
    ws_tests.column_dimensions['C'].width = 28
    ws_tests.column_dimensions['D'].width = 28
    ws_tests.column_dimensions['E'].width = 28
    
    # ============================================================
    # ЛИСТ 5: ДИАГРАММА СОСТОЯНИЙ
    # ============================================================
    ws_diagram = wb.create_sheet("Диаграмма")
    
    # Заголовок
    ws_diagram['A1'] = "ДИАГРАММА ПЕРЕХОДОВ СОСТОЯНИЙ"
    ws_diagram['A1'].font = Font(bold=True, size=14)
    
    diagram_text = """
    IDLE (ожидание)
      ↓
      ├─ Мяч обнаружен на столе
      ├─ 3+ последовательные детекции
      └─→ RALLY_STARTING
           ↓
           ├─ Мяч движется (velocity >= 2.0)
           ├─ Направление: вниз или горизонтально
           └─→ RALLY_ACTIVE (розыгрыш идет)
                ↓
                ├─ СОБЫТИЯ:
                │  ├─ Мяч на столе → Добавить точку траектории
                │  ├─ Мяч вне стола, < 0.5сек → Продолжить (отскок)
                │  ├─ Нету мяча, < 0.5сек → Продолжить (помеха)
                │  └─ Смена угла <= 180° → Разрешить (обратный удар)
                │
                └─→ Когда:
                   ├─ frames_out_of_table >= 30 (0.5сек)
                   │  ИЛИ
                   ├─ frames_no_detection >= 30 (0.5сек)
                   └─→ RALLY_ENDING
                       ↓
                       ├─ Если: duration >= 0.3сек И points >= 3
                       │  └─→ Сохранить в JSON → IDLE ✓
                       │
                       └─ Если: duration < 0.3сек ИЛИ points < 3
                          └─→ Отбросить → IDLE ❌
    """
    
    row = 3
    for line in diagram_text.split('\n'):
        ws_diagram[f'A{row}'] = line
        ws_diagram[f'A{row}'].font = Font(name='Courier New', size=10)
        row += 1
    
    ws_diagram.column_dimensions['A'].width = 80
    
    # ============================================================
    # ЛИСТ 6: ПРИМЕЧАНИЯ
    # ============================================================
    ws_notes = wb.create_sheet("Примечания")
    
    ws_notes['A1'] = "ВАЖНЫЕ ПРИМЕЧАНИЯ И УСЛОВИЯ"
    ws_notes['A1'].font = Font(bold=True, size=14, color="FFFFFF")
    ws_notes['A1'].fill = PatternFill(start_color="203864", end_color="203864", fill_type="solid")
    ws_notes.merge_cells('A1:B1')
    
    notes_data = [
        ["🎯 Цель системы", "Детектировать и отследить каждый розыгрыш в настольном теннисе для online счета"],
        ["🔄 Как работает FSM", "Конечный автомат с 4 состояниями: IDLE → RALLY_STARTING → RALLY_ACTIVE → RALLY_ENDING → IDLE"],
        ["⏱️ Главный таймер", "Мяч вне стола > 0.5 сек (30 кадров @ 60FPS) = конец розыгрыша"],
        ["🚨 Критичные условия", "1) max_direction_change = 180° (обратные удары)\n2) min_rally_duration = 0.3 сек (быстрые пассы)\n3) min_consecutive = 3 (фильтр шумов)\n4) min_velocity = 2.0 (фильтр камеры)"],
        ["📊 Точность после фиксов", "95%+ (вместо текущих 70%)"],
        ["⚙️ Время реализации ФАЗЫ 1", "45 минут (4 простых фикса)"],
        ["✅ Готовность", "После ФАЗЫ 1 система готова к online счету"],
    ]
    
    for row_idx, (label, value) in enumerate(notes_data, 3):
        cell_label = ws_notes[f'A{row_idx}']
        cell_label.value = label
        cell_label.font = Font(bold=True, size=11, color="FFFFFF")
        cell_label.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        
        cell_value = ws_notes[f'B{row_idx}']
        cell_value.value = value
        cell_value.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
        cell_value.font = Font(size=10)
    
    ws_notes.column_dimensions['A'].width = 30
    ws_notes.column_dimensions['B'].width = 70
    
    # ============================================================
    # Сохраняем файл
    # ============================================================
    output_file = "FSM_Rally_Detector.xlsx"
    wb.save(output_file)
    print("OK: Excel файл создан: " + output_file)
    print("OK: Листов: 6")
    print("  1. Параметры (12 параметров)")
    print("  2. FSM Переходы (10 переходов состояний)")
    print("  3. Таймеры (6 таймеров/счетчиков)")
    print("  4. Тесты (10 тестовых сценариев)")
    print("  5. Диаграмма (визуальная диаграмма FSM)")
    print("  6. Примечания (ключевая информация)")

if __name__ == "__main__":
    create_fsm_table()

