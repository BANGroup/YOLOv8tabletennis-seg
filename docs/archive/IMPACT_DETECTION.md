# 🎾 IMPACT DETECTION: Обнаружение ударов об стол

## Что это?

Система определяет, когда мяч ударяется об теннисный стол, и определяет **кто именно ударил** (sender) и кто принимал (receiver).

---

## 🔴 Основная проблема: Sender vs Receiver

### Теоретическое обоснование

Теннисный стол имеет **две половины**:

```
         Сетка
           ↓
  ┌─────────────────┐
  │  Player1 Side   │  ← Левая половина (X < width/2)
  ├─────────────────┤
  │  Player2 Side   │  ← Правая половина (X > width/2)
  └─────────────────┘
  
Player1 (слева)     Player2 (справа)
```

### Логика определения

**Когда мяч ударяется:**

```
УДАР НА ЛЕВОЙ СТОРОНЕ (X < 960 для 1920x1080):
├─ Кто ударил? → Player1
├─ sender = Player1
├─ receiver = Player2
└─ Мяч летит на Player2

УДАР НА ПРАВОЙ СТОРОНЕ (X > 960):
├─ Кто ударил? → Player2
├─ sender = Player2
├─ receiver = Player1
└─ Мяч летит на Player1
```

### Чередование ударов

```
Последовательность ударов:
1️⃣ Player1 ударяет    → sender=P1, receiver=P2
2️⃣ Player2 ударяет    → sender=P2, receiver=P1
3️⃣ Player1 ударяет    → sender=P1, receiver=P2
4️⃣ Player2 ударяет    → sender=P2, receiver=P1
... и так далее
```

---

## 🔍 Как работает обнаружение удара

### Критерии удара об стол

```
Удар = True, если:
1. ✓ Мяч находится в ЗОНЕ СТОЛА (Y > height * 0.65)
2. ✓ ВЕРТИКАЛЬНАЯ смена направления:
     - Движение ВНИЗ (v1[Y] > 0.15) +
     - Отскок ВВЕРХ (v2[Y] < -0.15)
3. ✓ Минимум ДВИЖЕНИЯ (не шум):
     - velocity > 2.0 пикс/кадр
4. ✓ Угол ПЕРЕЛОМА (достаточный):
     - angle >= 20°
5. ✓ ПОСЛЕДОВАТЕЛЬНЫЕ детекции:
     - min_consecutive = 3 (фильтр YOLO шума)
```

### Визуальное представление

```
         Мяч летит вниз
              ↓
        ┌─────────────┐
        │   СТОЛ      │  ← Y > height * 0.65
        └─────────────┘
              ↑↑↑
        Мяч отскочил ВВЕРХ
        (УДАР ОБНАРУЖЕН!)
```

---

## 📊 Параметры из smart_tracking анализа

| Параметр | Значение | Назначение |
|----------|----------|-----------|
| `max_direction_change` | 180° | Разрешить развороты на 180° |
| `min_rally_duration` | 0.3 сек | Включить быстрые пассы |
| `min_velocity` | 2.0 пикс/кадр | Фильтровать шум камеры |
| `min_consecutive_detections` | 3 | Фильтровать одиночный шум YOLO |
| `table_zone_start` | 65% высоты | Зона удара об стол |

---

## 🎯 Пример: 4 удара в 2 секундах

### Сценарий
```
Время: 06:35-06:37 (2 секунды видео)
Видеопоток: 59.94 FPS
Всего кадров: 120

Детектированы 4 удара:
```

### Ожидаемые удары

```
УДАР 1: ~0.3 сек
├─ Позиция X: 300 (ЛЕВАЯ сторона)
├─ sender = Player1
├─ receiver = Player2
└─ Вид: 🟢 Зеленая траектория → 🔴 Красная точка удара

УДАР 2: ~0.7 сек
├─ Позиция X: 1600 (ПРАВАЯ сторона)
├─ sender = Player2
├─ receiver = Player1
└─ Вид: 🟢 Траектория → 🔴 Красная точка удара

УДАР 3: ~1.2 сек
├─ Позиция X: 400 (ЛЕВАЯ сторона)
├─ sender = Player1
├─ receiver = Player2
└─ Вид: 🔴 Красная точка удара

УДАР 4: ~1.8 сек
├─ Позиция X: 1500 (ПРАВАЯ сторона)
├─ sender = Player2
├─ receiver = Player1
└─ Вид: 🔴 Красная точка удара
```

### JSON результат

```json
{
  "impacts": [
    {
      "impact_number": 1,
      "frame": 18,
      "time": 0.3,
      "position_x": 300,
      "sender": "Player1",
      "receiver": "Player2",
      "side": "LEFT",
      "angle": 120.5,
      "velocity_before": 15.2,
      "velocity_after": 12.1
    },
    {
      "impact_number": 2,
      "frame": 42,
      "time": 0.7,
      "position_x": 1600,
      "sender": "Player2",
      "receiver": "Player1",
      "side": "RIGHT",
      "angle": 130.2,
      "velocity_before": 18.5,
      "velocity_after": 14.3
    },
    // ... ещё 2 удара
  ]
}
```

---

## 🔧 Как определить стороны стола в коде

```python
def determine_side(impact_x, frame_width):
    """Определить какая сторона стола"""
    center_x = frame_width / 2
    
    if impact_x < center_x:
        return {
            'side': 'LEFT',
            'sender': 'Player1',
            'receiver': 'Player2'
        }
    else:
        return {
            'side': 'RIGHT',
            'sender': 'Player2',
            'receiver': 'Player1'
        }

# Пример
impact_x = 400
frame_width = 1920
result = determine_side(impact_x, frame_width)
# → {'side': 'LEFT', 'sender': 'Player1', 'receiver': 'Player2'}
```

---

## 📈 Как sender/receiver меняются при каждом ударе

```python
def track_rally(impacts):
    """Отслеживать rally с чередованием sender/receiver"""
    
    rally_sequence = []
    last_sender = None
    
    for i, impact in enumerate(impacts):
        current_side = impact['side']
        
        if i == 0:
            # Первый удар - определяем по стороне
            sender = 'Player1' if current_side == 'LEFT' else 'Player2'
        else:
            # Последующие удары - чередуем
            sender = 'Player2' if last_sender == 'Player1' else 'Player1'
        
        receiver = 'Player2' if sender == 'Player1' else 'Player1'
        
        rally_sequence.append({
            'number': i + 1,
            'sender': sender,
            'receiver': receiver,
            'time': impact['time']
        })
        
        last_sender = sender
    
    return rally_sequence

# Результат:
# [
#   {'number': 1, 'sender': 'Player1', 'receiver': 'Player2', 'time': 0.3},
#   {'number': 2, 'sender': 'Player2', 'receiver': 'Player1', 'time': 0.7},
#   {'number': 3, 'sender': 'Player1', 'receiver': 'Player2', 'time': 1.2},
#   {'number': 4, 'sender': 'Player2', 'receiver': 'Player1', 'time': 1.8}
# ]
```

---

## ⚠️ Важные замечания

### 1. Порядок определения важен!

```
НЕПРАВИЛЬНО ❌:
- Определяем sender/receiver только по X-координате
- Результат: ошибки в определении при отскоках

ПРАВИЛЬНО ✅:
- Сначала определяем стороны по X
- Затем чередуем роли логически
```

### 2. Граница стола

```
Для видео 1920x1080:
- CENTER_X = 960
- LEFT SIDE:  0-959
- RIGHT SIDE: 961-1920

⚠️ Не забыть добавить MARGIN для близких к центру ударов!
```

### 3. Особые случаи

```
ПРОБЛЕМА: Отскок близко к центру (X=960±50)
РЕШЕНИЕ: Использовать направление мяча для определения

ПРОБЛЕМА: Не видим сторону удара (окклюзия)
РЕШЕНИЕ: Использовать логику чередования
```

---

## 🎬 Визуализация на видео

### Пример вывода

```
Детектирование: Player1 (LEFT) → sender
                 Player2 (RIGHT) → receiver
Результат:
┌────────────────────────────────┐
│ Impact #1: Player1 SERVES      │
│ Receiver: Player2              │
│ Time: 0:00.3                   │
│ Speed: 15.2 px/frame           │
│ Angle: 120.5°                  │
└────────────────────────────────┘
```

---

## 📚 Ссылки на основные компоненты

- **SmartImpactDetector** - класс в `utils/impact_detector.py`
- **Параметры** - в `docs/PARAMETERS.md`
- **Архитектура** - в `docs/ARCHITECTURE.md`
- **Главный скрипт** - `scripts/main_processor.py`

---

## 🔬 Тестирование

### Проверить обнаружение 4 ударов

```bash
python scripts/main_processor.py
# Должно вывести:
# SMART УДАРОВ об СТОЛ: 4
```

### Проверить sender/receiver

```bash
cat results/tracked_2sec_smart_impacts/smart_impacts_analysis.json
# Должно показать:
# "impacts": [
#   {"number": 1, "sender": "Player1", "receiver": "Player2"},
#   {"number": 2, "sender": "Player2", "receiver": "Player1"},
#   ...
# ]
```

---

**✨ Документ завершен**

