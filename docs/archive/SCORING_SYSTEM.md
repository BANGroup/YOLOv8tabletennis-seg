# 🏆 SCORING SYSTEM: Система подсчета очков

Документ описывает, как система автоматически подсчитывает очки в настольном теннисе - от отдельных ударов до полного матча.

---

## 📊 Иерархия подсчета

```
Level 1: IMPACTS (Удары)
   ↓
Level 2: RALLIES (Розыгрыши)
   ↓
Level 3: POINTS (Очки)
   ↓
Level 4: GAMES (Игры)
   ↓
Level 5: MATCH (Матч)
```

---

## LEVEL 1: IMPACTS (Удары)

### Что это?
Каждый раз когда мяч **ударяется об стол**.

### Как определить?
```
Условия:
1. ✓ Y > height * 0.65      (в зоне стола)
2. ✓ Y движение: вниз→вверх (отскок)
3. ✓ Velocity > 2.0 px/frame (не шум)
4. ✓ Angle > 20°             (достаточный перелом)

Результат: Impact с sender/receiver
```

### Пример
```json
{
  "impact_id": 1,
  "time": 0.3,
  "frame": 18,
  "sender": "Player1",
  "receiver": "Player2",
  "position_x": 300,
  "side": "LEFT"
}
```

---

## LEVEL 2: RALLIES (Розыгрыши)

### Что это?
**Последовательность ударов** между двумя игроками, начиная с подачи одного и заканчивая ошибкой другого.

### Как строится rally?
```
Rally = [
  Impact(Player1→Player2),  ← Удар 1
  Impact(Player2→Player1),  ← Удар 2
  Impact(Player1→Player2),  ← Удар 3
  КОНЕЦ (один ошибся)       ← Победитель определен
]

Rally winner = Тот кто НЕ ошибся
```

### Структура rally
```json
{
  "rally_id": 1,
  "impacts": [
    {"sender": "Player1", "receiver": "Player2", "time": 0.3},
    {"sender": "Player2", "receiver": "Player1", "time": 0.7},
    {"sender": "Player1", "receiver": "Player2", "time": 1.2}
  ],
  "start_time": 0.3,
  "end_time": 1.2,
  "duration": 0.9,
  "winner": "Player1",
  "loser": "Player2",
  "impact_count": 3
}
```

### Как определить конец rally?
```
КРИТЕРИИ КОНЦА:
1. Мяч не вернулся на стол (> 0.5 сек отсутствия)
2. Игрок не мог ударить (находится далеко)
3. Мяч вышел за границы стола (по X или Y)
4. Мяч ударил в сетку

ТОГДА: Rally закончился → Определяем победителя
```

---

## LEVEL 3: POINTS (Очки)

### Что это?
Результат каждого **rally**.

### Правила подсчета

```
Стандартные правила настольного тенниса:

1. Winner rally = +1 очко
2. Loser rally = без изменений

ТЕКУЩИЙ СЧЕТ: Player1 5 - 6 Player2
```

### Как считается?

```python
def process_rally_result(rally_winner, current_score):
    """Добавить очко к rally winner"""
    
    if rally_winner == 'Player1':
        current_score['player1'] += 1
        print(f"Point to Player1! Score: {current_score}")
    else:
        current_score['player2'] += 1
        print(f"Point to Player2! Score: {current_score}")
    
    return current_score

# Результат
# Point to Player1! Score: {'player1': 6, 'player2': 6}
```

### Пример серии очков
```
Rally 1: Player1 выиграл → Score: 1-0
Rally 2: Player2 выиграл → Score: 1-1
Rally 3: Player1 выиграл → Score: 2-1
Rally 4: Player1 выиграл → Score: 3-1
Rally 5: Player2 выиграл → Score: 3-2
Rally 6: Player2 выиграл → Score: 3-3
```

---

## LEVEL 4: GAMES (Игры)

### Что это?
Первый кто достигает **11 очков** с разницей минимум **2 очка** выигрывает игру.

### Правила

```
УСЛОВИЕ ПОБЕДЫ В ИГРЕ:
- player1_score >= 11 AND (player1_score - player2_score) >= 2

ПРИМЕРЫ ПОБЕД:
✓ 11-9   → Player1 выиграл
✓ 11-8   → Player1 выиграл
✓ 12-10  → Player1 выиграл
✓ 13-11  → Player1 выиграл

✗ 11-10  → Продолжается (нужна разница 2)
✗ 11-11  → Продолжается ( 11-9, 11-8, и т.д.)
```

### Как считается?

```python
def check_game_winner(score):
    """Проверить выиграна ли игра"""
    
    p1, p2 = score['player1'], score['player2']
    
    if p1 >= 11 and (p1 - p2) >= 2:
        return 'Player1'  # Победа
    elif p2 >= 11 and (p2 - p1) >= 2:
        return 'Player2'  # Победа
    else:
        return None       # Продолжается

# Примеры
check_game_winner({'player1': 11, 'player2': 9})   # → 'Player1'
check_game_winner({'player1': 11, 'player2': 10})  # → None (продолжается)
check_game_winner({'player1': 13, 'player2': 11})  # → 'Player1'
```

### Пример одной игры

```
Score progression:
1-0 → 1-1 → 2-1 → 3-1 → 4-1 → 5-1 →
6-1 → 7-1 → 8-1 → 9-1 → 10-1 → 11-1

RESULT: Player1 выиграл игру 11-1 ✓
Games: 1-0 (Player1 ahead 1 game)
```

### Полная игра (долгая)

```
Score progression:
0-0 → 1-0 → 1-1 → 2-1 → ... → 10-10 →
10-11 → 11-11 → 12-11 → 12-12 → 13-12

RESULT: Player1 выиграл игру 13-12 ✓
Games: 1-0
```

---

## LEVEL 5: MATCH (Матч)

### Что это?
Первый кто выигрывает **2 игры** выигрывает **матч**.

### Правила

```
УСЛОВИЕ ПОБЕДЫ В МАТЧЕ:
- player1_games >= 2 (выиграл 2 игры)
- player2_games < 2

ПРИМЕРЫ ПОБЕД:
✓ 2-0   → Player1 выиграл матч
✓ 2-1   → Player1 выиграл матч
✓ 0-2   → Player2 выиграл матч
✓ 1-2   → Player2 выиграл матч

✗ 1-1   → Продолжается (нужно 2 игры)
✗ 0-0   → Начало (еще не начало)
```

### Как считается?

```python
def check_match_winner(games):
    """Проверить выигран ли матч"""
    
    g1, g2 = games['player1'], games['player2']
    
    if g1 >= 2:
        return 'Player1'  # Матч выигран
    elif g2 >= 2:
        return 'Player2'  # Матч выигран
    else:
        return None       # Продолжается

# Примеры
check_match_winner({'player1': 2, 'player2': 0})  # → 'Player1' ✓
check_match_winner({'player1': 2, 'player2': 1})  # → 'Player1' ✓
check_match_winner({'player1': 1, 'player2': 1})  # → None (продолжается)
check_match_winner({'player1': 0, 'player2': 0})  # → None (начало)
```

### Пример полного матча

```
GAME 1: Player1 11-9 Player2 (Games: 1-0)
GAME 2: Player2 13-12 Player1 (Games: 1-1)
GAME 3: Player1 11-8 Player2 (Games: 2-1)

RESULT: Player1 выиграл матч 2-1 ✓
Final: Player1 defeats Player2 in best-of-3
```

---

## 🔄 Полный пример: От ударов к матчу

### Сценарий: 2 секунды видео

```
IMPACTS:
1. Impact #1 @ 0.3s: Player1→Player2 (LEFT)
2. Impact #2 @ 0.7s: Player2→Player1 (RIGHT)
3. Impact #3 @ 1.2s: Player1→Player2 (LEFT)
4. Impact #4 @ 1.8s: [Player2 ошибся]

↓ RALLIES:

Rally #1:
  - Duration: 0.3-1.8s (1.5 сек)
  - Impacts: 3
  - Winner: Player1 (Player2 ошибся на Impact #4)
  - Points: +1 for Player1

↓ POINTS:

Before: Player1 5 - 6 Player2
After:  Player1 6 - 6 Player2 (Tied!)

↓ GAMES:

Current Game: Player1 6 - 6 Player2
Status: Continuing (need 11 with 2+ diff)

↓ MATCH:

Games: 0 - 0
Status: Game 1 in progress
```

---

## 📋 JSON Структура результатов

```json
{
  "match_info": {
    "video": "sources/GX010138.MP4",
    "duration_seconds": 2.0,
    "start_time": "06:35",
    "end_time": "06:37"
  },
  "impacts": [
    {
      "id": 1,
      "time": 0.3,
      "sender": "Player1",
      "receiver": "Player2",
      "side": "LEFT"
    },
    {
      "id": 2,
      "time": 0.7,
      "sender": "Player2",
      "receiver": "Player1",
      "side": "RIGHT"
    }
  ],
  "rallies": [
    {
      "id": 1,
      "start_time": 0.3,
      "end_time": 1.8,
      "impacts": [1, 2, 3],
      "winner": "Player1",
      "loser": "Player2"
    }
  ],
  "points": {
    "player1": 6,
    "player2": 6,
    "last_rally_winner": "Player1"
  },
  "games": {
    "player1": 0,
    "player2": 0,
    "current_game_score": "6-6"
  },
  "match": {
    "status": "in_progress",
    "winner": null,
    "games_to_win": 2
  }
}
```

---

## 🎯 Визуализация на видео

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  TABLE TENNIS CV SCORE             ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                    ┃
┃  MATCH SCORE                       ┃
┃  ═══════════════════════════════  ┃
┃  Player1 (0) - (0) Player2        ┃
┃  Games   2 - 1                    ┃
┃                                    ┃
┃  CURRENT GAME                      ┃
┃  ═══════════════════════════════  ┃
┃  Player1 (6) - (6) Player2        ┃
┃  Points  11 - 9 to win            ┃
┃                                    ┃
┃  LAST IMPACT                       ┃
┃  ═══════════════════════════════  ┃
┃  Rally #1 ✓ Player1 WINS           ┃
┃  Time: 00:01.8                    ┃
┃  Duration: 1.5s                   ┃
┃  Impacts: 3                        ┃
┃                                    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

---

## ⚙️ Параметры подсчета

| Параметр | Значение | Назначение |
|----------|----------|-----------|
| `points_to_win_game` | 11 | Очков нужно для победы в игре |
| `min_point_difference` | 2 | Минимальная разница между игроками |
| `games_to_win_match` | 2 | Игр нужно для победы в матче |
| `rally_timeout` | 0.5 сек | Максимально время без удара |

---

## 🧮 Алгоритм подсчета

```python
def process_match(impacts):
    """Главный цикл подсчета матча"""
    
    score = {'player1': 0, 'player2': 0}
    games = {'player1': 0, 'player2': 0}
    rallies = []
    
    current_rally = []
    
    for impact in impacts:
        # 1. Добавляем удар в текущий rally
        current_rally.append(impact)
        
        # 2. Проверяем конец rally (ошибка одного из игроков)
        if is_rally_ended(current_rally, impacts):
            rally_winner = determine_rally_winner(current_rally)
            rallies.append({
                'winner': rally_winner,
                'impacts': current_rally
            })
            
            # 3. Добавляем очко
            if rally_winner == 'Player1':
                score['player1'] += 1
            else:
                score['player2'] += 1
            
            # 4. Проверяем победу в игре
            game_winner = check_game_winner(score)
            if game_winner:
                games[f'{game_winner}'.lower()] += 1
                score = {'player1': 0, 'player2': 0}  # Reset
                
                # 5. Проверяем победу в матче
                match_winner = check_match_winner(games)
                if match_winner:
                    return {
                        'winner': match_winner,
                        'games': games,
                        'rallies': rallies
                    }
            
            current_rally = []
    
    return {'status': 'in_progress', 'games': games, 'rallies': rallies}
```

---

## 📚 Ссылки

- **IMPACT_DETECTION.md** - Как обнаруживаются удары
- **PARAMETERS.md** - Все параметры системы
- **ARCHITECTURE.md** - Архитектура системы
- **main_processor.py** - Главный скрипт обработки

---

**✨ Документ завершен**

