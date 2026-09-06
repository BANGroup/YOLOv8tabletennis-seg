# Scoring Capsule — Подсчёт очков в настольном теннисе

## Scope
**Трогаем:** подача, двойной отскок, сетка, аут, счёт
**НЕ трогаем:** детекция ударов (tracking_capsule), YOLO модель (кроме маски Net для зоны), визуализация

## Invariants (≤12)

1. **Player 1 = левая сторона** (X < center_x) — SSoT: `scripts/serve_detector.py` (`table_side`)
2. **Player 2 = правая сторона** (X >= center_x) — SSoT: `scripts/serve_detector.py`
3. **Смена подачи каждые 2 очка** — SSoT: `serve_change_interval`, `points_since_serve_change`
4. **Подача = пауза > 2.5 сек + подброс + удар на своей стороне** — SSoT: `serve_pause_threshold = 2.5`
5. **Двойной отскок = 2+ удара на одной стороне, интервал >= 0.6 сек** — SSoT: `min_double_bounce_interval`, `prev_impact_time`
6. **Очко за двойной отскок — противнику, отложенно** — SSoT: `pending_double_bounce`, `_award_deferred_double_bounce()` в конце розыгрыша
7. **Контакт с сеткой во время розыгрыша** — `is_impact_on_net()` (`NET_IMPACT_MARGIN=25` px) → `register_net_contact()` → `pending_net_contact`; очко не сразу — SSoT: `track_ball()`, `resolve_pending_net_contact()`
8. **Сетка при пропадании мяча** — `is_ball_in_net_zone()` (`dist >= -40` px) + `ball_lost` — SSoT: `_detect_rally_end_event()`
9. **Timeout pending net** — `NET_PENDING_MAX_AGE = 2.0` с, затем `_award_net_fault()` — SSoT: `_detect_rally_end_event()`
10. **Виновник сетки/аута** — сторона последнего отскока (`last_rally_impacts[-1].table_side`), fallback X — SSoT: `_fault_player_from_last_bounce()`, `_award_net_fault()`
11. **`pending_net_contact` / `pending_double_bounce` очищаются** при новом розыгрыше и ранних выходах — SSoT: `process_detection()`, `_detect_rally_end_event()`
12. **Аут = мяч далеко от центра + горизонтальная скорость > 10** — SSoT: `_detect_rally_end_event()`; off-table fallback — `is_ball_out_of_table()`

## Decisions (≤7)

| Решение | Обоснование |
|---------|-------------|
| Комбинированная подача (2/3) | Один признак ненадёжен — SSoT: `process_detection()` |
| Отложенный net contact | Контакт при отскоке не закрывает розыгрыш сразу — SSoT: `register_net_contact()`, `resolve_pending_net_contact()` |
| `NET_IMPACT_MARGIN=25` vs zone `-40` px | Узкий порог для impact, широкий для ball_lost — SSoT: `is_impact_on_net()`, `is_ball_in_net_zone()` |
| Аут по last bounce, не по X мяча | Текущая позиция мяча вводит в заблуждение — SSoT: `_fault_player_from_last_bounce()` |
| Отложенный double bounce | Очко в конце розыгрыша; `discard_stale_double_bounce()` если мяч ушёл на сторону соперника — SSoT: `_award_deferred_double_bounce()` |
| Удалены `speed < 15` и `moving_towards_center` | Блокировали реальные net-fault — SSoT: `_detect_rally_end_event()` |
| `prev_impact_time` до перезаписи | Корректный интервал двойного отскока — SSoT: `process_detection()` |

## Impact Paths

```
serve_detector.py
├── BallTracker.track_ball()
│   ├── impact on net → register_net_contact() → pending_net_contact
│   ├── next bounce → resolve_pending_net_contact() / discard_stale_double_bounce()
│   ├── missed_frames > 5 → _detect_rally_end_event()
│   │   ├── pending net age ≤ 2.0 s → _award_net_fault()
│   │   ├── is_ball_in_net_zone(ball_lost) → net fault
│   │   └── horizontal_speed > 10 + far_from_center → out (_fault_player_from_last_bounce)
│   └── ball lost off-table → out fallback
├── SmartImpactDetector.process_detection()
│   ├── Подача — serve_signals >= 2
│   ├── Двойной отскок → pending_double_bounce (не немедленный award)
│   └── Новый розыгрыш → clear pending_*
└── end rally (missed_frames) → _award_deferred_double_bounce()
    └── award_point() — event_time из bounce/pending time
```

## Validation Proof

Проверить на видео IMG_0291, сегмент 45–180 s, train14_ttseg_v3 (`docs/CHANGELOG.md` [06-09-2026]):

1. Счёт **8:6** (было 6:9); SMART impacts **85** (было 66)
2. Подтверждённые очки: t=52.3 s (P1), 65.1 s (P2), 74.1 s (P1)
3. `smart_impacts_analysis.json`: `point_history`, `net_events`, `out_events`, `double_bounce_events`, `score`, `serve_count`
4. Артефакт: `results/serve_detector_train14_img0291_consecfix/tracked_v5.mp4`

## Risks (≤7)

| Риск | Митигация |
|------|-----------|
| Сетка t≈109 s не даёт очко | Зафиксирован gap; калибровка net detection — open question |
| Очко t≈169.2 s теряется (нет аут-closure) | Розыгрыш тянется до double bounce t≈176 s |
| `min_frame_gap` склеивает отскоки (6 vs 1 кластер) | План: side-dependent gap |
| Ложный двойной отскок | min_interval=0.6 с + discard при смене стороны |
| Ложный аут при падении на пол | Требуем высокую горизонтальную скорость |
| Ловля мяча руками | Не моделируется — вне scope |
| Partial net recall | pending + ball_lost dual path |

## Open Questions (≤7)

1. Почему net t≈109 s не начисляет очко — порог margin или timing pending?
2. Как закрывать розыгрыш аутом при t≈169.2 s вместо отложенного double bounce?
3. Side-dependent `min_frame_gap` — параметры по left/right?
4. Нужна ли детекция let (сетка при подаче)?
5. Как отличить аут от внеигровых моментов?
6. Моделировать ли ловлю мяча руками как нарушение?

## Pointers (≤15)

- `scripts/serve_detector.py:500` — `register_net_contact()`
- `scripts/serve_detector.py:521` — `resolve_pending_net_contact()`
- `scripts/serve_detector.py:535` — `discard_stale_double_bounce()`
- `scripts/serve_detector.py:548` — `_award_net_fault()`
- `scripts/serve_detector.py:563` — `_award_deferred_double_bounce()`
- `scripts/serve_detector.py:576` — `_fault_player_from_last_bounce()`
- `scripts/serve_detector.py:929` — `is_ball_in_net_zone()` (`dist >= -40`)
- `scripts/serve_detector.py:948` — `is_impact_on_net()` (`NET_IMPACT_MARGIN=25`)
- `scripts/serve_detector.py:702` — `NET_IMPACT_MARGIN`, `NET_PENDING_MAX_AGE=2.0`
- `scripts/serve_detector.py:1123` — `_detect_rally_end_event()`
- `scripts/serve_detector.py:294` — `process_detection()` (serve, pending double)
- `scripts/serve_detector.py:588` — `award_point()` (event_time)
- `scripts/serve_detector.py:1453` — `--debug-trace` CLI
- `docs/CHANGELOG.md` — consecfix [06-09-2026]
- `results/serve_detector_train14_img0291_consecfix/` — validation output
