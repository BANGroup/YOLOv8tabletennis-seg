#!/usr/bin/env python3
"""beforeSubmitPrompt hook: инжектирует governance-напоминание при каждом промпте.

Prompt reinforcement — модель получает напоминание о запрете кодинга
в начале каждого цикла. Не блокирует, только усиливает policy.

КАСТОМИЗАЦИЯ: измените AGENT_MESSAGE под зоны вашего проекта.
"""
import json
import sys

# КАСТОМИЗАЦИЯ: укажите зоны кодеров вашего проекта
AGENT_MESSAGE = (
    "НАПОМИНАНИЕ GOVERNANCE: "
    "Оркестратор НЕ пишет код в зонах кодеров. "
    "Для изменений в коде — используй Task tool "
    "для делегирования соответствующему кодеру. "
    "Перед каждым Write/StrReplace/Shell проверяй путь. "
    "Agent KG ведется write-through: dispatch -> phase_start -> --write-handoff -> --close-workflow. "
    "Перед закрытием workflow проверь, что handoff governance/docs содержат trace_id и plan_step_id. "
    "Перед sync domain graph всегда запроси ApprovalGate у человека (с W_<workflow_id>) и policy verdict. "
    "High-risk закрывается только после governance-compliance и docs-curator."
)


def main() -> None:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        payload = {}

    result = {
        "continue": True,
        "agentMessage": AGENT_MESSAGE,
    }

    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
