from pathlib import Path
import sys

LESSON_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(LESSON_PATH.parents[2]))
from s8_submit import submit as _submit


def submit(t_code, rlz_file='realization.py'):
    return _submit(t_code, rlz_file, str(LESSON_PATH))


if __name__ == '__main__':
    raise SystemExit(submit('de08030801'))
