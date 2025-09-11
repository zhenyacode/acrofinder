import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from acrofinder.scanner import Scanner


# Проверяем, что тестовый словарь никуда не делся
DICT_PATH = Path("data/dicts/test_dict.txt")

if not DICT_PATH.exists():
    raise FileNotFoundError(
        f"Тестовый словарь не найден: {DICT_PATH.resolve()}\n"
    )


def test_scan_text():
    """Проверяет, что объект Scanner создаётся, scan_text отрабатывает, 
    отдаёт pd.DataFrame с правильной структурой и находит акростих 
    'когда' в тестовом предложении (в тестовом словаре должно быть слово когда)."""

    s = Scanner(dictionary_name="test_dict.txt", min_word_sizes=[5])
    result = s.scan_text(text = 'Каждый охотник грозился достать аркебузу.',
                levels = ['word'])
    assert isinstance(result, pd.DataFrame)
    expected_columns = ['id', 'n_gram_size', 'word', 'vicinity', 'context', 'level']
    assert list(result.columns) == expected_columns
    assert 'когда' in result.word.values


def test_scan_text_invalid_level_raises():
    s = Scanner(dictionary_name="test_dict.txt", min_word_sizes=[5])
    with pytest.raises(ValueError):
        s.scan_text(text="текст", levels=["invalid_level"])


def test_scan_text_empty_text_returns_empty_df():
    s = Scanner(dictionary_name="test_dict.txt", min_word_sizes=[5])
    result = s.scan_text(text="", levels=["word"])
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
