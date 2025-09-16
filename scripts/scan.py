# scripts/scan.py

import argparse
from pathlib import Path
import pandas as pd

# Поднимаем путь, чтобы импортировать Scanner из src
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from acrofinder.scanner import Scanner
from acrofinder.batch_scanner import BatchScanner

def main():
    parser = argparse.ArgumentParser(
        description="Поиск акростихов в текстах по первым буквам слов, предложений или абзацев."
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Путь к директории с .txt файлами для сканирования"
    )
    
    parser.add_argument(
        "--dict", "-d",
        type=str,
        default="ru_curated.txt",
        help="Имя файла словаря в data/dicts/ (по умолчанию: ru_curated.txt)"
    )
    
    parser.add_argument(
        "--levels", "-l",
        type=str,
        nargs="+",
        choices=["word", "sentence", "paragraph"],
        default=["word"],
        help="Уровни поиска: word, sentence, paragraph (можно несколько, через пробел)"
    )
    
    parser.add_argument(
        "--minlen", "-m",
        type=int,
        nargs="+",
        default=[5],
        help="Минимальная длина n-граммы для поиска (можно несколько значений, например: -m 3 4 5)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default="results.csv",
        help="Путь для сохранения результатов (по умолчанию: results.csv)"
    )

    args = parser.parse_args()

    # Валидация входных данных
    if not args.input.exists():
        print(f"Ошибка: директория {args.input} не существует")
        exit(1)

    # Создаём сканер
    scanner = Scanner(min_word_sizes=args.minlen, dictionary_name=args.dict)
    batch_scanner = BatchScanner(scanner, args.input)

    # Запускаем сканирование
    print("🔍 Начинаем поиск акростихов...")
    results_df = batch_scanner.scan_directory(levels=args.levels)

    # Сохраняем результаты
    results_df.to_csv(args.output, index=False, encoding='utf-8')
    print(f"✅ Результаты сохранены в {args.output}")
    print(f"📊 Найдено кандидатов: {len(results_df)}")

    if len(results_df) == 0:
        print("⚠️  Предупреждение: ничего не найдено. Проверьте параметры или корпус.")

if __name__ == "__main__":
    main()