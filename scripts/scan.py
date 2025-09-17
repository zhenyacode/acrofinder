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
        description="Поиск акростихов в текстах по первым буквам слов, предложений или абзацев.",
        add_help=False
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
        default="wordforms_20k.txt",
        help="Имя файла словаря в data/dicts/ (по умолчанию: wordforms_20k.txt)"
    )
    
    parser.add_argument(
        "--levels", "-l",
        type=str,
        nargs="+",
        metavar="LEVEL",
        default=["sentence"],
        help="""
        Уровни поиска: word (акростих, образованный первыми буквами каждого слова),
        sentence (акростих, образованный первыми буквами предложений), 
        paragraph (акростих, образованный первыми буквами абзацев -- также подходит 
        для поиска по строчкам стихотворения)
        """
    )

    parser.add_argument(
        "--vicinity", "-v",
        type=int,
        default=5,
        help="""
        Размер окрестностей: сколько символов слева и справа от найденного сочетания
            показывать в таблице результатов в поле vicinity (по умолчанию 5)
        """
    )

    parser.add_argument(
        "--custom_dict", "-c",
        type=str,
        default=None,
        help="""
        Заданный набор слов, который нужно найти среди акростихов
        """
    )

    parser.add_argument(
        "--minlen", "-m",
        type=int,
        default=5,
        help="Минимальная длина сочетания, образующего акростих (по умолчанию 5 символов)"
    )

    parser.add_argument(
        '-h', '--help',
        action='help',
        default=argparse.SUPPRESS,
        help='Показать эту справку и выйти'
    )

    args = parser.parse_args()

    # Валидация входных данных
    if not args.input.exists():
        print(f"Ошибка: директория {args.input} не существует")
        exit(1)

    # Создаём сканер
    scanner = Scanner(min_word_size=args.minlen,
                      vicinity_range=args.vicinity,
                      dictionary_name=args.dict,
                      custom_dict_search=args.custom_dict)
    batch_scanner = BatchScanner(scanner, args.input)

    # Запускаем сканирование
    # print("🔍 Начинаем поиск акростихов...")
    batch_scanner.scan_directory(levels=args.levels)

    # # Сохраняем результаты
    # results_df.to_csv(args.output, index=False, encoding='utf-8')
    # print(f"✅ Результаты сохранены в {args.output}")
    # print(f"📊 Найдено кандидатов: {len(results_df)}")

if __name__ == "__main__":
    main()