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
        для поиска по строчкам стихотворения). Все необходимые уровни нужно ввести
        через пробел (например --level word paragraph)
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
        "--neighbours", "-n",
        type=bool,
        default=False,
        help="""
        Фильтровать возможные акростихи по наличию соседей: если True, то оставлять 
        в результатах только те сочетания первых букв, складывающиеся в слова, у 
        которых слева или справа также есть сочетания, складывающиеся в слова 
        минимальной длины min_neighbour_len
        """
    )

        
    parser.add_argument(
        "--minneighbourlen", "-mn",
        type=int,
        default=2,
        help="""
        Если включена фильтрация по наличию слов среди соседей найденной формы, 
        минимальная длина соседей (например, чтобы можно было исключать одно- 
        и двухбуквенные слова, попадающиеся случайно). Значение по умолчанию 2
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

    scanner = Scanner(min_word_size=args.minlen,
                      vicinity_range=args.vicinity,
                      dictionary_name=args.dict,
                      custom_dict_search=args.custom_dict)
    batch_scanner = BatchScanner(scanner, args.input)

    batch_scanner.scan_directory(levels=args.levels, 
                                 filter_by_neighbours=args.neighbours, 
                                 min_neighbour_len=args.minneighbourlen)

if __name__ == "__main__":
    main()