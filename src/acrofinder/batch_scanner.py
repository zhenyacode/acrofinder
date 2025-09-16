from .scanner import Scanner
from pathlib import Path
import pandas as pd
from typing import List, Optional
from datetime import datetime


from tqdm import tqdm


class BatchScanner:
    """
    Организовывает работу Scanner по всей заданной директории
    """

    def __init__(self, scanner: Scanner, 
                 directory_path: Optional[Path] = None,
                 output_dir: Optional[Path] = None) -> None:
        """
        Инициализирует BatchScanner.

        Аргументы:
            scanner (Scanner): экземпляр сканера для поиска акростихов.
            directory_path (Path, optional): путь к директории с текстами (.txt). 
                По умолчанию: <project_root>/data/texts.
            output_dir (Path, optional): директория для сохранения результатов. 
                По умолчанию: <project_root>/results. Создаётся, если не существует.
        """
        self.scanner = scanner

        # Определяем корень проекта (поднимаемся от src/acrofinder/ на 2 уровня)
        project_root = Path(__file__).parent.parent.parent

        # Устанавливаем директорию текстов по умолчанию
        if directory_path is None:
            directory_path = project_root / "data" / "texts"

        # Проверяем, что директория с текстами существует
        if not directory_path.exists() or not directory_path.is_dir():
            raise FileNotFoundError(f"Директория с текстами не найдена: {directory_path}")

        self.directory = directory_path

        # Устанавливаем директорию результатов по умолчанию
        if output_dir is None:
            output_dir = project_root / "results"

        # Создаём директорию результатов, если её нет
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

        self.scanner = scanner
        self.directory = directory_path



    def scan_directory(self, levels: List[str] = ['word'], 
                       filter_by_neighbours: bool = False, 
                       min_neighbour_len: int = 1, 
                       save_file: bool = True) -> pd.DataFrame:
        """
        Сканирует все .txt файлы в директории, возвращает сводный DataFrame с кандидатами.
        """

        results = [] 
        
        files = list(self.directory.glob("*.txt"))

        for file_path in tqdm(files, desc="Processing files"):
            try:
                text = file_path.read_text(encoding='utf-8')
            except:
                text = file_path.read_text(encoding='windows-1251')
            df = self.scanner.scan_text(text, levels, filter_by_neighbours, min_neighbour_len)
            df['source_file'] = file_path.name 
            results.append(df)

            if save_file:
                now = datetime.now()
                filename = f"results_{now.strftime('%y%m%d')}_{int(now.timestamp())}.csv"
                
                output_path = self.output_dir / filename
                df.to_csv(output_path, index=False, encoding='utf-8')



# ДОБАВИТЬ СОХРАНЕНИЕ В ДИРЕКТОРИЮ РЕЗУЛЬТАТОВ

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

