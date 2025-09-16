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
                       save_results: bool = True) -> pd.DataFrame:
        """
        Сканирует все .txt файлы в директории, возвращает сводный DataFrame с кандидатами.
        """

        results = [] 
        total_chars = 0

        files = list(self.directory.glob("*.txt"))

        for file_path in tqdm(files, desc="Processing files"):
            try:
                text = file_path.read_text(encoding='utf-8')
            except:
                text = file_path.read_text(encoding='windows-1251')
            df = self.scanner.scan_text(text, levels, filter_by_neighbours, min_neighbour_len)
            df['source_file'] = file_path.name 
            results.append(df)

            total_chars += len(text)

        if save_results:
            scan_time = datetime.now()
            timestamp = int(scan_time.timestamp())
            csv_filename = f"{scan_time.strftime('%y%m%d')}_{timestamp}_results.csv"
            txt_filename = f"{scan_time.strftime('%y%m%d')}_{timestamp}_meta.txt"

            # Сохраняем CSV
            csv_path = self.output_dir / csv_filename
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"✅ Результаты сохранены: {csv_path.name}")

            # Генерируем и сохраняем TXT-отчёт
            scan_params = {
                "levels": levels,
                "filter_by_neighbours": filter_by_neighbours,
                "min_neighbour_len": min_neighbour_len,
            }
            report = self._generate_scan_report(
                scan_time=scan_time,
                files_processed=len(files),
                total_chars=total_chars,
                total_candidates=len(df),
                scan_params=scan_params
            )
            txt_path = self.output_dir / txt_filename
            txt_path.write_text(report, encoding='utf-8')
            print(f"📄 Отчёт сохранён: {txt_path.name}")


        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()




    def _generate_scan_report(self,
                              scan_time: datetime,
                              files_processed: int,
                              total_chars: int,
                              total_candidates: int,
                              scan_params: dict) -> str:
        """
        Генерирует УПРОЩЁННЫЙ текстовый отчёт о сканировании.
        """
        lines = []
        lines.append("📊 КРАТКИЙ ОТЧЁТ О СКАНИРОВАНИИ")
        lines.append("=" * 50)
        lines.append(f"📅 Дата и время:     {scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"📂 Директория:       {self.directory.resolve()}")
        lines.append("")
        lines.append("⚙️  ПАРАМЕТРЫ:")
        for key, value in scan_params.items():
            lines.append(f"   • {key:<20} {value}")
        lines.append("")
        lines.append(f"📁 Файлов обработано: {files_processed}")
        lines.append(f"📝 Символов всего:    {total_chars:,}".replace(",", " "))
        lines.append(f"🎯 Кандидатов найдено: {total_candidates}")
        lines.append("")
        lines.append("✅ Готово.")

        return "\n".join(lines)