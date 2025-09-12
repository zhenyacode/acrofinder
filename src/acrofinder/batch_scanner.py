from .scanner import Scanner
from pathlib import Path
import pandas as pd
from typing import List

from tqdm import tqdm


class BatchScanner:
    """
    Организовывает работу Scanner по всей заданной директории
    """

    def __init__(self, scanner: Scanner, directory_path: Path) -> None:

        self.scanner = scanner
        self.directory = directory_path

    def scan_directory(self, levels: List[str] = ['word']) -> pd.DataFrame:
        results = [] 
        
        files = list(self.directory.glob("*.txt"))

        for file_path in tqdm(files, desc="Processing files"):
            try:
                text = file_path.read_text(encoding='utf-8')
            except:
                text = file_path.read_text(encoding='windows-1251')
            df = self.scanner.scan_text(text, levels)
            df['source_file'] = file_path.name 
            results.append(df)
    
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

