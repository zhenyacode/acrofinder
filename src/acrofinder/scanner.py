import re
from typing import List, Set, Dict
import pandas as pd
from pathlib import Path

from dataclasses import dataclass, asdict

@dataclass
class AcrosticCandidate:
    start_pos: int
    n_gram_size: int
    word: str
    vicinity: str
    context: str
    level: str

    def to_dict(self):
        return asdict(self)


class Scanner:
    """
    Осуществляет поиск акростихов в тексте
    """

    def __init__(self, min_word_sizes: List[int] = [5],
                 dictionary_name: str ="") -> None:

        """
        Создаёт объект Scanner, инициализирует конфигурацию (vicinity-, context-, 
        addendum-range, паттерны поиска), загружает словарь, делает из него 
        словари n-грамм для более эффективного поиска

        Аргументы:
            min_word_sizes [int, int, ...]: скольким буквам нужно совпасть со словом из словаря, 
            чтобы сочетание букв попало в кандидаты в акростихи 
            dictionary_name (str): название файла со словарём, лежащего в папке dicts, 
            на основе которого будет вестись поиск
        """    
        self.PATTERNS = {'paragraph': r'(?<=\n)[^A-Za-zА-Яа-яЁё\n]*[A-Za-zА-Яа-яЁё]',
                         'sentence': r'(?<=\n)[^A-Za-zА-Яа-яЁё\n]*[A-Za-zА-Яа-яЁё]|(?<=[\.\!\?])[^A-Za-zА-Яа-яЁё\n]*[A-Za-zА-Яа-яЁё]',
                         'word': r'\b[A-Za-zА-Яа-яЁё]'}
    
        # Надо сделать кастомизируемым или минимум: прибить
        # гвоздями наиболее удобные настройки
        # self.vicinity_range = vicinity_range
        # self.context_range = context_range
        # self.addendum_range = addendum_range

        self.vicinity_range = 5
        self.context_range = 100 # вместо этого надо просто все слова/ предложения, 
                                # составляющие сочетание, возвращать
        self.addendum_range = 1


        self.cache_results = {}

        # пока заглушка вместо загрузки словаря
        self.dictionary = self._load_dictionary(dictionary_name)

        self.min_word_sizes = min_word_sizes
        self.n_dicts = self._load_n_dicts(self.dictionary, self.min_word_sizes)


    def scan_text(self, text: str, levels:List[str] = ['word']) -> pd.DataFrame:
        """
        Ищет все возможные акростихи в переданном тексте, возвращает датафрейм с 
        кандидатами (+ окрестности слева и справа) и контекстом в тексте 
        -- все уровни в одном датафрейме с соответствующим значением
        в столбце уровень

        Аргументы:
            text (str): текст, в котором производится поиск акростихов
            levels [str, str, ...]: набор уровней, на которых производится поиск (слова, 
            предложения, абзацы)

        Возвращает:
            results (pd.DataFrame): сводная таблица результатов поиска 
            (id / размер n-граммы / слово-кандидат / окрестности / контекст / уровень поиска)

        """

        # TO DO: реализовать последующую фильтрацию найденных кандидатов, пытаясь достроить до
        # full_word, чтобы отсечь побольше случайных совпадений

        valid_levels = ['paragraph', 'sentence', 'word']
        for level in levels:
            if level not in valid_levels:
                raise ValueError(f"Invalid level: {level}. Expected one of: 'paragraph', 'sentence', 'word'")


        all_candidates = []  


        for level in levels:
            candidates = self._get_candidates(text, level)
            all_candidates.extend(candidates)  

        # Создаём ОДИН DataFrame в конце
        columns = ['start_pos', 'n_gram_size', 'word', 'vicinity', 'context', 'level']
        results = pd.DataFrame([c.to_dict() for c in all_candidates], 
                               columns=columns) if all_candidates else pd.DataFrame(columns=columns)

        return results




    def _get_candidates(self, text: str, level: str) -> Dict[str, str]:
        """
        Получает для текста на соответствующем уровне (слова, предложения, парграфы) набор первых 
        букв, из которого получает список n-грамм всех размеров, заданных self.min_word_sizes,
        по каждому списку проходит, сравнивая n-граммы со словарными, и если есть совпадение,
        получает окрестности и контекст и включает соответствующее совпадение в результаты
        """

        first_letters, matches = self._get_first_letters_and_matches(text, level)

        candidates = []

        # для n-граммы каждого размера получаем кандидатов
        for n_gram_size in self.min_word_sizes:
            n_grams = self._get_n_grams(n_gram_size, first_letters)
            n_dict = self.n_dicts[n_gram_size]
            all_n_grams = range(len(n_grams))
            
            for id in all_n_grams:
                possible_word = n_grams[id]
                if possible_word in n_dict:
                    candidate = self._make_candidate(text, possible_word, level, 
                                                     first_letters, matches, id, 
                                                     n_gram_size)
                    candidates.append(candidate)
        
        return candidates


    def _make_candidate(self, text: str, word: str, level: str, 
                        first_letters: List[str], matches: List[re.Match],
                        id: int, n_gram_size: int) -> AcrosticCandidate:
        """
        Собирает на входящих параметрах из слова, окрестностей и контекста
        строчку про кандидата в соответствующей форме.
        """
        
        vicinity = self._get_vicinity(first_letters, id, n_gram_size)
        context = self._get_context(text, matches, id)

        start_position = matches[id].span()[0]

        candidate = AcrosticCandidate(start_pos=start_position,
                                      n_gram_size=n_gram_size,
                                      word=word,
                                      vicinity=vicinity,
                                      context=context,
                                      level=level)
        return candidate

    def _get_first_letters_and_matches(self, text: str, 
                                       level: str) -> tuple[List[str], List[re.Match]]:
        """
        Проходит по тексту и возвращает первые буквы каждого объекта 
        соответствующего уровня (параграфы, предложения, слова), а также
        соответствующие мэтч-объекты
        """

        very_first_letter = r'^[^A-Za-zА-Яа-яЁё\n]*[A-Za-zА-Яа-яЁё]|'
        pattern = very_first_letter+self.PATTERNS[level]
        matches = list(re.finditer(pattern, text))
        first_letters = [m.group()[-1].lower() for m in matches]

        return first_letters, matches


    def _get_vicinity(self, first_letters: List[str], id: int, n_gram_size: int) -> str:
        """
        Возвращает окрестности слева и справа для заданных позиций (т.е. для позиций в списке
        первых букв найденного кандидата в акростихи), если слева не хватает символов, заполняет их
        нижними подчёркиваниями
        """

        word_start = id
        word_end = id + n_gram_size

        left_vicinity = "".join(first_letters[:word_start])
        right_vicinity = "".join(first_letters[word_end:])

        if len(left_vicinity) < self.vicinity_range:
            left_dummy = '_' * (self.vicinity_range - len(left_vicinity))
            left_vicinity = left_dummy + left_vicinity

        if len(right_vicinity) < self.vicinity_range:
            right_dummy = '_' * (self.vicinity_range - len(right_vicinity))
            right_vicinity = right_dummy + right_vicinity

        word = "".join(first_letters[word_start:word_end])


        return "".join(left_vicinity) + "_" + word.upper() + "_" + "".join(right_vicinity)


    def _get_context(self, text:str, matches: List[re.Match], id: int) -> str:
        """
        Получает контекст по позиции найденного совпадения
        """
        position = matches[id].span()[0]

        context = text[position: position + self.context_range]

        return context


    def _load_n_dicts(self, dictionary: Set[str], n_gram_sizes: List[int]) -> Dict[int, Set[str]]:
        """
        Принимает на вход множество слов и возвращает словари n-грам, сделанных из слов длины >= n
        для каждого n из n_gram_sizes
        (оболочка над методом _get_n_gram_dict, обрабатывающим единственное значение n)
        """

        n_dicts = {}

        for n_gram_size in n_gram_sizes:
            n_dicts[n_gram_size] = self._get_n_gram_dict(self.dictionary, n_gram_size)

        return n_dicts


    def _get_n_gram_dict(self, dictionary: Set[str], n_gram_size: int) -> Set[str]:
        """
        Принимает на вход множество слов и единственный размер n-граммы, 
        возвращает словарь n-грам, сделанных из слов длины >= n
        """

        n_dict = [word[0:n_gram_size] for word in dictionary if len(word) >= n_gram_size]
        return set(n_dict)


    def _get_n_grams(self, n_gram_size: int, letters: list[str]) -> List[str]:
        """
        Принимает на вход размер n-граммы, список букв, и составляет скользящим окном
        из букв n-граммы заданного размера
        """
        
        n_grams = []
        for i in range(len(letters)):
            n_grams.append(''.join(letters[i:i+n_gram_size]))
   
        return n_grams


    def _load_dictionary(self, dictionary_name: str) -> Set[str]:
        """
        Загружает словарь из файла txt
        """ 

        path = self._get_dict_path(dictionary_name)
        with open(path, encoding='utf-8') as f:
            words = f.read().splitlines()


        return set(words)


    def _get_dict_path(self, filename: str) -> Path:
        """
        Делает из названия файла со словарём путь до словаря, по которому его можно
        загрузить
        """
        # Поднимаемся от src/acrofinder/ на два уровня — в корень проекта
        root_dir = Path(__file__).parent.parent.parent 
        dict_path = root_dir / "data" / "dicts" / filename

        if not dict_path.exists():
            raise FileNotFoundError(f"Dictionary file not found: {dict_path}")
        return dict_path
