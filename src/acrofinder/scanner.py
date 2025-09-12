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

        self.cache_results = {}

        # пока заглушка вместо загрузки словаря
        self.dictionary = self._load_dictionary(dictionary_name)
        # TODO здесь потенциал оптимизации -- если длинных слов слишком мало
        # то проверять все n-граммы, добивая их до max_word_length малоосмысленно
        self.max_word_length = len(max(self.dictionary, key=len))
        # print(f'{self.max_word_length = }') # в 20к словаре было  19
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




    def _get_candidates(self, text: str, level: str) -> List[AcrosticCandidate]:
        """
        Формирует список слов-кандидатов из последовательности первых букв элементов текста
        на заданном уровне (слова, предложения или абзацы).

        Алгоритм:
          1. Извлекает первые буквы всех единиц указанного уровня.
          2. Строит из них n-граммы длиной из self.min_word_sizes.
          3. Для каждой n-граммы проверяет:
             - если она соответствует префиксу слов словаря, то постепенно расширяет её,
               добавляя следующие буквы;
             - на каждом шаге фиксирует слово как кандидата, если оно есть в словаре.
          4. В результате собирает все подходящие слова (как короткие, так и удлинённые),
             с указанием их позиции, «окрестности» и текстового контекста.

        Таким образом, из первых букв сочетания «рыбаки» будут выделены кандидаты «рыб», «рыба»,
        «рыбак» и «рыбаки» (при условии, что такие слова есть в словаре).

        Возвращает:
            List[AcrosticCandidate]: найденные слова-кандидаты и сопутствующая информация.
        """

        text = self._normalize_text(text)
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
                    
                    n_addenda = range(self.max_word_length - len(possible_word))
                    
                    for _ in n_addenda:
                        if possible_word in self.dictionary:
                            candidate = self._make_candidate(text, possible_word, level, 
                                                        first_letters, matches, id, 
                                                        len(possible_word))
                            candidates.append(candidate)

                        last_len = len(possible_word)
                        possible_word = self._add_letter(first_letters, id, possible_word)
                        if last_len == len(possible_word):
                            break


        return candidates


    def _normalize_text(self, text: str) -> str:
        text = self._normalize_spaced_letters(text)
        text = self._normalize_hyphenated_letters(text)

        return text


    def _normalize_spaced_letters(self, text: str) -> str:
        """
        Убирает из текста вот т а к у ю разрядку пробелами, чтобы исключить 
        ложные срабатывания при поиске кандидатов
        """
        pattern = re.compile(r'(?:[A-Za-zА-Яа-яЁё](?:\s{1,3}[A-Za-zА-Яа-яЁё]){2,})')

        def replacer(match: re.Match) -> str:
            return match.group(0).replace(" ", "")

        return pattern.sub(replacer, text)

    def _normalize_hyphenated_letters(self, text: str) -> str:
        """
        Убирает из текста вот т-а-к-о-й вариант написания, чтобы исключить
        ложные срабатывания
        """

        pattern = re.compile(r'(?:[A-Za-zА-Яа-яЁё](?:[-–—]{1,3}[A-Za-zА-Яа-яЁё]){2,})')

        def replacer(match: re.Match) -> str:
            return re.sub(r'[-–—]', '', match.group(0))  # убрать все виды дефисов

        return pattern.sub(replacer, text)


    def _add_letter(self, first_letters: List[str], id: int, possible_word: str) -> str:
        """
        Возвращает потенциальное слово с прибавленной следующей буквой из first_letters 
        из [р, ы, б, а, к, и] и рыба вернет рыбак 
        """

        next_letter = ""

        if id+len(possible_word) < len(first_letters):
            next_letter = first_letters[id+len(possible_word)]

        return possible_word + next_letter 


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

        left_vicinity = first_letters[:word_start][-1*(n_gram_size):]
        right_vicinity = first_letters[word_end:][:(n_gram_size)]

        left_vicinity = "".join(left_vicinity)
        right_vicinity = "".join(right_vicinity)

        if len(left_vicinity) < self.vicinity_range:
            left_dummy = '_' * (self.vicinity_range - len(left_vicinity))
            left_vicinity = left_dummy + left_vicinity

        if len(right_vicinity) < self.vicinity_range:
            right_dummy = '_' * (self.vicinity_range - len(right_vicinity)-1)
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
