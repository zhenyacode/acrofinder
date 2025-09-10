import re
from typing import List, Set, Dict
import pandas as pd


class Scanner:
    """
    Осуществляет поиск акростихов в тексте

    Аргументы:
        min_word_sizes [int, int, ...]: скольким буквам нужно совпасть со словом из словаря, 
        чтобы нас заинтересовать 
        dictionary_path (str): путь к словарю в формате txt, 
        на основе которого будет вестись поиск
    """

    def __init__(self, min_word_sizes: List[int] = [5],
                 dictionary_path: str =""):
    
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
        # self.dictionary = self._load_dictionary(dictionary_path)
        self.dictionary = {'альфа', 'бета'}
        self.first_letters = []

        self.min_word_sizes = min_word_sizes
        self.n_dicts = self._load_n_dicts(self.dictionary, self.min_word_sizes)


    def scan_text(self, text: str, levels:List[str] = ['word']) -> pd.DataFrame:
        """
        Ищет всех кандидатов на акростихи в переданном тексте, 
        возвращает датафрейм с кандидатами (+ окрестности слева и справа), контекстом
        и адресом в тексте -- все уровни в одном датафрейме с соответствующим значением
        в столбце уровень

        
        Аргументы:
        text (str): текст, в котором производится поиск акростихов
        levels [str, str, ...]: набор уровней, на которых ищем акростихи (слова, предожения, абзацы)
        """

# TO DO: реализовать последующую фильтрацию найденных кандидатов, пытаясь достроить до
# full_word, чтобы отсечь побольше случайных совпадений

        valid_levels = ['paragraph', 'sentence', 'word']
        for level in levels:
            if level not in valid_levels:
                raise ValueError(f"Invalid level: {level}. Expected one of: 'paragraph', 'sentence', 'word'")


        self.text = text

        self.first_letters = {'paragraph': None,
                              'sentence': None,
                              'word': None}
                
        all_rows = []  # ← собираем ВСЕ строки здесь

        for level in levels:
            self.first_letters[level] = self._get_first_letters(text, level)


        for level in levels:
            rows = self._get_candidates(self.first_letters[level], level)
            all_rows.extend(rows)  # ← добавляем в общий список

        # Создаём ОДИН DataFrame в конце
        columns = ['id', 'n_gram_size', 'word', 'vicinity', 'context', 'level']
        return pd.DataFrame(all_rows, columns=columns) if all_rows else pd.DataFrame(columns=columns)



    def _get_first_letters(self, text: str, level: str) -> list[str]:
        """
        Проходит по тексту и возвращает первые буквы каждого объекта 
        соответствующего уровня (параграфы, предложения, слова)
        """

        very_first_letter = r'^[^A-Za-zА-Яа-яЁё\n]*[A-Za-zА-Яа-яЁё]|'
        found = re.findall(very_first_letter+self.PATTERNS[level], text)
        self.cache_results[level] = list(re.finditer(very_first_letter+self.PATTERNS[level], text))
        return [elem[-1].lower() for elem in found]


    def _get_candidates(self, first_letters: List[str], level: str) -> dict[str, str]:
        """
        Получает для первых букв список n-грамм всех размеров, заданных self.min_word_sizes,
        по каждому списку проходит, сравнивая n-граммы со словарными, и если есть совпадение,
        получает окрестности и контекст и включает соответствующее совпадение в результаты
        """

        rows = []
        for n_gram_size in self.min_word_sizes:
            n_grams = self._get_n_grams(n_gram_size, first_letters)
            n_dict = self.n_dicts[n_gram_size]
            for id in range(len(n_grams)):
                word = n_grams[id]
                if word in n_dict:
                    vicinity = self._get_vicinity(level, id, n_gram_size)
                    context = self._get_context(level, id, self.context_range)
                    rows.append({
                        'id': id,
                        'n_gram_size': n_gram_size,
                        'word': word,
                        'vicinity': vicinity,
                        'context': context,
                        'level': level
                    })
        return rows




    def _get_vicinity(self, level: str, id: int, n_gram_size: int) -> str:
        """
        Возвращает окрестности слева и справа для заданных позиций (т.е. для позиций в списке
        первых букв найденного кандидата в акростихи), если слева не хватает символов, заполняет их
        нижними подчёркиваниями
        """
        if id < self.vicinity_range:
            l_range = id
            r_range = self.vicinity_range
            left_dummy = '_' * (self.vicinity_range - id)
            
        else:
            l_range = self.vicinity_range
            r_range = self.vicinity_range
            left_dummy = ''

        vicinity = self.first_letters[level][id - l_range : id + n_gram_size + r_range]

        return left_dummy + ''.join(vicinity)


    def _get_context(self, level: str, id: int, range: int) -> str:
        """
        Из кэшированных результатов соответствующего поиска первых букв (объект re.Match, хранится в 
        self.cache_results[level][id], где id -- это порядковый номер n-граммы в списке, что 
        соответствует порядковому номеру первой буквы) получает позицию в исходном тексте и дальше 
        возвращает соответствующий слайс текста
        """
        position = self.cache_results[level][id].span()[0]

        context = self.text[position: position + range]

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


    def _get_n_gram_dict(self, dictionary: Set[str], n_gram_size: int) -> set[str]:
        """
        Принимает на вход множество слов и единственный размер n-граммы, 
        возвращает словарь n-грам, сделанных из слов длины >= n
        """

        n_dict = [word[0:n_gram_size] for word in dictionary if len(word) >= n_gram_size]
        return set(n_dict)


    def _get_n_grams(self, n_gram_size: int, letters: list[str]) -> list[str]:
        """
        Принимает на вход размер n-граммы, список букв, и составляет скользящим окном
        из букв n-граммы заданного размера
        """
        
        n_grams = []
        for i in range(len(letters)):
            n_grams.append(''.join(letters[i:i+n_gram_size]))
   
        return n_grams




# TO DO методы

    def _load_dictionary(self, dictionary_path) -> set[str]:
        """
        Загружает словарь из файла txt
        """ 

        pass
