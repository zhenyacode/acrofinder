# ТУТ КОД ИЗ ПРОШЛОЙ РАЗРАБОТКИ

class CandidateParser():

    def __init__(self, dictionary: list[str], n_gram_sizes: list[int], 
                 vicinity_range: int, context_range: int, addendum_range: int) -> None:
        self.dictionary = dictionary
        self.n_gram_sizes = n_gram_sizes
        self.vicinity_range = vicinity_range
        self.context_range = context_range
        self.addendum_range = addendum_range

        self.cache_results = {}

        self.n_dicts = {}
        for n_gram_size in self.n_gram_sizes:
            self.n_dicts[n_gram_size] = self.__get_n_gram_dict(self.dictionary, n_gram_size)

        self.patterns = {'paragraph': r'(?<=\n)[^A-Za-zА-Яа-яЁё\n]*[A-Za-zА-Яа-яЁё]',
                         'sentence': r'(?<=\n)[^A-Za-zА-Яа-яЁё\n]*[A-Za-zА-Яа-яЁё]|(?<=[\.\!\?])[^A-Za-zА-Яа-яЁё\n]*[A-Za-zА-Яа-яЁё]',
                         'word': r'\b[A-Za-zА-Яа-яЁё]'}


    def __get_n_gram_dict(self, dictionary: list[str], n_gram_size: int) -> set[str]:
        n_dict = [word[0:n_gram_size] for word in dictionary if len(word) >= n_gram_size]
        return set(n_dict)
    

    def __get_first_letters(self, text: str, mode: str) -> list[str]:
        very_first_letter = r'^[^A-Za-zА-Яа-яЁё\n]*[A-Za-zА-Яа-яЁё]|'
        # very_first_letter = ''
        found = re.findall(very_first_letter+self.patterns[mode], text)
        self.cache_results[mode] = list(re.finditer(very_first_letter+self.patterns[mode], text))
        return [elem[-1].lower() for elem in found]

    
    def __get_candidates(self, letters: list[str], mode: str) -> dict[str, str]:
        
        candidates = {}

        for n_gram_size in self.n_gram_sizes:
            n_grams = self.__get_n_grams(n_gram_size, letters)
            n_dict = self.n_dicts[n_gram_size]

            for id in range(len(n_grams)):
                word = n_grams[id]
                if word in n_dict:
                    vicinity = self.__get_vicinity(mode, id, n_gram_size)
                    context = self.__get_context(mode, id, self.context_range)
                    candidates[str(id)+'-'+str(n_gram_size)] = [word, vicinity, context]
            
        return candidates


    def __get_n_grams(self, n_gram_size: int, letters: list[str]) -> list[str]:
        n_grams = []
        for i in range(len(letters)):
            n_grams.append(''.join(letters[i:i+n_gram_size]))
   
        return n_grams


    def __get_vicinity(self, mode: str, id: int, n_gram_size: int) -> str:
        
        if id < self.vicinity_range:
            l_range = id
            r_range = self.vicinity_range
            left_dummy = 'x' * (self.vicinity_range - id)
            
        else:
            l_range = self.vicinity_range
            r_range = self.vicinity_range
            left_dummy = ''

        vicinity = self.first_letters[mode][id - l_range : id + n_gram_size + r_range]

        return left_dummy + ''.join(vicinity)


    def __get_context(self, mode: str, id: int, range: int) -> str:
        
        position = self.cache_results[mode][id].span()[0]

        context = self.text[position: position + range]

        return context


    def find_candidates(self, text: str, levels: list[str]) -> dict[str, dict]:

        self.text = text

        self.first_letters = {'paragraph': None,
                              'sentence': None,
                              'word': None}
                
        for level in levels:
            self.first_letters[level] = self.__get_first_letters(text, level)

        results = {}

        for level in levels:
            letters = self.first_letters[level]
            results[level] = self.__get_candidates(letters, level)

        self.text = None

        return results
    


    def check_full_words(self, candidates: dict[str, dict]) -> dict[str, dict]:
        
        full_words = {}

        print('ПРОВЕРКА ПОЛНЫХ СЛОВ')

        for file in candidates:

            print(f'Проверяю {file}, {list(candidates.keys()).index(file)+1} из {len(candidates)}')

            full_words[file] = {}
            for mode in candidates[file]:
                full_words[file][mode] = {}
                print('\nFull word check')
                print(f'{file}, {list(candidates.keys()).index(file)+1} из {len(candidates)}, режим {mode}')


                for candidate in candidates[file][mode]:
                    n_gram = candidates[file][mode][candidate][0]
                    vicinity = candidates[file][mode][candidate][1]
                    context = candidates[file][mode][candidate][2]

                    right = vicinity[self.vicinity_range + len(n_gram):]
                    best_candidate = None

                    for addendum in range(0, self.addendum_range):
                        current_candidate = n_gram + right[:addendum]
                        if current_candidate in self.dictionary:
                            best_candidate = current_candidate
                    
                    if best_candidate != None:
                        if best_candidate not in full_words[file][mode].keys():
                            full_words[file][mode][best_candidate] = [best_candidate, vicinity, context]
                        elif context == full_words[file][mode][best_candidate][2]:
                            pass
                        else:
                            i = 0
                            while True:
                                new_name = best_candidate + '-' + str(i)
                                if new_name not in full_words[file][mode].keys():
                                    full_words[file][mode][new_name] = [best_candidate, vicinity, context]
                                    break
                                i += 1
    
        return full_words


    def check_vicinity(self, candidates: dict[str, dict], min_vicinity_word_size: int, mode: str = 'full') -> dict[str, dict]:
        # проверка: radiant -- расходясь влево и вправо от кандидата
        # full -- просто проверяем все окрестности вокруг кандидата в любых вариантах
        # возможно, режимы излишни, пока реализую только фулл
        # UPDATE: реализовываю только радиант, потому что с фуллом много слишком


        found_acros = {}

        for file in candidates:

            found_acros[file] = {}

            for mode in candidates[file]:
                print(f'\nVICINITY check {file}, {list(candidates.keys()).index(file)+1} из {len(candidates)}')
                print(f'режим {mode}')

                found_acros[file][mode] = {}

                for candidate in candidates[file][mode]:
                        
                    central_word = candidates[file][mode][candidate][0]
                    vicinity = candidates[file][mode][candidate][1]
                    context = candidates[file][mode][candidate][2]

                    left_vicinity = vicinity[:self.vicinity_range]
                    right_vicinity = vicinity[self.vicinity_range+len(central_word):]
                    
                    left_foundings = []
                    right_foundings = []
                    
                    for n_gram_start in range(self.vicinity_range-min_vicinity_word_size, 0, -1):
                        n_gram_end = self.vicinity_range
                        n_gram = left_vicinity[n_gram_start:n_gram_end]
                        if len(n_gram) >= min_vicinity_word_size:
                            if n_gram in self.dictionary and len(n_gram) > 0:
                                left_foundings.append(n_gram)                    
                    
                    for n_gram_end in range(min_vicinity_word_size, self.vicinity_range):
                        n_gram_start = 0
                        n_gram = right_vicinity[n_gram_start:n_gram_end]
                        if len(n_gram) >= min_vicinity_word_size:
                            if n_gram in self.dictionary and len(n_gram) > 0:
                                right_foundings.append(n_gram)
                    
                    if len(left_foundings) + len(right_foundings) > 0:
                        left = '-'.join(left_foundings)
                        right = '-'.join(right_foundings)
                        found_acros[file][mode][central_word] = [left+'-'+central_word+'-'+right, vicinity, context]
    
        return found_acros