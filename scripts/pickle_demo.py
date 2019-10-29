# jak zapisywac i odczytywac przy uzyciu pickle
# wiecej info: https://www.journaldev.com/15638/python-pickle-example
# jeszcze wiecej info: https://wiki.python.org/moin/UsingPickle

import pickle  # import biblioteki pickle

# moje_oceny = {'Algebra 1': 3, 'Analiza 3': 3.5, 'Bazy danych': 5}  # slownik
matrix = [[1, 2, 3], [4, 5, 6]]

print(matrix)

# filename_oceny = 'oceny'
filename_macierz = 'macierz' # nazwano plik do ktorego zapiszemy macierz

# rozpoczecie zapisywanie do pliku
outfile = open(filename_macierz, 'wb')  # w oznacza zapisywanie do pliku, b - binary mode

# wrzucenie macierzy do pliku przy pomocy polecenia dump
pickle.dump(matrix, outfile)

# zakonczenie zapisywania do pliku
outfile.close()

# rozpoczecie czytania z pliku
infile = open(filename_macierz, 'rb')  # czytanie z pliku

# zaladowanie macierzy z pliku picklowego
matrix2 = pickle.load(infile)

# zakonczenie czytania z pliku
infile.close()

print(matrix2)
