# SKIPOD_4_Course
Курс по распределённым системам. Задания 1 и 2 вариант 14

#Как запускать
(macOS Big Sur)
Компиляция: mpicc <имя_файла>.c -o <имя_файла>
Запуск с количеством процессов <=4: 
    mpirun -np <кол-во процессов> <имя_файла>

Запуск с бОльшим количеством процессов (до 16-ти): 
    mpirun --filename host.txt -np <кол-во процессов> <имя_файла>


Возможно придется переопределить некоторую переменную окружения, в 
случае ошибки после выполнения программы:
    export OMPI_MCA_btl_vader_backing_directory=/tmp