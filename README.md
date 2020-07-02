# dynamic-accuracies
Experiments with second- and third-order inexact optimization methods with 
dynamic inner accuracies.


To plot the graphs from the paper 
"Inexact Tensor Methods with Dynamic Accuracies" by N. Doikov and Yu. Nesterov
(https://arxiv.org/abs/2002.09403) 
use the following commands:

Figure 1:
$ python3 run_experiment_logreg.py
(It is required to download the files 
    "data/mushrooms.txt", 
    "data/w8a.txt", 
    "data/a8a.txt"
from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/ in advance).

Figure 2:
$ python3 run_experiment_logsumexp.py

Figures 3, 4:
$ python3 run_experiment_logsumexp_exact.py

Figure 5:
$ python3 run_experiment_logreg_exact.py
(It is required to download the files 
    "data/mushrooms.txt", 
    "data/w8a.txt", 
    "data/a8a.txt",
    "data/phishing.txt",
    "data/splice.txt"
from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/ in advance).

Figure 6:
$ python3 run_experiment_averaging.py

The results will be placed into "plots/*".
