# Dynamic Recurrent Neural Network with Homeostatic Plasticity

This is an implementation of homeostatic plasticity in continuous-time recurrent neural networks, following previous work by Williams and Noble <a href="https://www.sciencedirect.com/science/article/abs/pii/S0303264706001729?via%3Dihub">Homeostatic plasticity improves signal propagation in continuous-time recurrent neural networks</a> (2007). 

The main part of the code is contained within the CTRNN class. The rest of the classes are use to evolve the neural network to produce oscillations. 

We are using this to better understand the role that homeostatic plasticity plays in making neural networks more robust, flexible, and adaptive. This is work in collaboration with graduate student, Lindsay Stolting. 

## Instructions for use

1. Compile using the Makefile: 
```
$ make
```
2. Perform an evolutionary run (takes 2 seconds): 
```
$ ./main
```
3. Visualize the evolutionary progress and the resulting dynamics of the best evolved homeostatic neural circuit:
```
$ python viz.py
```




