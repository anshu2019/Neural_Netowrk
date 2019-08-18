# Neuralnet

This code implements a Neural Network from scratch to label hand written text.

Datasets We will be using a subset of an Optical Character Recognition (OCR) dataset. This data includes
images of all 26 handwritten letters; our subset will include only the letters “a,” “e,” “g,” “i,” “l,” “n,” “o,”
“r,” “t,” and “u.”

Model specifications -
1. single hidden layer
2. Sigmoid activation
3. Softmax output
4. Backpropagation 

approach for SGD
procedure SGD(Training data D, test data Dt)
2: Initialize parameters ;  . Use either RANDOM or ZERO from Section 2.2.1
3: for e 2 f1; 2; : : : ;Eg do . For each epoch
4: for (x; y) 2 D do . For each training example
5: Compute neural network layers:
6: o = object(x; a; b; z; ^y; J) = NNFORWARD(x; y;; )
7: Compute gradients via backprop:
8:
g = rJ
g = rJ
)
= NNBACKWARD(x; y;; ; o)
9: Update parameters:
10:     􀀀 
g
11:     􀀀 
g
12: Evaluate training mean cross-entropy JD(; )
13: Evaluate test mean cross-entropy JDt(; )
14: return parameters ; 


