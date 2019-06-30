## Perception Algorithm

A single perception can only be used to implement linearly separable functions. It takes both real and boolean inputs and associates a set of weights to them, along with a bias. And provide output in zero or one

For perception algorithm is used to classify the linear relation.
For instance we have number of student data set which contain pass and fail. And we want to classify them by a linear equation.
We take a random line equation to classify. Now we move the line to get better fit.

#### Steps:
We start with a random weight w and bias b and pass the value in equation wx + b
###### Note: first calculate the prediction for the input then subtract the prediction with actual output and consider the result 1 0r zero
1. For every misclassified point
    1. If the prediction is **zero**, that means positive point in negative area
        - Update weight and bias as follows
        - Where i = 1..n
        - Change w to w(i)+ax(i), where a is alpha that represent learning rate
        - Change b to b+a
    2. If the prediction is **1**, means negative point in positive area. 1.
   Update weight and bias same as previous but negative this time.
   
## Linear Regression

Reference Video:
[Liner Model](https://www.youtube.com/watch?v=l-Fe9Ekxxj4) 

Reference Video:
[Linear Regression in Pytorch way](https://www.youtube.com/watch?v=113b7O3mabY)

Reference article:
[NoteBook](https://towardsdatascience.com/introduction-to-machine-learning-algorithms-linear-regression-14c4e325882a)

Implement Linear Model
[Code](https://github.com/Iamsdt/MLNotes/blob/master/src/day1/LinearModel.ipynb)

## Gradient Descent

Reference article:
[NoteBook](https://towardsdatascience.com/introduction-to-machine-learning-algorithms-linear-regression-14c4e325882a)


Reference Video: [Gradient
Desecent](https://www.youtube.com/watch?v=b4Vyma9wPHo)

Reference Video:
[Udacity](https://www.youtube.com/watch?v=1j4bERmqmOU)

Implement Linear Model
[Code](https://github.com/Iamsdt/MLNotes/blob/master/src/day1/GradientDescent.ipynb)


#### Extra
How to Do Linear Regression using Gradient Descent:
[Siraj Raval](https://www.youtube.com/watch?v=XdM6ER7zTLk)

## Logistic Regression

Reference Video:
[Logistic Regression](https://www.youtube.com/watch?v=GAKTBQn7yKo)

Reference Video:
[Logistic Regression: Andrew Nag](https://www.youtube.com/watch?v=-la3q9d7AKQ)

Reference Video:
[Binary Logistic Regression: Siraj Raval](https://www.youtube.com/watch?v=H6ii7NFdDeg)

## Activation Function
Reference Video:
[Which Activation Function Should I Use?: Siraj Raval](https://www.youtube.com/watch?v=-7scQpJT7uo)

## Loss Function
Reference Video:
[Loss Functions Explained
: Siraj Raval](https://www.youtube.com/watch?v=IVVVjBSk9N0)

