# deep

![](http://cranlogs.r-pkg.org/badges/grand-total/deep)

> A small R library for creating deep neural networks.


# Installing

I am pleased to say that this package is on CRAN, so one can install it
by running `install.packages("deep")`.


# Using

Here is an example:

First load the package: `library(deep)`, then prepare a dataset, here we
used the classic *iris* dataset.

```
dataset <- iris
dataset$Petal.Length <- NULL
dataset$Petal.Width <- NULL
dataset <- dataset[dataset$Species != "versicolor",]
dataset$Code <- as.integer(dataset$Species == "virginica")
dataset <- dataset[sample(nrow(dataset)),]
```

Now that we have the data on the format we whant, let's create the net.

```
net <- neuralNet(2, perceptronLayer(1))
```

The `neuralNet` class contructor takes arguments that control the shape
of the net: the length of the input vector, this case, 2,
and the layers, which is just one layer with one perceptron neuron. Now
we can train the net.

```
net$train(dataset[,c(1,2)], dataset$Code, epochs = 5000)
```

Check if the accuracy is satisfying:

```
dataset$Calc <- sapply(1:nrow(dataset), function(x) net$output(dataset[x,c(1,2)]))
length(which(dataset$Code==dataset$Calc))/nrow(dataset)
```

You can train it for more epochs if needed, to get the method use the
function `output`:

```
net$output(c(1,2))
```

For more help consult the *man* folder or run `?neuralNet` on your console.


## Note to contributors

This project is in a very initial state and many features are missing:

- new layers, like dropout, pooling, convolution etc
- a better implementation of the backpropagation algorithm
- new training algorithms
- a plot method tho view all neurons and weights in a nice way
- a plot during training, just because it's nice
- anything you propose that makes sense

I intend to convert some methods using RCPP for better performance.
