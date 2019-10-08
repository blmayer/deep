#' deep: A Neural Networks Framework
#'
#' The deep package provides classes for layers, types of neurons and the
#' neural network as a whole.
#'
#' @docType package
#' @name deep
NULL


#' The main NeuralNetwork class, that holds the layers.
#'
#' @field eta The learning tax, representes the size of the weight adjustment
#' between each epoch of training.
#'
#' @field layers This field is a list of the layers of the network, you can use
#' subsetting to inspect them.
#'
neuralNet <- setRefClass(
    "NeuralNetwork",
    fields = list(eta = "numeric", layers = "vector")
)

#' @method initialize This runs when you instantiate the class, either with
#' \code{neuralNet$new()} or \code{neuralNet()}, the intent is to set up the
#' internal fields and layers passed in ... .
#'
#' @method compute Gives the output of passing data inside the network.
#' @param input The actual data to be fed into the network, this input's
#' dimentions vary with the first layer in your network.
#' @return Vector of the outputs of the last layer in your network.
#'
#' @method train Run the backpropagation algorithm to train all layers of
#' the network
#' @param ins The list of vectors of inputs to the first layer in the network
#' @param outs The list of vectors of outputs of the last layer in the network
#' @param epochs How many rounds of training to run
#' @param tax This is the learning rate, aka eta
#' @param maxErr A contition to early stop the training process
#' @return Vector of computed values of the same size of the last layer
#'
neuralNet$methods(
    initialize = function(input, ...) {
        # Initialize each layer
        layers <<- vector("list", ...length())
        for (i in 1:...length()) {
            l <- ...elt(i)
            switch (class(l),
                "PerceptronLayer" = {
                    layers[[i]] <<- perceptronLayer(l$n, input)
                    input <- l$n
                },
                "McCullochPittsLayer" = {
                    layers[[i]] <<- mcCullochPittsLayer(l$n, input)
                    input <- l$n
                },
                stop("argument in ... is not a layer!")
            )
        }
    },
    compute = function(input) {
        for (layer in layers) {
            input <- layer$output(input)
        }
        input
    },
    train = function (ins, outs, epochs = 1, tax = .01, maxErr = 0) {
        for (i in 1:ins) {
            temps <- list(ins[[i]])
            for (l in 1:length(layers)) {
                temps[[l+1]] <- layers[[l]]$output(temps[[l]])
            }

            # Train each layer with the recorded input and output
            err <- sum(outs[i]^2 - temps[[l+1]]^2)/2
            temps[[l+1]] <- outs[i]
            for (l in length(layers):1) {
                layers[[l]]$train(temps[[l]], temps[[l+1]])

                # Update error
                for (o in 1:length(temps[[l]])) {
                    for (n in 1:layers[[l]]$n) {
                        d <- err*layers[[l]]$neurons[[n]]$ws[[o]]
                        temps[[l]][o] <- temps[[l]][o] + d
                    }
                }
            }
        }
    }
)
