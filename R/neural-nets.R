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
#' @method initialize This run when you instantiate the class, either with
#' \code{$new()} or \code{neuralNet()}, the intent is to set up the fields.
#'
#' @method compute Gives the output of passing data inside the network.
#' @param input The actual data to be fed into the network, this input's
#' dimentions vary with the first layer in your network.
#' @return Vector of the outputs of the last layer in your network.
#'
neuralNet <- setRefClass(
    "NeuralNetwork",
    fields = list(eta = "numeric", layers = "vector"),
    methods = list(
        initialize = function(input, ...) {
            # Initialize each layer
            layers <<- vector("list", ...length())
            for (i in 1:...length()) {
                switch (class(...elt(i)),
                        "PerceptronLayer" = {
                            layers[[i]] <<- perceptronLayer(...elt(i)$n, input)
                            input <- ...elt(i)$n
                        },
                        "McCullochPittsLayer" = {
                            layers[[i]] <<- mcCullochPittsLayer(...elt(i)$n, input)
                            input <- ...elt(i)$n
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
        }
    )
)
