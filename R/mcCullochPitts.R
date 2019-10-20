#' The McCullochPitts neuron class, that implements the logic of the
#' McCullochPitts neuron model.
#'
#' @field ws The matrix of weights that multiply the input vector, it can be a
#' vector, a matrix or an array.
#'
#' @field bias The bias value.
#'
#' @method initialize This runs when you instantiate the class, either with
#' \code{mcCullochPitts$new()} or \code{mcCullochPitts()}, the intent is to
#' create a new McCullochPitts neuron with given weights and bias.
#'
#' @method output Gives the output of passing data to the neuron.
#' @param inputs The actual data to be fed to the nuron, this input's
#' dimentions vary with the chosen weights dimentions.
#' @return The computed value using the McCullochPitts model.
#'
#' @method train Runs the training algorithm to adjust the weights and bias of
#' the neuron
#' @param ins The list of vectors of inputs to the first layer in the network
#' @param outs The list of vectors of outputs of the last layer in the network
#' @param epochs How many rounds of training to run
#' @param tax This is the learning rate, aka eta
#' @param maxErr A contition to early stop the training process
#' @return Vector of computed values of the same size of the last layer
#'
mcCullochPitts <- setRefClass(
    "McCullochPitts",
    fields = list(ws = "numeric", bias = "numeric"),
    methods = list(
        initialize = function(ws, bias) {
            ws <<- ws
            bias <<- bias
        },
        output = function(inputs) {
            sum(ws*inputs) + bias
        },
        train = function(ins, out, epochs = 1, tax = .01, maxErr = 0) {
            repeat {
                err <- out - output(ins)
                ws <<- unlist(ws + tax*err*ins, use.names = F)
                bias <<- bias + err*tax

                epochs <- epochs - 1
                if (epochs < 1 || abs(err) < maxErr) {
                    break()
                }
            }
        }
    )
)

#' The McCullochPittsLayer class, that implements a layer of McCullochPitts
#' neurons.
#'
#' @field n The number of neurons to create in the layer
#'
#' @field dims A vector of dimensions of the inputs to the layer
#'
#' @field neurons A list with the internal neurons
#'
#' @method initialize This runs when you instantiate the class, either with
#' \code{mcCullochPittsLayer$new()} or \code{mcCullochPittsLayer()}, the intent
#'  is to set up the layer of \code{n} McCullochPittsLayer neurons with random
#'  weights and biases.
#'
#' @method output Gives the output of passing data to the layer.
#' @param input The actual data to be fed to the layer, this input's
#' dimentions vary with the chosen \code{n}.
#' @return The computed value using the McCullochPittsLayer model.
#'
#' @method train Runs the training algorithm to adjust the weights and bias of
#' the neurons
#' @param ins The list of vectors of inputs to the first layer in the network
#' @param outs The list of vectors of outputs of the last layer in the network
#' @param epochs How many rounds of training to run
#' @param tax This is the learning rate, aka eta
#' @param maxErr A contition to early stop the training process
#' @return Vector of computed values of the same size of the last layer
#'
mcCullochPittsLayer <- setRefClass(
    "McCullochPittsLayer",
    fields = list(n = "numeric", dims = "vector", neurons = "vector"),
    methods = list(
        initialize = function(n, len = 1) {
            n <<- n
            dims <<- c(n, len)
            neurons <<- replicate(n, mcCullochPitts(runif(len), runif(1)))
        },
        output = function(input) {
            sapply(neurons, function(x) x$output(input))
        },
        train = function(ins, outs, epochs = 1, tax = .01, maxErr = 0) {
            for (i in 1:n) {
                lapply(1:n, function(x) neurons[[x]]$train(ins, outs[i]))
            }
        }
    )
)
