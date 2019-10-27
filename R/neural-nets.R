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
#' @examples
#' # Create a dataset
#' ds <- iris
#' ds$Petal.Length <- NULL
#' ds$Petal.Width <- NULL
#' ds <- ds[ds$Species != "versicolor",]
#' ds$Code <- as.integer(ds$Species == "virginica")
#' ds <- ds[sample(nrow(ds)),]
#'
#' # Create the network
#' net <- neuralNet(2, perceptron(1))
#'
#' # Train the network, takes a while
#' net$train(ds[,c(1,2)], ds$Code, epochs = 5000)
#'
#' # Check the output
#' net$output(c(1,2))
#'
#' # See accuracy
#' ds$Calc <- sapply(1:nrow(ds), function(x) net$output(ds[x,c(1,2)]))
#' length(which(ds$Code==ds$Calc))/nrow(ds)
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
        nLayers <- length(layers)
        for (e in 1:epochs) {
            # Initialize changes vector
            ch <- vector("list", nLayers)
            ch[[1]] <- vector("list", layers[[1]]$n)
            for (neu in 1:layers[[1]]$n) {
                ch[[1]][[neu]] <- list(
                    "ws" = vector("numeric", length(ins[1,])),
                    "b" = vector("numeric", 1)
                )
            }
            if (nLayers > 1) {
                for (l in 2:nLayers) {
                    ch[[l]] <- vector("list", layers[[l]]$n)
                    for (ne in 1:layers[[l]]$n) {
                        ch[[l]][[ne]] <- list(
                            "ws" = vector("numeric", layers[[l-1]]$n),
                            "b" = vector("numeric", 1)
                        )
                    }
                }
            }

            for (i in 1:nrow(ins)) {
                inputs <- vector("list", nLayers + 1)
                inputs[[1]] <- ins[i,]

                # Record each output
                for (l in 1:nLayers) {
                    inputs[[l+1]] <- layers[[l]]$output(inputs[[l]])
                }
                cost <- outs[i,] - inputs[[nLayers+1]]
                newErr <- cost

                # Calculate weight changes
                li <- nLayers
                for (l in rev(layers)) {
                    err <- newErr
                    newErr <- vector("numeric", length(inputs[[li]]))
                    ni <- 1
                    for (neu in l$neurons) {
                        d <- neu$ws*inputs[[li]]*err[[ni]]*tax
                        db <- err[[ni]]*tax

                        ch[[li]][[ni]][["ws"]] <- ch[[li]][[ni]][["ws"]] + d
                        ch[[li]][[ni]][["b"]] <- ch[[li]][[ni]][["b"]] + db

                        newErr <- newErr + err*neu$ws
                        ni <- ni + 1
                    }
                    li <- li - 1
                }
            }

            # Average changes and apply
            li <- 1
            for (l in layers) {
                ni <- 1
                for (neu in l$neurons) {
                    neu$ws <- neu$ws + ch[[li]][[ni]]$ws
                    neu$bias <- neu$bias + ch[[li]][[ni]]$b
                    ni <- ni + 1
                }
                li <- li + 1
            }
        }
    }
)
