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
        train = function(ins, outs, epochs = 1, tax = .01, maxErr = 0) {
            err <- vector("numeric", nrow(ins))
            repeat {
                for (i in 1:nrow(ins)) {
                    err[i] <- outs[i] - output(ins[i,])
                }
                e <- mean(err)
                ws <<- unlist(ws + tax*e*ins[i,], use.names = F)
                bias <<- bias + e*tax

                epochs <- epochs - 1
                if (epochs == 0 || abs(e) < maxErr) {
                    break()
                }
            }
        }
    )
)

mcCullochPittsLayer <- setRefClass(
    "McCullochPittsLayer",
    fields = list(n = "numeric", dims = "vector", neurons = "vector"),
    methods = list(
        initialize = function(n, len) {
            n <<- n
            dims <<- c(n, len)
            neurons <<- replicate(n,
                                  mcCullochPitts$new(ws = runif(len), bias = 1))
        },
        output = function(inputs) {
            lapply(neurons, function(x) x$output(inputs))
        },
        train = function(ins, outs, epochs = 1, tax = .01, maxErr = 0) {
            for (i in 1:n) {
                neurons[[i]]$train(ins, outs[,i], epochs, tax, maxErr)
            }
        }
    )
)

perceptron <- setRefClass(
    "Perceptron",
    fields = list(ws = "numeric", bias = "numeric"),
    methods = list(
        initialize = function(ws, bias) {
            ws <<- ws
            bias <<- bias
        },
        output = function(inputs) {
            as.integer(sum(ws*inputs) + bias > 0)
        },
        train = function(ins, outs, epochs = 1, tax = .01, maxErr = 0) {
            err <- vector("numeric", nrow(ins))
            repeat {
                for (i in 1:nrow(ins)) {
                    err[i] <- outs[i] - output(ins[i,])
                }
                e <- mean(err)
                ws <<- unlist(ws + tax*e*ins[i,], use.names = F)
                bias <<- bias + e*tax

                epochs <- epochs - 1
                if (epochs == 0 || abs(e) < maxErr) {
                    break()
                }
            }
        }
    )
)

perceptronLayer <- setRefClass(
    "PerceptronLayer",
    fields = list(n = "numeric", dims = "vector", neurons = "vector"),
    methods = list(
        initialize = function(n, len = 1) {
            n <<- n
            dims <<- c(n, len)
            neurons <<- replicate(n, perceptron(runif(len), 1))
        },
        output = function(inputs) {
            sapply(neurons, function(x) x$output(inputs))
        },
        train = function(ins, outs, epochs = 1, tax = .1, maxErr = 0) {
            lapply(neurons,
                   function(x) x$train(ins, outs, epochs, tax, maxErr))
        }
    )
)

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
                        layers[[i]] <<- McCullochPittsLayer(...elt(i)$n, input)
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
