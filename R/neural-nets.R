mcCullochPitts <- setRefClass(
    "McCullochPitts",
    fields = list(ws = "numeric", bias = "numeric"),
    methods = list(
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
        train = function(ins, outs, epochs = 1, tax = .01, maxErr = 0,
            errf = function(x) x) {
            for (i in 1:n) {
                neurons[[i]]$train(ins, outs[,i], epochs, tax, maxErr, errf)
            }
        }
    )
)

perceptron <- setRefClass(
    "Perceptron",
    fields = list(ws = "numeric", bias = "numeric"),
    methods = list(
        output = function(inputs) {
            as.integer(sum(ws*inputs) + bias > 0)
        },
        train = function(ins, outs, epochs = 1, tax = .1, maxErr = 0,
                         errf = function(x) x) {
            epoch <- 1
            repeat {
                err <- 0
                for (i in 1:nrow(ins)) {
                    e <- errf(outs[i,] - output(ins[i,]))
                    ws <<- ws + e*tax*ins[i,]
                    bias <<- bias + e*tax
                    err <- err + e
                }
                epoch <- epoch + 1
                if (epoch > epochs || abs(err) < maxErr) {
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
        initialize = function(n, len) {
            n <<- n
            dims <<- c(n, len)
            neurons <<- replicate(n, perceptron$new(ws = runif(len), bias = 1))
        },
        output = function(inputs) {
            lapply(neurons, function(x) x$output(inputs))
        },
        train = function(ins, outs, epochs = 1, tax = .1, maxErr = 0,
                         errf = function(x) x) {
            lapply(neurons, 
                   function(x) x$train(ins, outs, epochs, tax, maxErr, errf))
        }
    )
)
