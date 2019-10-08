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
