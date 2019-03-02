library(R6)
library(e1071)
library(readr)
library(dplyr)


load_training <- function(path="data/training_data.tsv", n_max=Inf) {
    readr::read_tsv(path, n_max=n_max) %>% select(-1)
    # pictures <- df %>% select(-1, -number)
    # numbers <- df %>% select(number)
    # list(numbers=numbers, data=pictures)
}

NeuralNetwork <- R6Class(
    "NeuralNetwork", 
    public = list(
        num_layers = NULL,
        sizes = NULL,
        biases = NULL,
        weights = NULL,
        
        # sizes: Number of neurons in each layer of network
        initialize = function(sizes, verbose=FALSE) {
            
            self$num_layers <- length(sizes)
            self$sizes <- sizes
            
            # Initial biases for each node in each layer
            self$biases <- lapply(
                sizes[2:length(sizes)],
                function(size) {
                    matrix(rnorm(size, 0, 1))
                }
            )
            
            # Matrix with weights between nodes in subsequent layers
            # weight_pairs <- rbind(sizes[1:length(sizes)-1], sizes[2:length(sizes)])
            size_pairs <- self$size_pairs(sizes)
            self$weights <- lapply(
                size_pairs,
                function(pair) {
                    matrix(rnorm(pair[1] * pair[2], 0, 1), nrow=pair[2])
                }
            )
            
            if (verbose) {
                print("Biases")
                print(self$biases)
                print("Weights")
                print(self$weights)
            }
        },
        
        # Return output of network if a is input
        feedforward = function(only_data, biases=NULL, weights=NULL) {
            
            if (is.null(biases)) {
                biases <- self$biases
            }
            
            if (is.null(weights)) {
                weights <- self$weights
            }
            
            operating_data <- only_data
            for (index in seq_len(length(biases))) {
                layer_weights <- weights[[index]]
                layer_bias <- biases[[index]]
                operating_data <- self$sigmoid(layer_weights %*% operating_data) + layer_bias
            }
            
            operating_data
        },
        
        #' Train neural network using mini-batch stochastic gradient descent
        #'
        #' Train format: Row with datapoints, and category in last?
        #'
        #' training_data: (MAYBE NOT) List of tuples (x, y) representing training inputs and desired outputs
        #' test_data: If provided, evaluating after each epoch (useful for tracking, but slowing down)
        SGD = function(training_data, epochs, mini_batch_size, eta, test_data=NULL, debug=FALSE) {
            
            if (!is.null(test_data)) {
                n_test <- length(test_data[[1]])
            }
            
            n <- length(training_data[[1]])
            
            sapply(
                seq_len(epochs),
                function(epoch) {
                    shuffled_train_data <- training_data[sample(seq_len(nrow(training_data))),]
                    chunks <- split(shuffled_train_data, trunc(0:(nrow(shuffled_train_data)-1) / mini_batch_size))
                    
                    for (chunk_name in names(chunks)) {
                        chunk <- chunks[[chunk_name]]
                        # Command: UPDATE MINI BATCH?
                        new_pars <- self$update_mini_batch(chunk, eta)
                        
                        if (debug) {
                            message("Biases before: ", paste(self$biases, collapse=", "))
                            message("Biases after: ", paste(new_pars$biases, collapse=", "))
                        }
                        
                        self$weights <- new_pars$weights
                        self$biases <- new_pars$biases
                    }
                    
                    if (!is.null(test_data)) {
                        message("Epoch: ", epoch, " Test: <tests> / ", n_test)
                    }
                    else {
                        message("Epoch ", epoch, " complete")
                    }
                }
            )
        },
        
        # Update network's weights and biases by applying gradient descent using backpropagation
        # to a single mini batch. 
        # mini_batch list of tuples (x, y) 
        # eta learning rate 
        update_mini_batch = function(mini_batch, eta) {
            
            nabla_b <- NULL
            nabla_w <- NULL
            
            apply(mini_batch, 1, function(row) {
                data <- row[1:length(row)-1]
                annot <- row[length(row)]
                
                message(data)
                message(annot)
                
                # delta_nablas <- self$backprop(data, annot)
                # delta_nabla_b <- delta_nablas$b
                # delta_nabla_w <- delta_nablas$w
            })
            
            list(weights=self$weights, biases=self$biases)
            
            # self$weights <- NULL
            # self$biases <- NULL
            
        },
        
        # Return tuple (nabla_b, nabla_w) representing gradient for cost function C_x
        # nabla_b and nabla_w are layer-by-layer lists of numpy arrays, similar to self$biases and self$weights
        backprop = function(datapoints, outcome, biases, weights) {
            
            nabla_b <- NULL
            nabla_w <- NULL
            activation <- datapoints
            
            # List to store activations, layer by layer
            activations <- list()
            
            # List to store z vectors, layer by layer
            zs <- list()
            
            sapply(seq_len(biases), function(index) {
                b <- biases[[index]]
                w <- weights[[index]]
                z <- w %*% activation + b
                zs <- c(zs, z)
                activation <- sigmoid(z)
                activations <- c(activations, activation)
            })
            
            # Backward pass
            delta <- self$cost_derivative(activations[length(activations)-1], outcome) *
                self$sigmoid_prime(zs[length(zs-1)])
            
            nabla_b[length(nabla_b)-1] <- delta
            nabla_w[length((nabla_w)-1)] <- delta %*% t(activations[length(activations)-2])
            
            list(b=nabla_b, w=nabla_w)
        },
        
        # Number of test inputs where neural network gives correct results
        # Highest output is assumed for index with highest activation in final layer
        evaluate = function(test_data) {
            results <- self$feedforward(test_data)
            length(which(results == test_data$number))
        },
        
        # Vector of partial derivatives (C_x)
        cost_derivative = function(output_activations, y) {
            output_activations - y
        },
        
        sigmoid = function(z) {
            1 / (1 + exp(-z))
        },
        
        # Derivative of the sigmoid function
        sigmoid_prime = function(z) {
            e1071::sigmoid(z) * (1 - e1071::sigmoid(z))
        },
        
        size_pairs = function(sizes) {
            lapply(as.list(strsplit(paste(sizes[1:length(sizes)-1], sizes[2:length(sizes)]), " ")), as.numeric)
        }
    )
)

test_network <- function() {
    nn <- NeuralNetwork$new(c(4,3,2))
    nn$feedforward(c(1,2,1,2))
    
    dummy_data <- data.frame(
        c1=abs(rnorm(n=10, mean=0, sd=1)),
        c2=abs(rnorm(n=10, mean=0, sd=1)),
        c3=abs(rnorm(n=10, mean=0, sd=1)),
        c4=abs(rnorm(n=10, mean=0, sd=1)),
        outcome=rep(c(0,1), 5)
    )
    
    nn$SGD(dummy_data, epochs=2, mini_batch_size=3, eta=0.01, debug=TRUE)
}

