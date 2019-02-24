library(R6)
library(e1071)
library(readr)


load_training <- function(path) {
    df <- readr::read_tsv(path)
    pictures <- df %>% select(-1, -number)
    numbers <- df %>% select(number) %>% unlist()
    list(numbers=numbers, data=pictures)
}

NeuralNetwork <- R6Class(
    "NeuralNetwork", 
    public = list(
        num_layers = NULL,
        sizes = NULL,
        biases = NULL,
        weights = NULL,
        
        # sizes: Number of neurons in each layer of network
        initialize = function(sizes) {
            
            self$num_layers <- length(sizes)
            self$sizes <- sizes
            
            # Initial biases for each node in each layer
            self$biases <- lapply(
                sizes,
                function(size) {
                    matrix(rnorm(sizes[1], 0, 1))
                }
            )
            
            # Matrix with weights between nodes in subsequent layers
            weight_pairs <- rbind(sizes[1:length(sizes)-1], sizes[2:length(sizes)])
            self$weights <- apply(
                weight_pairs,
                2,
                function(pair) {
                    matrix(rnorm(pair[1] * pair[2], 0, 1), nrow=pair[1])
                }
            )
        },
        
        # Return output of network if a is input
        feedforward = function(a) {
            
            vapply(
                seq_len(length(self$biases)),
                apply(self$weights[[1]], 2, function(col) {
                    e1071::sigmoid(col %*% a) + self$biases[[1]]
                }),
                0
            )
        },
        
        # Train neural network using mini-batch stochastic gradient descent
        #
        # training_data: List of tuples (x, y) representing training inputs and desired outputs
        # test_data: If provided, evaluating after each epoch (useful for tracking, but slowing down)
        SGD = function(training_data, epochs, mini_batch_size, eta, test_data=NULL) {
            if (!is.null(test_data)) {
                n_test <- length(test_data[[1]])
            }
            
            n <- length(training_data[[1]])
            
            vapply(
                seq_len(epochs),
                function(epoch) {
                    
                },
                rep(0)
            )
        },
        
        # Update network's weights and biases by applying gradient descent using backpropagation
        # to a single mini batch. 
        # mini_batch list of tuples (x, y) 
        # eta learning rate 
        update_mini_batch = function(mini_batch, eta) {
            
        },
        
        # Return tuple (nabla_b, nabla_w) representing gradient for cost function C_x
        # nabla_b and nabla_w are layer-by-layer lists of numpy arrays, similar to self$biases and self$weights
        backprop = function(x, y) {
            
        },
        
        # Number of test inputs where neural network gives correct results
        # Highest output is assumed for index with highest activation in final layer
        evaluate = function(test_data) {
            
        },
        
        # Vector of partial derivatives (C_x)
        cost_derivative = function(output_activations, y) {
            output_activations - y
        }
    ),
    private = list(
        
    )
)

sigmoid <- function(z) {
    1 / (1 + exp(-z))
}

# Derivative of the sigmoid function
sigmoid_prime <- function(z) {
    e1071::sigmoid(z) * (1 - e1071::sigmoid(z))
}

