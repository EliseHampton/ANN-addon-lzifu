function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda, init_nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);
%size(X)
%size(y)
%'hello'

% You need to return these values correctly
error_train = zeros(m/100, 1);
error_val   = zeros(m/100, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%yval
%Xval
for i = 100:100:m
    % Compute train/cross validation errors using training examples 
    % X(1:i, :) and y(1:i), storing the result in 
    % error_train(i) and error_val(i)
    
    % This section trians the network i.e. finds theta1 and theta2
    options = optimset('MaxIter', 1000);
    lambda = 3;
    costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X(1:i,:), y(1:i), lambda);
    [nn_params, cost] = fmincg(costFunction, init_nn_params, ...
                                options);
    %'next iteration'

    % now want to compute the cost on the test and CV set with
    % lambda = 0
    
    tmp1 = CostFunctionT(nn_params, ...
                         input_layer_size, ...
                         hidden_layer_size, ...
                         num_labels, X(1:i,:), y(1:i), 0);
    error_train(i/100) = tmp1;
    %size(X(1:i,:))
    %size(Xval)
    tmp2 = CostFunctionT(nn_params, ...
                         input_layer_size, ...
                         hidden_layer_size, ...
                         num_labels, Xval, yval, 0);
    error_val(i/100) = tmp2;

           
end




% -------------------------------------------------------------

% =========================================================================

end
