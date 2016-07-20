function [error_train, error_val] = ...
    IterationCurve(X, y, Xval, yval, lambda, init_nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels)
% ITERATIONCURVE Generates the train and cross validation set errors needed 
%       to plot a learning curve
%       [error_train, error_val] = ...
%       ITERATIONCURVE(X, y, Xval, yval, lambda, init_nn_params,
%       input_layer_size, hidden_layer_size, num_labels) returns the train and
%       cross validation set errors for an iteration curve. 

% Number of training examples
m = size(X, 1);

% values to return
error_train = zeros(m/100, 1);
error_val   = zeros(m/100, 1);


for i = 100:100:m
    % Compute train/cross validation errors using training examples 
    % X(1:i, :) and y(1:i), storing the result in 
    % error_train(i) and error_val(i)
    
    % This section trians the network i.e. finds thetas
    options = optimset('MaxIter', 2000);
    lambda = 0.03;
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

    tmp2 = CostFunctionT(nn_params, ...
                         input_layer_size, ...
                         hidden_layer_size, ...
                         num_labels, Xval, yval, 0);
    error_val(i/100) = tmp2;

           
end

end
