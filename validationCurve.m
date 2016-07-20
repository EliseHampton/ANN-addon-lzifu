function [lambda_vec, error_train, error_val,error_test] = ...
    validationCurve(X, y, Xval, yval,Xtest,ytest,init_nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%
for i = 1:length(lambda_vec)
    lambda = lambda_vec(i);
    % Compute train / val errors when training linear 
    % regression with regularization parameter lambda
    % You should store the result in error_train(i)
    % and error_val(i)
    
    %theta = trainLinearReg(X,y,lambda);
    %tmp1 = linearRegCostFunction(X,y,theta,0);
    %error_train(i) = tmp1;
    
    %tmp2 = linearRegCostFunction(Xval,yval,theta,0);
    %error_val(i) = tmp2;
    
    % This section trians the network i.e. finds theta1 and theta2
    options = optimset('MaxIter', 1000);
    %lambda = 0.03;
    costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
    [nn_params, cost] = fmincg(costFunction, init_nn_params, options);

    % now want to compute the cost on the test and CV set with
    % lambda = 0
    
    tmp1 = CostFunctionT(nn_params, ...
                         input_layer_size, ...
                         hidden_layer_size, ...
                         num_labels, X, y, 0);
    error_train(i) = tmp1;
    tmp2 = CostFunctionT(nn_params, ...
                         input_layer_size, ...
                         hidden_layer_size, ...
                         num_labels, Xval, yval, 0);
    error_val(i) = tmp2;
    
    tmp3 = CostFunctionT(nn_params, ...
                         input_layer_size, ...
                         hidden_layer_size, ...
                         num_labels, Xtest, ytest, lambda);
    error_test(i) = tmp3;
       
end
%
%










% =========================================================================

end
