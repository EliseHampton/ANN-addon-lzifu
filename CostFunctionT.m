function [J] = CostFunctionT(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):(((hidden_layer_size * (input_layer_size + 1)) + (hidden_layer_size*(1+hidden_layer_size))))), ...
                 hidden_layer_size, (hidden_layer_size + 1));

Theta3 = reshape(nn_params((1 + ((hidden_layer_size * (input_layer_size + 1)) + (hidden_layer_size*(1+hidden_layer_size)))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%

%implement the changes in y to be vectors
y_matrix = eye(num_labels)(y,:);
%size(y_matrix)

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

%add X0's
X = [ones(m, 1) X];
% Compute second layer or hidden layer
[z2] = Theta1*X';
[a2] = sigmoid(z2);
%add a0's
a2 = [ones(1, m); a2];
% Compute third layer or output layer
[z3] = Theta2*a2;
[a3] = sigmoid(z3);

a3 = [ones(1, m); a3];
[z4] = Theta3*a3;
[a4] = sigmoid(z4);

%calculate cost function
temp1 = -y_matrix.*log(a4)';
%size(temp1)
temp2 = -(1-y_matrix).*log(1-a4)';
%size(temp2)
J = sum((1/m)*sum(temp1+temp2));%+(lambda/(2*m))*sum(Theta2(2:size(Theta2,1),:).^2);
                                %size(J)

%sum theta1 and 2 and add to J
sum_theta1 = sum(sum(Theta1(:,2:size(Theta1,2)).^2));
sum_theta2 = sum(sum(Theta2(:,2:size(Theta2,2)).^2));
sum_theta3 = sum(sum(Theta3(:,2:size(Theta3,2)).^2));

J = J + (lambda/(2*m))*(sum_theta1 + sum_theta2 + sum_theta3);







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
%grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
