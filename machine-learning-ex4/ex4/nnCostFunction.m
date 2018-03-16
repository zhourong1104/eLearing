function [J grad] = nnCostFunction(nn_params, ...
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

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ones(m,1), X];


for i = 1:m
  t = zeros(num_labels,1);
  t(y(i)) = 1;
  temp(i,:) = t';
end
y = temp;
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Z2 = X * Theta1';
H2 = (ones(size(Z2)) + exp(-1 * (Z2) )) .**(-1);
Z3 = [ones(m,1),H2] * Theta2';

H = (ones(size(Z3)) + exp(-1 * (Z3) )) .**(-1);

for j = 1:num_labels
  J(j) = (-1) * sum(log(H(:,j)')*y(:,j) + log(1 .- H(:,j)') * (1 .- y(:,j)) ) /m ;  
end

J = sum(J);

J = J + lambda *(sum(sum(Theta1((hidden_layer_size+1):end).**2))+...
         sum(sum(Theta2(num_labels+1:end).**2)))/(2*m);



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients


Delta2 = 0;
Delta1 = 0;
for k= 1:m
  z2 = X(k,:) * Theta1';
  a2 = sigmoid(z2);
  size(a2);
  z3 = [1,a2] * Theta2';
  a3 = sigmoid(z3);
  [value pos] = max(a3);
  a3_p = zeros(size(a3));
  a3_p(pos) = 1; 
  a3_p;
  y(k,:);
  delta3 = abs(a3_p - y(k,:));
  delta2 = Theta2(:,2:end)' *delta3' .* sigmoidGradient(z2');
  size(delta2);
  Delta2 = Delta2 + delta3'*[1,a2];
  Delta1 = Delta1 + delta2*X(k,:);
end
Delta1;
Delta2;
Theta2_grad = Delta2/m + lambda * Theta2/m;
Theta1_grad = Delta1/m + lambda * Theta1/m;
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
