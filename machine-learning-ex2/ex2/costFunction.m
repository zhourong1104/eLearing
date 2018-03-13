function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
size(y)
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

z = X * theta;

H = (ones(size(z)) + exp(-1 * (z) )) .**(-1);

J = (-1) * sum(log(H')*y + log(1 .- H') * (1 .- y) ) /m;


for i = 1: size(theta)
  grad(i) = sum(X(:,i)' *(H - y)) / m;
end


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
