function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
z = X * theta;

H = (ones(size(z)) + exp(-1 * (z) )) .**(-1);

J = (-1) * sum(log(H')*y + log(1 .- H') * (1 .- y) ) /m + lambda * ...
     (sum(theta .* theta) - theta(1)**2) /(2*m);


grad(1) = sum(X(:,1)' *(H - y)) / m;
for i = 2: size(theta)
  grad(i) = sum(X(:,i)' *(H - y)) / m + lambda*theta(i) / m;
end


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
