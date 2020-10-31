function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X*theta;
var1 = h-y;
J = (var1'*var1)/2/m + (lambda/2/m)*theta(2:end)'*theta(2:end);

var2 = (X'*(h - y))/m;
grad(1) = var2(1);

var3 = ((X'*(h - y)) + lambda*theta)/m ;
grad(2:end) = var3(2:end);

fprintf("\n\n")
fprintf("Dimensions of X = %i x %i \n\n", size(X,1), size(X,2))

fprintf("Dimensions of (h - y) = %i x %i \n\n\n\n", size(h-y,1), size(h-y,2))

fprintf("Dimensions of X'.(h - y) = %i x %i \n\n", size(X'*(h - y),1), size(X'*(h - y),2))
fprintf("Value of X'.(h - y) = [%f ; %f] \n\n", X'*(h - y))

fprintf("\n\n")
fprintf("Dimensions of (h -y).*X = %i x %i \n\n", size((h -y).*X,1), size((h -y).*X,2))
fprintf("Dimensions of sum((h -y).*X) = %i x %i \n\n", size(sum((h - y).*X),1), size(sum((h - y).*X),2))
var = sum((h - y).*X);
fprintf("Value of sum((h -y).*X)) = [%f , %f] \n\n", var)







% =========================================================================

grad = grad(:);

end
