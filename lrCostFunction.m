function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
m = length(y); 

J = -1/m*(y'*log(sigmoid(X*theta))+(1-y)'*log(1-sigmoid(X*theta)))+lambda/(2*m)*theta(2:end)'*theta(2:end);
grad = 1/m*X'*(sigmoid(X*theta)-y)+lambda/m*[0;theta(2:end)];

end
