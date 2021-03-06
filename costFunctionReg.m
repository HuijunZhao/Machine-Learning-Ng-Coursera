function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
m = length(y); % number of training examples

J = -1/m*(y'*log(sigmoid(X*theta))+(1-y)'*log(1-sigmoid(X*theta)))+lambda/(2*m)*theta(2:end)'*theta(2:end);
grad = zeros(size(theta));
grad(1) = 1/m*X(:,1)'*(sigmoid(X*theta)-y);

for j = 2:size(theta)    
 grad(j) = 1/m*X(:,j)'*(sigmoid(X*theta)-y)+lambda/m*theta(j);
end

end
