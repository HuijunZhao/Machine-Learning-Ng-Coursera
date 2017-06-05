function [J, grad] = costFunction(theta, X, y)

m = length(y); 

J = -1/m*(y'*log(sigmoid(X*theta))+(1-y)'*log(1-sigmoid(X*theta)));
grad = zeros(size(theta));
for j = 1:size(theta)    
 grad(j) = 1/m*X(:,j)'*(sigmoid(X*theta)-y);
end
end
