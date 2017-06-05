function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 

m = size(X, 1);
% Add ones to the X data matrix
X = [ones(m, 1) X];
[~,p] = max(sigmoid(X*(all_theta')), [], 2);

end
