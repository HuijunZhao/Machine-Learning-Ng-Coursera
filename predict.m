function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
a1 = [ones(size(X,1),1), X];
a2 = [ones(size(X,1),1), sigmoid(a1*Theta1')];
[~,p]=max(sigmoid(a2*Theta2'),[],2);
end
