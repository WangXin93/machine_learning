function [error_train, error_val] = ...
    learningCurveWithRandom( X, y, Xval, yval, lambda )
%LEARNINGCURVEWITHRANDOM Summary of this function goes here
%   Detailed explanation goes here

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

%================codes here===========
for i = 1:m    
    for t = 1:50
         index = randperm(m,i);
         X_sample = X(index,:);
         y_sample = y(index);
         theta = trainLinearReg(X_sample,y_sample,lambda);
         error_train(i) = error_train(i) + linearRegCostFunction(X_sample,y_sample,theta,0);
         error_val(i) = error_val(i) + linearRegCostFunction(Xval,yval,theta,0);
    end
    error_train(i) = error_train(i) / 50;
    error_val(i) = error_val(i) / 50;
end

%===========================
end

