function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = [0.01;0.03;0.1;0.3;1;3;10;30];
sigma = [0.01;0.03;0.1;0.3;1;3;10;30];
n = size(C);
ind1 = 0;
ind2 = 0;
x1 = [1 2 1]; x2 = [0 4 -1];
cost = 10000;
for i=1:n
  for j = 1:n
    model =  svmTrain(X, y, C(i), @(x1, x2) gaussianKernel(x1, x2, sigma(j)));
    predictions = svmPredict(model, Xval);
    cost1 = mean(double(predictions ~= yval));
    if(cost > cost1)
       ind1 = i;
       ind2 = j;
       cost = cost1;
    end;
  end;
end;
C = C(ind1);
sigma = sigma(ind2);       
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
