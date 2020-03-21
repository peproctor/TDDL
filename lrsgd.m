function [wsgd,cost] = lrsgd(X,y,lambda,numIter)
% Syntax:       [wsgd,cost] = lrsgd(X,y,lambda,numIter)
%               
% Inputs:       X is a (D x N) matrix whose columns are examples/features
%               
%               y is a vector of length N whose values are the labels -1 and +1
%
%               lambda is the regularization paramter
%
%               numIter is the number of iterations to perform
%               
% Outputs:      wsgd is a vector of length D that defines the classifier
%
%               cost is a vector of length numIter of the cost at each iteration

[D,N] = size(X);

cost = zeros(numIter,1);
wsgd = zeros(D,1);
for kk = 1:numIter
    mu = 100/kk;
    idx = randperm(N);
    for ii = 1:N
        ind = idx(ii);
        g = (-y(ind)*X(:,ind)/(1 + exp(y(ind)*X(:,ind)'*wsgd)) + lambda*wsgd) / N;
        wsgd = wsgd - mu*g;
    end
    cost(kk) = sum(log(1 + exp(-y.*(X'*wsgd))))/N + lambda*wsgd'*wsgd;
end
