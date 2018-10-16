function acc = ComputeAccuracy(X,y,W,b)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X (d x N)
% y (N x 1)
% W (K x d)
% b (K x 1)

% method 1
P = EvaluateClassifier(X,W,b);
[~,Index] = max(P);
acc = sum(Index == y')/length(y);
% method 2
% use onehot Y .* onehot outcome and then sum up
