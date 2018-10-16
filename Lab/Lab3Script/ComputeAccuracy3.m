function acc = ComputeAccuracy3(X,y,W,b,GDparams,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X (d x N)
% y (N x 1)
% W (K x d)
% b (K x 1)

% method 1
[P,~,~,~,~,~] = EvaluateClassifier3(X,W,b,GDparams,varargin);
[~,Index] = max(P);
acc = sum(Index == y')/length(y);
% method 2
% use onehot Y .* onehot outcome and then sum up
