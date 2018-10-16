function  J = ComputeCost3(X,Y,W,b, GDparams,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lambda = GDparams.lambda;
[P,~,~,~,~,~] = EvaluateClassifier3(X,W,b,GDparams,varargin);
D = size(X,2);
% Y'P ---> (1 x K) x (K x 1) if Y is a vector
% Y.*p for extracting the matched P if Y is a matrix. 
J1  = -1/D*sum(log(sum(Y.*P)));
J2  = 0;
L   = size(W,1);
for i = 1:L
    J2 = J2 + lambda*(sum(sum(W{i}.^2)));
end
J = J1 + J2;