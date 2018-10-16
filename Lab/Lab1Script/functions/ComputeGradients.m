function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X (d x N)
% Y (K x N)
% P (K x N)
% W (K x d)
% drad_w (K x d)
% grad_b (K x 1)
N = size(Y,2);
[K,~] = size(W);
[d,~] = size(X);
grad_W = zeros(K,d);
grad_b = zeros(K,1);
% calculate g following the lecture 3
for i = 1:N
    Yn = Y(:,i);
    Pn = P(:,i);
    Xn = X(:,i);
    % (1 x k)/(1 x k x k x k) x [(K x K)- K x 1 x 1 x K]= 1 x K
    % so the size of g ( 1 x K )
    g = - Yn'/(Yn'*Pn)*(diag(Pn)-Pn*Pn');
    % check whether g == g2
%     g2 = -(Yn-Pn)';
%     if g ~= g2
%         disp('g and g2 not equal!');
%         break;
%     end
    % gradient J w.r.t bias = g
    grad_b = grad_b +g';
    % gradient J w.r.t W = g'x 
    grad_W = grad_W +g'*Xn';
end

grad_W = (1/N)*grad_W+ 2*lambda*W;
grad_b = (1/N)*grad_b;

end