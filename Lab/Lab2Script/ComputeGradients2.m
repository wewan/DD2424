function [grad_W,grad_b] = ComputeGradients2(X,h,S1, Y, P, W, lambda)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  WEI WANG @copyright
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X (d x N)
% Y (K x N)
% P (K x N)
% S1 (m x N)
% h  (m x N)
% W2 (K x m)
% W1 (m x d)
% b2 (K x 1)
% b1 (m x 1)
% drad_w1 (m x d)
% grad_b1 (m x 1)
% drad_w2 (K x m)
% grad_b2 (K x 1)
N = size(X,2);
m = size(S1,1);
[K,~] = size(W);
[d,~] = size(X);
grad_W1 = zeros(m,d);
grad_b1 = zeros(m,1);
grad_W2 = zeros(K,m);
grad_b2 = zeros(K,1);
% calculate g following the lecture 4
for i = 1:N
    Yn = Y(:,i); % (K x 1)
    Pn = P(:,i); % (K x 1)
    hn = h(:,i); % (m x 1)
    Xn = X(:,i); % (d x 1)
    S1n= S1(:,i);% (m x 1)
    % according to the Lecture 4 calculate g 
    %( 1 x K )
    g = - Yn'/(Yn'*Pn)*(diag(Pn)-Pn*Pn');
    % gradient L w.r.t b2 = g
    %( K x 1 )
    grad_b2 = grad_b2 +g';
    % gradient L w.r.t W2 = g'h; hn (m x 1)
    %( K x m )-> (K x 1 x 1 x m)
    grad_W2 = grad_W2 +g'*hn';  
    % update g
    % (1 x m)
    g = g*W{2};
    % assumingg relu activation
    % (1 x m)--> (1 x m x m x m)
    g = g*diag(S1n>0);
    grad_b1 = grad_b1 + g';
    grad_W1 = grad_W1 +g'*Xn';
    
end
% gradient J
grad_W1 = (1/N)*grad_W1+ 2*lambda*W{1};
grad_W2 = (1/N)*grad_W2+ 2*lambda*W{2};
grad_b1 = (1/N)*grad_b1;
grad_b2 = (1/N)*grad_b2;
grad_W = {grad_W1,grad_W2};
grad_b = {grad_b1,grad_b2};

end