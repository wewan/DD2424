function [a,h,o,p] = Evaluatesynth(X,h0,RNN)
n = size(X,2);
m = size(RNN.W,1);
k = size(RNN.c,1);
a = zeros(m,n);
h = zeros(m,n+1);
o = zeros(k,n);
p = zeros(k,n);
h_t = h0;
h(:,1) = h0;
for i = 1:n
    [a_t,h_t,o_t,p_t] = synthesize(X(:,i),h_t,RNN);
    h(:,i+1) = h_t;
    a(:,i) = a_t;
    o(:,i) = o_t;
    p(:,i) = p_t;
end
end