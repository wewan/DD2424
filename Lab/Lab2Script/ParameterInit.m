function [W,b,GDparams]= ParameterInit(X,Y,hidden_notes,rng_number,n_epochs,n_batch,eta,rho,check,ifplot,ifmomentum,decay_rate)
% hidden_notes = m
% X  (d x N)
% y  (N x 1)
% Y  (K x N)
% W2 (K x m)
% W1 (m x d)
% b2 (K x 1)
% b1 (m x 1)
% P  (K x N)
rng(rng_number);
mu = 0;
sigma = 0.001;
W1 = mu + sigma*randn(hidden_notes,size(X,1),'double');%W1 (m x d)
W2 = mu + sigma*randn(size(Y,1),hidden_notes,'double');%W2 (K x m)
% b1 = mu + sigma*randn(size(hidden_notes,1),1,'double');
% b2 = mu + sigma*randn(size(Y,1),1,'double');
b1 = double(zeros(hidden_notes,1));
b2 = double(zeros(size(Y,1),1));
W = {W1,W2};
b = {b1,b2};

GDparams.n_batch = n_batch;
GDparams.eta = eta;
GDparams.n_epochs = n_epochs;
GDparams.rho = rho;
GDparams.ifplot = ifplot;
GDparams.ifmomentum = ifmomentum;
GDparams.decay_rate = decay_rate;

% check answer
if check
    sprintf('mean of W1 is %f',mean2(W{1}))
    sprintf('standard deviation of W1 is %f',std2(W{1}))
    sprintf('mean of W2 is %f',mean2(W{2}))
    sprintf('standard deviation of W2 is %f',std2(W{2}))
%     sprintf('mean of b1 is %f',mean2(b{1}))
%     sprintf('standard deviation of b1 is %f',std2(b{1}))
end
end