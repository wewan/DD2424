function [GDparam,RNN] = ParamInit(ind_to_char)
rngnumber = 40;
rng(rngnumber)
m = 100;
K = size(ind_to_char,1);

GDparam.m = m;
GDparam.K = K;
GDparam.eta = 0.1;
GDparam.epochnum = 30;
GDparam.seq_length = 25;

sig = 0.01;
RNN.b = zeros(m,1);
RNN.c = zeros(K,1);
RNN.U = randn(m, K)*sig;
RNN.W = randn(m, m)*sig;
RNN.V = randn(K, m)*sig;
end