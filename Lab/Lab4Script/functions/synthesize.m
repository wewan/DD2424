function [a_t,h_t,o_t,p_t] = synthesize(x_t,h_t,RNN)
a_t = RNN.W*h_t + RNN.U*x_t + RNN.b;
h_t = tanh(a_t);
o_t = RNN.V*h_t + RNN.c;
p_t = exp(o_t)./repmat(sum(exp(o_t)),size(o_t,1),1);
end