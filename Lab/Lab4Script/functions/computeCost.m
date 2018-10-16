function ce = computeCost(x_t,h_t,Y,RNN)
    [~,~,~,p_t] = synthesize(x_t,h_t,RNN);
    ce = -log(Y'*p_t);
end