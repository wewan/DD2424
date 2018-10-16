function l1 = ComputeLoss(X, Y, RNN_try, hprev)
    l1 = 0;
    h = hprev;
    n = size(X,2);
    for i = 1:n
        [~,h,~,p_t] = synthesize(X(:,i),h,RNN_try);
        l1 = l1 - log(Y(:, i)'*p_t);
    end
end
