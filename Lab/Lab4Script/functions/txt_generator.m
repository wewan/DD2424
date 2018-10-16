function [generated_onehot,generated_txt] = txt_generator(n,h,GDparam,x_0, RNN,char_to_ind,ind_to_char)
h_t = h;
x_t= to_onehot(x_0,char_to_ind);
generated_onehot = zeros(GDparam.K,n);
generated_txt = '';
for i = 1:n 
    [~,h_t,~,p_t] = synthesize(x_t,h_t,RNN);
    cp = cumsum(p_t);
    a = rand;
    ixs = find(cp-a>0);
    ii = ixs(1);
    x_t = zeros(GDparam.K,1);
    x_t(ii,1) = 1;
%     generated_txt = strcat(generated_txt,ind_to_char(ii));
    generated_txt = [generated_txt,ind_to_char(ii)];
    generated_onehot(:,i) = x_t;
end
