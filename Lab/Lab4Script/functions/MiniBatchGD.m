function  smooth_box = MiniBatchGD(RNN,GDparam)

[ind_to_char,char_to_ind,book_data] = Read_Data('data/Goblet.txt');
epoch = 1;
smooth_box = [];
e = 1;
iter = 1;
f = fieldnames(RNN)';
for i=1:numel(f)
  Mthe.(f{i}) = zeros(size(RNN.(f{i})));
end
while epoch <= GDparam.epochnum 
       
    X_chars = book_data(e:e+GDparam.seq_length-1);
    Y_chars = book_data(e+1:e+GDparam.seq_length);
    X_trans = to_onehot(X_chars,char_to_ind);
    Y_trans = to_onehot(Y_chars,char_to_ind);
    if e == 1 && iter ==1
        h0 = zeros(GDparam.m,1);
    else
        h0 = h(:,end);
    end
    [a,h,~,p] = Evaluatesynth(X_trans,h0,RNN);
    ce = ComputeLoss(X_trans, Y_trans, RNN, h0);
    grads = ComputeGradients(X_trans,Y_trans,RNN,a,h,p);
    %%%%%%
    if epoch == 1&&e ==1
        smooth_loss = ce;
        smooth_box = [smooth_box,smooth_loss];
        sprintf('smooth_loss: %f',smooth_loss)
        save('smooth_box.mat','smooth_box');
    else
        smooth_loss = 0.999* smooth_loss + 0.001 * ce;
        smooth_box = [smooth_box,smooth_loss];
    end
    
    %f = fieldnames(RNN)';
    % Adagrad
    %for i=1:numel(f) 
    for f = fieldnames(grads)'
      % clip gradient
      grads.(f{1}) = max(min(grads.(f{1}), 10), -10);
      Mthe.(f{1}) = Mthe.(f{1})+grads.(f{1}).^2;
      RNN.(f{1}) = RNN.(f{1})-GDparam.eta*grads.(f{1})./sqrt(Mthe.(f{1})+1e-9);
    end
   % show loss
%     if rem(iter,100) == 0
%         smooth_box = [smooth_box,smooth_loss];
%         save('smooth_box.mat','smooth_box');
%     end
    % show txt
    if rem(iter,10000) == 0 || iter ==1
        n = 1000;
        x_0 = X_chars(:,1);
        sprintf('smooth_loss: %f',smooth_loss)  
        save('smooth_box.mat','smooth_box');
        [~,generated_txt] = txt_generator(n,h0,GDparam,x_0, RNN,char_to_ind,ind_to_char);
        sprintf('-------- epoch %d iterataion %d -------------',epoch,iter)
%         sprintf('generated txt :\n')
        disp(generated_txt);

    end

    e = e+GDparam.seq_length;
    iter = iter +1;
    if  e> length(book_data)-GDparam.seq_length-1
        epoch = epoch +1;
        e = 1; 
        save('RNN.mat','RNN');
    end

end

end

