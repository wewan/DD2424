function transfered = to_onehot(data,char_to_ind)
   K = size(char_to_ind,1);
   transfered = zeros(K,size(data,2));
   for i = 1:size(data,2)
       transfered(char_to_ind(data(i)),i) = 1;
   end
end