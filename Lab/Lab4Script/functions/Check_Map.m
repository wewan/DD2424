function Check_Map(ind_to_char,char_to_ind)
 k = size(ind_to_char,1);
 for i = 1:k
     if char_to_ind(ind_to_char(i)) ~= i 
         sprintf('There is smothing wrong with Mapping !!')
         break  
     end
 end
 sprintf('good !')

end