function [d_indx, c_value, lambda_max] = most_violated_analysis(X, y, alpha, B, group_ind)
%  The most_violated_analysis in FGM
%  Written by Mingkui Tan. 
%  If any comments, please email tanmingkui@gmail.com
%%
c =((alpha.*y)'*X)';
c_new = c.*c;
if (0==group_ind)
    [u, v]=sort(c_new,'descend');
    lambda_max = u(1);
    d_indx = v(1:B);
    d_indx = sort(d_indx)';
    c_value = c(d_indx);
else
    group_number = length(group_ind) - 1;
    group_score = zeros(group_number,1);
    group_size = group_ind(2:end) - group_ind(1:end-1);
    for i = 1:group_number
        group_score(i) = sum(c_new([group_ind(i)+1:group_ind(i+1)]))/(group_size(i));
    end
    [u, v]=sort(group_score,'descend');
    
    %d_indx stores the indices of selected features
    d_indx = [];
    for i = 1:B
        d_indx = [d_indx [group_ind(v(i))+1:group_ind(v(i)+1)]];
    end
    lambda_max = u(1);
    d_indx = sort(d_indx);
    c_value = c(d_indx);
end

end

