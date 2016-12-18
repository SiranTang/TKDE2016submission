function [f_obj, gw] = calculate_fobj_gw(Data, y, v, p, B, C, sub_feat_ind)
% Calculate the objective value and gradient at the point v w.r.t the
% logistic loss
%   
%%
y_test = zeros(length(y), 1);

for k=1:p
    v_ = v(sub_feat_ind(k)+1:sub_feat_ind(k+1));
    x_data = Data{k};
    y_test = y_test + x_data*v_;
end
    %x_i = exp(-y_i w'x)
    x_i = - y_test.*y;
    expywx =  exp(x_i); 
    fv_test = log(1+expywx);
    f_obj = C*sum(fv_test);

    expcoef = expywx./(1+expywx);
    expcoefy = expcoef.*y;

    for k=1:p
        y_testy = repmat(expcoefy,1,size(Data{k},2)); 
        dw = y_testy.*Data{k};
        %dw = \sum dw_i
        dw_sum = sum(dw,1);
        gw(sub_feat_ind(k)+1:sub_feat_ind(k+1)) = -C*dw_sum;
    end
    
end

