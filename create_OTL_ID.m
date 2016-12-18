function [ID_old, ID_new] = create_OTL_ID(NumOld, n)
% n - dataset size
%

ID_old = randperm(NumOld);
IdxNew = NumOld+1: n;

ID_new=[];
for i=1:20,
    perm_t=randperm(n-NumOld);
    ID_new = [ID_new; IdxNew(perm_t)];
end