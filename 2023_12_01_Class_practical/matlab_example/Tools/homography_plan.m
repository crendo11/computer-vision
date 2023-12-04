function h=homography_plan(p,p_d)

[n,m]=size(p);

%M=[];
for j=1:m
    M(3*(j-1)+1,1) =  0 ;
    M(3*(j-1)+1,2) =  0 ;
    M(3*(j-1)+1,3) =  0 ;
    M(3*(j-1)+1,4) = -p_d(1,j)*p(3,j) ;
    M(3*(j-1)+1,5) = -p_d(2,j)*p(3,j) ;
    M(3*(j-1)+1,6) = -p_d(3,j)*p(3,j) ;
    M(3*(j-1)+1,7) =  p_d(1,j)*p(2,j) ;
    M(3*(j-1)+1,8) =  p_d(2,j)*p(2,j) ;
    M(3*(j-1)+1,9) =  p_d(3,j)*p(2,j) ;

    M(3*(j-1)+2,1) =  p_d(1,j)*p(3,j) ;
    M(3*(j-1)+2,2) =  p_d(2,j)*p(3,j)  ;
    M(3*(j-1)+2,3) =  p_d(3,j) *p(3,j);
    M(3*(j-1)+2,4) =  0 ;
    M(3*(j-1)+2,5) =  0 ;
    M(3*(j-1)+2,6) =  0 ;
    M(3*(j-1)+2,7) = -p_d(1,j)*p(1,j) ;
    M(3*(j-1)+2,8) = -p_d(2,j)*p(1,j) ;
    M(3*(j-1)+2,9) = -p_d(3,j)*p(1,j) ;

    M(3*(j-1)+3,1) = -p_d(1,j)*p(2,j) ;
    M(3*(j-1)+3,2) = -p_d(2,j)*p(2,j) ;
    M(3*(j-1)+3,3) = -p_d(3,j)*p(2,j) ;
    M(3*(j-1)+3,4) =  p_d(1,j)*p(1,j) ;
    M(3*(j-1)+3,5) =  p_d(2,j)*p(1,j) ;
    M(3*(j-1)+3,6) =  p_d(3,j)*p(1,j) ;
    M(3*(j-1)+3,7) =  0 ;
    M(3*(j-1)+3,8) =  0 ;
    M(3*(j-1)+3,9) =  0 ;
end

[u,s,v]=svd(M);

% kbir=s(1,1);
% nkbir=1;
% [nl,nc]=size(s);
% for i=2:9
%     if (kbir > s(i,i))
%         kbir=s(i,i);
%         nkbir=i;
%     end
% end

for i=1:3
    for j=1:3
        h(i,j)=v(3*(i-1)+j,end);
    end
end
% display('est la condition d''�galit�..............');
% M*v(:,end);
% 
% display('Teste du calcul d''homographie..........');
        