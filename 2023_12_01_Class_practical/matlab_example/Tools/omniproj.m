function x=omniproj(X,csi)

rho=sqrt(X(1,:).^2+X(2,:).^2+X(3,:).^2);
den=X(3,:)+csi*rho;
x=[X(1,:)./den; X(2,:)./den; ones(1,size(X,2))];
