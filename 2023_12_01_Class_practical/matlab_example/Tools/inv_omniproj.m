function X=inv_omniproj(x,csi)
	gama = sqrt(1+(1-csi^2)*(x(1,:).^2.+x(2,:).^2));
	lam = (csi+gama)./(x(1,:).^2+x(2,:).^2+1);
	X=[lam.*x(1,:); lam.*x(2,:); lam-csi];
