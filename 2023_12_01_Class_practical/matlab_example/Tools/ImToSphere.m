function [out,phi_vec,theta_vec] =ImToSphere(I,H,csi,imsph_dim, method)
% INPUT 
% I: Image source
% H: Panoramic camera calibration
% csi: Mirror parameter
% imsph_dim: Spherical image dimension (the size of the out matrice). Note that using fst needs a dimension power 2 

% OUTPUT
% out: Output image in the spherical coordinates (phi,theta)

% HADJ-ABDELKADER Hicham
% May 2007

[nl,nc]=size(I);
[phi,theta]=sphgrid(imsph_dim); 
size(phi)
out=zeros(size(phi));
phi_vec=phi(:)';
theta_vec=theta(:)';

switch method
	case 1
		S=[cos(phi_vec).*sin(theta_vec); sin(phi_vec).*sin(theta_vec); cos(theta_vec)]; % cartesian coordinates
		x=omniproj(S,csi);
		p=round(H*x);
	case 2 % Daniilidis method
		u=(cot(theta_vec/2).*cos(phi_vec));
		v=(cot(theta_vec/2).*sin(phi_vec));
		p=[u; v; ones(size(u))];
		p=round(H*p);
end

pu = reshape(p(1,:),imsph_dim,imsph_dim);
pv = reshape(p(2,:),imsph_dim,imsph_dim);


%[k,l]=find(p(1,:)>0 & p(1,:)<=nc & p(2,:)>0 & p(2,:)<nl);
%out(l)=I(p(2,l)+nl*(p(1,l)-1));

[XXij YYij]=meshgrid(1:nc,1:nl);

size(pu)
size(pv)
size(XXij)
size(YYij)

out= interp2(XXij,YYij,double(I),pu,pv,'linear');
    out(isnan(out)) = 0;