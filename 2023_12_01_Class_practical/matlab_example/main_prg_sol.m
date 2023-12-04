% Camera paramaters
K= [425.19303  0       692.86729;  
	0      424.86463  572.11922; 
	0      0       1];
csi=0.98754;

% Load and normalize the reference image
tmp1=imread('images/Im_R0_T0.pgm') ;
img1=double(tmp1(:,:,1));
I1=img1/max(img1(:));
figure(1);
imshow(I1); hold on;

% Generate the spherical reference image
% .. call your function taking as argement the loaded image I1, the
% calibration matrix K, the miror parameter csi and the size of the
% spherical image N and M.
% The function return the spherical image Is
%.....................................
%.....................................

Bw=2*512;

% -----------------------------------------------
% Generate the spherical image
[Is1,phi_vec,theta_vec]=ImToSphere(I1,K,csi,Bw,1);
%Is = Passage_ImageP_ImageS(I1,K, Bw ,csi);


% use the showing function with spherical option 
% The function is provided.
figure;
yashow(Is1,'Spheric'); colormap gray; hold on;