import cv2 as cv
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt



def sphgrid(n):
    """
    
    Compute a equi-angular spherical grid given by theta[i][_],phi[_][j], \n
    i=0..n-1, j=0..n-1 theta[i][j] = (2*i+1)*pi/(2*n) and phi[i][j] = j*2*pi/n
    
    Parameters
    ----------
    n : int
        size of each grid..

    Returns
    -------
    phi : float matrix with size n*n
        Matrix which is compute \n
        i=0..n-1, j=0..n-1 phi[i][j] = j*2*pi/n.
    theta : float matrix with size n*n
        Matrix which is compute \n 
        i=0..n-1, j=0..n-1 theta[i][j] = (2*i+1)*pi/(2*n).


    """
    phi = np.zeros((n,n))
    theta = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            phi[i][j]=j*2*np.pi/n
            theta[i][j]=(2*i+1)*np.pi/(2*n)
    return (phi,theta)            
    
def omniproj(X,csi):
    """
    
    Compute the matrix of spherical coordinate adjust for the choosen camera

    Parameters
    ----------
    X : float matrix
        Matrix which contains every spherical cordinate.
    csi : float number
        one of the parameter of the camera that is normally equals to 0.

    Returns
    -------
    x : float matrix
        matrix with every cordinate but adjust for the camera.

    
    """
    rho = np.sqrt(X[0][:]*X[0][:]+X[1][:]*X[1][:]+X[2][:]*X[2][:])
    den = X[2][:]+rho*csi
    x = np.zeros((3,np.shape(X)[1]))
    for i in range(np.shape(X)[1]):
        x[0][i] = X[0][i]/den[i] 
        x[1][i] = X[1][i]/den[i]
        x[2][i] = 1
    return x
    

def ImToSphere(I, H, csi, imsph_dim):
    """
    
    transform only a black and white spherical picutre to 
    a black and white rectangular picture without deform it.

    Parameters
    ----------
    I : float matrix
        matrix where are every pixel of our black and white source picture.
    H : float matrix
        Panoramic camera calibration (one of the parameter of the camera).
    csi : float
        Mirror parameter (one of the parameter of the camera).
    imsph_dim : int
        size of the return matrix.

    Returns
    -------
    out : float matrix
        matrix that match with the picture but now in rectangular form.

    """
    [nl,nc] = np.shape(I)
    [phi,theta] = sphgrid(imsph_dim) 
    out = np.zeros((np.shape(phi)))
    phi_vect = np.ravel(phi,order='F').reshape(1,-1)
    theta_vect = np.ravel(theta,order='F').reshape(1,-1)
    S = np.zeros((3,imsph_dim*imsph_dim))
    S[0][:] = np.cos(phi_vect)*np.sin(theta_vect)
    S[1][:] = np.sin(phi_vect)*np.sin(theta_vect)
    S[2][:] = np.cos(theta_vect)
    x = omniproj(S, csi)
    p = np.dot(H,x)
    p = np.around(p,0)
    
    pu = np.reshape(p[0][:], (imsph_dim,imsph_dim))
    pv = np.reshape(p[1][:], (imsph_dim,imsph_dim))
    XXij, YYij = np.meshgrid(np.arange(1, nc+1), np.arange(1, nl+1))
    XXij = XXij.flatten()
    YYij = YYij.flatten()
    interpolator = LinearNDInterpolator((XXij, YYij), I.flatten()) 
    out = interpolator(pu,pv)
    out[np.isnan(out)] = 0
    return (out)

def yashow_spherique(mat):
    """
    
    Plot the rectangular picture on the surface of the sphere.

    Parameters
    ----------
    mat : int matrix
        matrix that contains every pixels of the rectangular picture.

    Returns
    -------
    None.

    """
    img = np.dstack((mat, mat, mat))
    
    # define a grid matching the map size, subsample along with pixels
    theta = np.linspace(0, np.pi, img.shape[0])
    phi = np.linspace(0, 2*np.pi, img.shape[1])

    count = 180 # keep 180 points along theta and phi
    theta_inds = np.linspace(0, img.shape[0] - 1, count).round().astype(int)
    phi_inds = np.linspace(0, img.shape[1] - 1, count).round().astype(int)
    theta = theta[theta_inds]
    phi = phi[phi_inds]
    img = img[np.ix_(theta_inds, phi_inds)]

    theta,phi = np.meshgrid(theta, phi)
    R = 1

    # sphere
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)
    
    # create 3d Axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x.T, y.T, z.T, facecolors=img/255, cstride=1, rstride=1) # we've already pruned ourselves

    # make the plot more spherical
    ax.axis('scaled')
    plt.show()



# Camera paramaters
K= [[425.19303, 0, 692.86729],[0, 424.86463, 572.11922],[0, 0, 1]]
H = np.array([[425.19303, 0, 692.86729],[0, 424.86463, 572.11922],[0, 0, 1]])
csi=0.98754;


# Load and normalize the reference image
tmp1=cv.imread("../../images/Im_R0_T0.pgm", 0)
I1 = tmp1/np.max(tmp1)


 # Generate the spherical reference image
 # .. call your function taking as argement the loaded image I1, the
 # calibration matrix K, the miror parameter csi and the size of the
 # spherical image N and M.
 # The function return the spherical image Is
Bw = 2*512

# Generate the sperical image
Is1 = ImToSphere(I1, K, csi, Bw)
Is2=np.transpose(Is1)*255
cv.imshow("image",Is2/255)
cv.waitKey(0)
cv.destroyAllWindows()
yashow_spherique(Is2)