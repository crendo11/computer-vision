"""
This file contains the code and functions for the 
unified projection model for any camera.
"""

import numpy as np


def intrinsic_matrix(fu,fv,alpha_uv, u0, v0):
    """
    Compute the intrinsic matrix K from the parameters of the camera.

    Parameters
    ----------
    fu : float
        focal length in the u direction.
    fv : float
        focal length in the v direction.
    alpha_uv : float
        skew coefficient.
    u0 : float
        principal point in the u direction.
    v0 : float
        principal point in the v direction.

    Returns
    -------
    K : numpy array
        The intrinsic matrix of the camera.
    """
    K = np.array([[fu, alpha_uv, u0],
                  [0, fv, v0],
                  [0, 0, 1]])
    return K

#def project_point()