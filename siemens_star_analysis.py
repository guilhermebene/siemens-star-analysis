import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def show_acquisition(recon_sum_norm, recon_sum, recon_sum_normalized, x0, y0):    
    
    plt.figure(figsize=(16,16), tight_layout=True)
    
    plt.subplot(131)
    plt.imshow(recon_sum_norm, cmap='gray')
    plt.title('Normalization image')

    plt.subplot(132)
    plt.imshow(recon_sum, cmap='gray')
    plt.scatter(x=y0,y=x0,s=8,color='r')
    plt.title('Acquired image')

    plt.subplot(133)
    plt.imshow(recon_sum_normalized, cmap='gray')
    plt.title('Normalized image')


def show_radii(img, x0, y0, R_MAX, R_MIN, title=None):
    
    plt.figure(figsize=(8,8))
    plt.imshow(img, cmap='gray')
    plt.scatter(x=y0, y=x0, s=4, color='r')

    if title is not None:
        plt.title(title)

    theta = np.arange(0,360,0.1)

    x = np.zeros(len(theta))
    y = np.zeros(len(theta))

    for index,angle in enumerate(theta*np.pi/180):
        x[index] = x0 + R_MAX*np.sin(angle)
        y[index] = y0 + R_MAX*np.cos(angle)

    plt.scatter(y,x,s=4)

    for index,angle in enumerate(theta*np.pi/180):
        x[index] = x0 + R_MIN*np.sin(angle)
        y[index] = y0 + R_MIN*np.cos(angle)

    plt.scatter(y,x,s=4)


def get_freq(radius, Np):
    '''
    Returns spatial frequency in cycles/pixel
    '''
    freq = Np/(2*np.pi*radius)
    return freq


def pix_to_mm(res_radius: int, siemens_radius: int, phys_mag: float, ext_r: int,
              zoom:int=1) -> float:
    
    return res_radius * siemens_radius * phys_mag / (ext_r * zoom)    


def calculate_lpmm(radius_pix: int, siemens_freq: int, siemens_radius: int,
                   phys_mag: float, ext_r: int, zoom:int=1) -> float:
    """Calculates resolution in linepairs per millimter (lp/mm).

    Args:
        radius_pix (int): 
            Radius in pixels at which resolution is determined.
        siemens_freq (int): 
            Amount of black black bars in the Siemens Star.
        siemens_radius (int): 
            Siemens Star radius in mm.
        phys_mag (float):
            Physical magnification due to the system's optics.
        ext_r (int): 
            External radius in pixels.
        zoom (int, optional): 
            Zoom applied to the acquisition. If present, the external radius 
            (ext_r) must be the same as in the image without zoom and this 
            function will calculate the correct external radius after zooming. 
            Defaults to 1.

    Returns:
        float: resolution in lp/mm.
    """
    
    
    radius_mm = pix_to_mm(radius_pix, siemens_radius, phys_mag, ext_r, zoom)
    theta = 2 * math.pi  / siemens_freq
    c = 2 * radius_mm * math.sin(theta/2)
    
    return 1/c


def calculate_contrast(maxima, minima):
    
    Imax = np.median(maxima)
    Imax_mean = np.mean(maxima)
    Imax_std = np.std(maxima)
    Imax_unc = Imax_std/np.sqrt(len(maxima))
    
    Imin = np.median(minima)
    Imin_mean = np.mean(minima)
    Imin_std = np.std(minima)
    Imin_unc = Imin_std/np.sqrt(len(minima))
    
    contrast = (Imax-Imin)/(Imax+Imin)
    
    dImax2 = (2*Imax/(Imax+Imin)**2)**2
    dImin2 = (2*Imin/(Imax+Imin)**2)**2
    
    contrast_unc = np.sqrt(dImax2 * (Imax_unc**2) + dImin2 * (Imin_unc**2))
    
    return contrast, contrast_unc, Imax, Imin


def object_resolution(res_radius: int, siemens_radius: int, siemens_freq: int,
                phys_mag: float, ext_r: int, zoom:int=1) -> float:
    """Calculates resolution in mm.

    Args:
        res_radius (int): 
            Radius in pixels at which resolution is determined.
        siemens_radius (int): 
            Siemens Star radius in mm.
        siemens_freq (int): 
            Amount of black black bars in the Siemens Star.
        phys_mag (float): 
            Physical magnification due to the system's optics.
        ext_r (int): 
            External radius in pixels.
        zoom (int, optional): 
            Zoom applied to the acquisition. If present, the external radius 
            (ext_r) must be the same as in the image without zoom and this 
            function will calculate the correct external radius after zooming. 
            Defaults to 1.

    Returns:
        float: 
            Real resolution at the object plane.
    """

    res_r = pix_to_mm(res_radius, siemens_radius, phys_mag, ext_r, zoom)
    res = 2 * np.pi * res_r / siemens_freq

    return res


def find_resolution(img, x0, y0, radii, interactive=False):

    d_theta = 0.0001
    theta = np.arange(0,2*np.pi, d_theta)

    d = int(10*np.pi/180/d_theta * 2/3)

    contrast = np.zeros(len(radii))
    contrast_unc = np.zeros(len(radii))
    
    for index, R in enumerate(radii):

        values = np.zeros(len(theta))
        x = np.around(x0 + R*np.cos(theta)).astype('int')
        y = np.around(y0 + R*np.sin(theta)).astype('int')
        
        for i in range(len(theta)):
            values[i] = img[x[i],y[i]]

        # Finding maxima and minima
        maxima,_ = find_peaks(values,distance=d)
        minima,_ = find_peaks(-values,distance=d)

        contrast[index],contrast_unc[index],Imax,Imin = calculate_contrast(
                                                            values[maxima],
                                                            values[minima])
        
        if interactive:
            plt.figure()
            plt.plot(theta, values, label='profile')
            plt.scatter(theta[maxima], values[maxima], label='maxima')
            plt.scatter(theta[minima], values[minima], label='minima')
            plt.axhline(Imax, label=f'median maximum = {Imax:.2f}')
            plt.axhline(Imin, label=f'median minimum = {Imin:.2f}')
            plt.xlabel('theta (rad)')
            plt.ylabel('Normalized intensity')
            plt.title(f'R={R} pix, contrast={contrast[index]:.3f}')
            plt.legend()
            plt.waitforbuttonpress()
            plt.close()

    ind = np.abs(contrast - 0.1).argmin()
    
    # Forcing the contrast to be at least 0.1
    if contrast[ind] < 0.1:
        ind += 1
    
    res_radius = radii[ind]
    res_MTF = contrast[ind]

    print(f'Found resolution at R={res_radius} pix, MTF={res_MTF}')

    return res_radius, res_MTF, contrast, contrast_unc


def plot_MTF_radius(radii, contrast, contrast_unc=None):

    plt.figure()
    plt.plot(radii,contrast, label='MTF')
    plt.axhline(0.1, label='Resolution limit') # Resolution limit at 10% of the MTF
    
    if contrast_unc is not None:
        plt.errorbar(radii,contrast,yerr=contrast_unc, label='MTF error')
        
    plt.xlabel('Radius (pix)')
    plt.ylabel('Contrast')
    plt.legend()


def plot_MTF_freq(radii, contrast, contrast_unc=None):    
    
    freqs = [get_freq(R,36) for R in radii]
    
    plt.figure()
    plt.plot(freqs,contrast, label='MTF')
    plt.axhline(0.1, label='Resolution limit') # Resolution limit at 10% of the MTF
    
    if contrast_unc is not None:
        plt.errorbar(freqs,contrast,yerr=contrast_unc, label='MTF error')
        
    plt.xlabel('f (cycles/pixel)')
    plt.ylabel('Contrast')
    plt.title('MTF')
    plt.legend()


def reciprocal_func(x, A):
        return A/x


def resolution_curve_coeffs(zooms, resolutions):

    popt, pcov = curve_fit(reciprocal_func, zooms, resolutions)

    return popt[0]   