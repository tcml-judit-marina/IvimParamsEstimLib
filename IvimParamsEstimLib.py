"""
Last Update: August 14, 2020

Created during 2019-2020 as part of a final project towards BSc in biomedical engineering.
TCML lab at the Department of Biomedical Engineering, Technion - IIT, Haifa, Israel


@ authors: Judit Ben Ami & Marina Khizgilov
Under the supervision of Dr. Moti Frieman
Special thanks for Elad Rotman for phantom_simulation function.

"""

# ******************************************************************************************************************* #
# ******************************************************************************************************************* #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ IvimParamsEstimLib ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ******************************************************************************************************************* #
# ******************************************************************************************************************* #

"""
This is a multi-purpose Python library for IVIM parameters estimation. 
The model used for IVIM–based biexponential analysis is:
 For a specific b value ,b_k, the DW signal in the [i,j] pixel is modeled by:
 DW-MRI_signal_k[i,j] = B0[i,j] * { f[i,j]*exp[-b_k(D_star[i,j]+D[i,j])] + (1-f[i,j])*exp(-b_k*D[i,j]) }
 where:
     B0 ---------> The initial MRI image (the value of the signal for b=0, with only B0 MRI field)
     f ----------> Image of the relative area of pseudo-diffusion (perfusion) [a.u.]
     b_k --------> The b-value [sec/(mm^2)]
     D_star -----> Image of the pseudo-diffusion (perfusion) coefficient [(mm^2)/sec]
     D ----------> Image of the diffusion coefficient [(mm^2)/sec]
     
This library includes:
    •	4 IVIM parameters estimation algorithms: SEGb, SEG, LSQ, BSP.
    •	Phantom creating function. 
    •	Error calculation functions. 
"""

# imports
import numpy as np
from scipy.optimize import least_squares
import scipy.special as sps  # modified bessel function of the first kind
import math
import multiprocessing
import SimpleITK as sitk
import matplotlib.pyplot as plt
import time


# ***************************************************************************************************************** #
# ***************************************************************************************************************** #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PRIVATE FUNCTIONS SECTION !!! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ***************************************************************************************************************** #
# ***************************************************************************************************************** #

# define mathematical function
def __ivim_to_norm_dw_signal(b, D, D_star, Fp):
    # np.multiply multiply arguments element-wise.
    arg1 = np.multiply(-b, (D_star + D))
    arg2 = np.multiply(-b, D)
    return np.multiply(Fp, np.exp(arg1)) + np.multiply((1 - Fp), np.exp(arg2))


def __ivim_to_reconstruct_image(num_of_pixels, b_vals, b_num, estim_D, estim_D_star, estim_f):
    # Calculating the estimated images with 5 steps:
    # step 1: calculating the first exponent
    # step 2: calculating the product of the first exponent with f
    # step 3: calculating the second exponent
    # step 4: calculating the product of the second exponent with 1-f
    # step 5: sum of step 4 and step 2 (dimensions: num of b values x num of pixels)
    tmp1_images = np.exp(
        -np.matmul(np.reshape(b_vals, (b_num, 1)), np.reshape((estim_D_star + estim_D), (1, num_of_pixels))))
    tmp1_images = np.tile(estim_f, (b_num, 1)) * tmp1_images
    tmp2_images = np.exp(-np.matmul(np.reshape(b_vals, (b_num, 1)), np.reshape(estim_D, (1, num_of_pixels))))
    tmp2_images = np.tile((np.ones(num_of_pixels) - estim_f), (b_num, 1)) * tmp2_images
    final_images = tmp1_images + tmp2_images
    return final_images


def __rician(x, ne, sig):
    # sps.iv == Modified Bessel function of the first kind of real order.
    return (x / (sig ** 2)) * np.exp(-(x ** 2 + ne ** 2) / (2 * sig ** 2)) * sps.iv(2, x * ne / sig ** 2)


def __circle_index(out_radius, in_radius, sx=256, sy=256):
    x_center = sx / 2
    y_center = sy / 2
    # im_blank = np.zeros(sx * sy).reshape([sx, sy]) # for DEBUG
    x_vec = np.arange(sx).reshape(1, sx)
    y_vec = np.arange(sy).reshape(sy, 1)
    x_mat = np.repeat(x_vec, sy, axis=0)
    y_mat = np.repeat(y_vec, sx, axis=1)
    value_index_in = (x_mat - x_center) ** 2 + (y_mat - y_center) ** 2 >= in_radius ** 2
    value_index_out = (x_mat - x_center) ** 2 + (y_mat - y_center) ** 2 < out_radius ** 2
    return tuple(np.logical_and(value_index_in, value_index_out))


def __phantom(rads=[40, 80, 100], sx=256, sy=256):
    phantom = np.zeros(sx * sy).reshape([sx, sy])
    for i in range(len(rads)):
        if i == 0:
            phantom[[__circle_index(rads[i], 0, sx, sy)]] = [i + 1]
        else:
            phantom[[__circle_index(rads[i], rads[i - 1], sx, sy)]] = [i + 1]
    return phantom


def __lsq_fun_(lsq_D_star, signal, b, seg_D, seg_f):
    estim_signal = seg_f * np.exp(-b * (lsq_D_star[0] + seg_D)) + (1 - seg_f) * np.exp(-b * seg_D)
    return estim_signal - signal


def __seg_multi_(num_of_pixels, SEGb_estim_D_star, sub_200_images_values, sub_200_b_values, SEG_estim_D,
                 SEG_estim_f, SEG_estim_D_star, dbg_features):
    for i in range(num_of_pixels):
        lsq_D_star0 = np.array([SEGb_estim_D_star[i]])
        try:
            lsq_D_star_res = least_squares(__lsq_fun_, lsq_D_star0, bounds=(0, 1.1), xtol=None, args=(
                sub_200_images_values[:, i], sub_200_b_values, SEG_estim_D[i], SEG_estim_f[i]))
            SEG_estim_D_star[i] = np.maximum(0, lsq_D_star_res.x[0])
        except ValueError:
            SEG_estim_D_star[i] = 0
        if dbg_features is True:
            seconds = time.time()
            print("Iterat num is ", i)


def __lsq_fun_all_(IVIM_params, signal, b):
    # IVIM_params[0] = D , IVIM_params[1] = D_star , IVIM_params[2] = f
    estim_signal = IVIM_params[2] * np.exp(-b * (IVIM_params[1] + IVIM_params[0])) + (
            1 - IVIM_params[2]) * np.exp(-b * IVIM_params[0])
    return estim_signal - signal


def __lsq_multi_(num_of_pixels, SEG_estim_D, SEG_estim_D_star, SEG_estim_f, work_images, b, LSQ_estim_D,
                 LSQ_estim_D_star, LSQ_estim_f, dbg_features):
    for i in range(num_of_pixels):
        IVIM_params = np.array([SEG_estim_D[i], SEG_estim_D_star[i],
                                SEG_estim_f[
                                    i]])  # IVIM_params[0] = D , IVIM_params[1] = D_star , IVIM_params[2] = f
        try:
            IVIM_params_res = least_squares(__lsq_fun_all_, IVIM_params, bounds=([0, 0, 0], [0.02, 1.1, 1]),
                                            xtol=None, args=(work_images[:, i], b))
            LSQ_estim_D[i] = np.maximum(0, IVIM_params_res.x[0])
            LSQ_estim_D_star[i] = np.maximum(0, IVIM_params_res.x[1])
            LSQ_estim_f[i] = np.maximum(0, IVIM_params_res.x[2])
        except ValueError:
            LSQ_estim_D[i] = 0
            LSQ_estim_D_star[i] = 0
            LSQ_estim_f[i] = 0
        if dbg_features is True:
            seconds = time.time()
            print("Iterat num is ", i, " Seconds since epoch = ", seconds)


def __evn_gauss_func(pixel_index, x_wid, y_wid, M, theta):
    # This function outputs a random guess for theta for pixel i

    # pixel_index%y_wid == row of the pixel
    # reminder of (pixel_index/y_wid) == col of the pixel
    pixel_row, pixel_col = divmod(pixel_index, y_wid)
    # Finding boundrys of env
    min_row = max(pixel_row - M, 0)
    max_row = min(pixel_row + M, y_wid)
    min_col = max(pixel_col - M, 0)
    max_col = min(pixel_col + M, x_wid)
    # Taking only rlv pixels (in env)
    rlv_env_theta = []
    for i in range(x_wid * y_wid):
        curr_row, curr_col = divmod(i, y_wid)
        if curr_row >= min_row and curr_row <= max_row and curr_col >= min_col and curr_col <= max_col:
            rlv_env_theta.append(theta[:, i])
    rlv_env_theta = np.transpose(np.array(rlv_env_theta))
    # Find distrb: calc avg & sigma
    mu_theta = np.mean(rlv_env_theta, axis=1)
    sigma_theta = np.cov(rlv_env_theta)
    # Random theta for pixel (returns 3 log_IVIM values)
    random_theta = np.random.multivariate_normal(mu_theta, sigma_theta, 1)
    return random_theta


def __plot_ivim_maps(D, D_star, f, method_name, plot_flag, save_plot_flag, save_plot_prefix):
    # D, D_star,f
    fig, ax = plt.subplots(1, 3, figsize=(20, 20))
    ax[0].imshow(D_star, cmap="inferno", clim=(0, 0.08))
    ax[0].set_title('D* pseudo diffusion', fontsize=20)
    ax[1].imshow(D, cmap="inferno", clim=(0, 0.002))
    ax[1].set_title('D diffusion', fontsize=20)
    ax[2].imshow(f, cmap="inferno", clim=(0, 1))
    ax[2].set_title('f fraction', fontsize=20)
    fig.suptitle(method_name+' IVIM Maps', x=0.5, y=0.83, fontsize=26)
    if save_plot_flag is True:
        plt.savefig(save_plot_prefix+'_ivim_maps.png', transparent=True)
    if plot_flag is True:
        plt.show(block=False)
    return


def __plot_dwmri_images(b_value, dwmri_images, method_name, plot_flag, save_plot_flag, save_plot_prefix):
    b_num = len(b_value)
    max_value = np.max(dwmri_images)
    fig, ax = plt.subplots(2, math.ceil(b_num / 2), figsize=(20, 20))
    b_id = 0
    _, b_rem = divmod(b_num, 2)
    for i in range(2):
        for j in range(math.ceil(b_num / 2)):
            if b_rem == 1 and b_id == b_num:
                if b_num in [1, 2]:
                    ax[i].axis('off')
                else:
                    ax[i, j].axis('off')
                break
            if b_num in [1, 2]:
                ax[i].imshow(dwmri_images[b_id, :, :], cmap='gray', clim=(0, max_value))
                ax[i].set_title("b-value = " + str(b_value[b_id]), fontsize=20)
                ax[i].set_xticks([])
                ax[i].set_yticks([])
                b_id += 1
            else:
                ax[i, j].imshow(dwmri_images[b_id, :, :], cmap='gray', clim=(0, max_value))
                ax[i, j].set_title("b-value = " + str(b_value[b_id]), fontsize=20)
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                b_id += 1
    fig.suptitle(method_name+' DWMRI Images', x=0.5, y=0.98, fontsize=26)
    if save_plot_flag is True:
        plt.savefig(save_plot_prefix+'_dwmri_images.png', transparent=True)
    if plot_flag is True:
        plt.show(block=False)


####################################################################################################################
####################################################################################################################
############################################## dwmri_images_to_ndarray #############################################
####################################################################################################################
####################################################################################################################
def dwmri_images_to_ndarray(file_path, b_value, slice_num=0, plot_flag=False, save_plot_flag=False,
                            save_plot_prefix="001"):
    """
    :param file_path:        Type: List. Size: b_num
                             Paths of the DW-MRI images to convert. All images must be the same size, images
                              needs to be in dim [z,y,x].
    :param b_value:          Type: ndarray, Size: [b_num,]
                             Vector containing the B-values the DW-MRI image was taken at.
    :param slice_num:        (Optional, Default: 0)
                             Type: ndarray, Shape: [b_num,]
                             For 3D Images, choose a specific 2D slice.
    :param plot_flag:        (Optional, Default: False)
                             Type: boolean
                             If True, plots the output figure.
    :param save_plot_flag:   (Optional, Default: False)
                             Type: boolean
                             If True, saves the output figure as a PNG file to current working 	directory.
    :param save_plot_prefix: (Optional, Default: False)
                             Type: boolean
                             Prefix of the name of the file the figure will be saved as.
    :return: slice:          Type: ndarray, shape [b_num, y_wid, x_wid]
                             Conversion result matrix.
    """

    # ########################################### input errors checker ########################################### #
    input_valid = 'valid'
    # file_path have yp be a list - checked by sitk
    if not np.all(np.greater_equal(b_value, 0)):
        print("b is invalid. All elements must be non-negative.")
        input_valid = 'error'
    # slice_num - checked by the line " nda[slice_num, :, :] "
    if plot_flag not in [True, False]:
        print("plot_flag value is invalid.")
        print("plot_flag must be True or False.")
        input_valid = 'error'
    if save_plot_flag not in [True, False]:
        print("save_plot_flag value is invalid.")
        print("save_plot_flag must be True or False.")
        input_valid = 'error'
    if type(save_plot_prefix) != str:
        print("save_plot_prefix value is invalid.")
        print("save_plot_prefix must be a string.")
        input_valid = 'error'
    # assert
    assert (input_valid is 'valid'), \
        "Error in IvimParamsEstimLib.dwmri_images_to_ndarray function. View printed comments."

    # ########################################### images to ndarray ########################################### #
    slice = []
    for i in range(len(file_path)):
        reader = sitk.ImageFileReader()
        reader.SetFileName(file_path[i])
        image = reader.Execute()
        nda = sitk.GetArrayFromImage(image)
        if nda.ndim == 2:
            slice.append(nda)
        elif nda.ndim == 3:
            slice.append(nda[slice_num, :, :])
        else:
            assert (
                    nda.ndim == 2), "Error in image dimensions. Dimensions of image in file must be 2 or 3."
    slice = np.array(slice)  # 1st dim: b num , 2st dim: y-axis, 3st dim: x-axis

    # plot
    if plot_flag is True or save_plot_flag is True:
        __plot_dwmri_images(b_value, slice, 'Original', plot_flag, save_plot_flag, save_plot_prefix)

    return slice


####################################################################################################################
####################################################################################################################
################################################ phantom_simulation ################################################
####################################################################################################################
####################################################################################################################
def phantom_simulation(b_val, D_val, D_star_val, Fp_val, B0_val, x_wid=64, y_wid=64, rads=None, noise_type='NaN',
                       SNR=0.1, plot_flag=False, save_plot_flag=False, save_plot_prefix="001"):
    """
    :param b_val:            Type: ndarray, Size: [b_num,]
                             Vector containing the B-values the DW-MRI image was taken at.
    :param D_star_val:       Type: ndarray, Size: [N+1,]
                             Vector containing the D values of the simulated image.
    :param D_val:            Type: ndarray, Size: [N+1,]
                             Vector containing the D* values of the simulated image.
    :param Fp_val:           Type: ndarray, Size: [N+1,]
                             Vector containing the f values of the simulated image.
    :param B0_val:           Type: ndarray, Size: [N+1,]
                             Vector containing the B0 values of the simulated image.
    :param x_wid:            (Optional, Default: 64)
                             Type: integer
                             Number of pixels at x-axis of simulated image.
    :param y_wid:            (Optional, Default: 64)
                             Type: integer
                             Number of pixels at y-axis of simulated image.
    :param rads:             (Optional, Default: [10,20,30])
                             Type: ndarray, Size: [N,]
                             Sort Vector containing the radii of the circles of simulate image.
                             The radii are in ascending order of length (all elements must be different), where
                             the value of the first index of the vector is the radius of the inner circle,
                             and Value of the last index of the vector is the radius of the external circle.
    :param noise_type:       (Optional, Default: 'NaN')
                             Type: string. Possible values: ‘NaN’, ‘gaussian’, ‘rayleigh’, ‘rice’,
                                                            ‘non_centralized_chi2’
                             Type of noise to be added to the simulated DW-MRI images.
    :param SNR:              (Optional, Default: 0.1)
                             Type: float
                             SNR value for noise.
    :param plot_flag:        (Optional, Default: False)
                             Type: boolean
                             If True, plots the output figure.
    :param save_plot_flag:   (Optional, Default: False)
                             Type: boolean
                             If True, saves the output figure as a PNG file to current working 	directory.
    :param save_plot_prefix: (Optional, Default: False)
                             Type: boolean
                             Prefix of the name of the file the figure will be saved as.
    :return: dwi_images:     Type: ndarray, shape [b_num,y_wid, x_wid]
                             Simulated DW-MRI images.
             B0_phantom:     Type: ndarray, shape [y_wid, x_wid]
                             Initial simulated DW-MRI image taken at b=0.
             D_phantom:      Type: ndarray, shape [y_wid, x_wid]
                             Simulated D map.
             D_phantom:      Type: ndarray, shape [y_wid, x_wid]
                             Simulated D* map.
             Fp_phantom:     Type: ndarray, shape [y_wid, x_wid]
                             Simulated f map.
    """

    # ########################################### input errors checker ########################################### #
    if type(b_val) not in [list, np.ndarray]:
        b_val = [b_val]
    if type(D_val) not in [list, np.ndarray]:
        D_val = [D_val]
    if type(D_star_val) not in [list, np.ndarray]:
        D_star_val = [D_star_val]
    if type(Fp_val) not in [list, np.ndarray]:
        Fp_val = [Fp_val]
    if type(B0_val) not in [list, np.ndarray]:
        B0_val = [B0_val]
    if rads is not None:
        if type(rads) not in [list, np.ndarray]:
            rads = [rads]

    input_valid = 'valid'
    if not np.all(np.greater_equal(b_val, 0)):
        print("b_val is invalid. All elements must be non-negative.")
        input_valid = 'error'
    if not np.all(np.greater_equal(D_val, 0)):
        print("IVIM parameter 'D_val' is invalid. Please enter a valid 'D_val'")
        print("All 'D_val' elements must be larger or equal to 0")
        input_valid = 'error'
    if not np.all(np.greater_equal(D_star_val, 0)):
        print("IVIM parameter 'D_star_val' is invalid. Please enter a valid 'D_star_val'")
        print("All 'D_star_val' elements must be larger or equal to 0")
        input_valid = 'error'
    if not np.all(np.greater_equal(Fp_val, 0)) and np.all(np.less_equal(Fp_val, 1)):
        print("IVIM parameter 'Fp_val' is invalid. Please enter a valid 'Fp_val'")
        print("All 'Fp_val' elements must be between 0 to 1")
        input_valid = 'error'
    if not np.all(np.greater_equal(B0_val, 0)):
        print("B0_val is invalid. All elements must be non-negative.")
        input_valid = 'error'
    if np.shape(D_val) != np.shape(D_star_val):
        print("D_val, D_star_val, Fp_val, D_val must be at the same length, and rads must have one element less.")
        input_valid = 'error'
    if np.shape(D_star_val) != np.shape(Fp_val):
        print("D_val, D_star_val, Fp_val, D_val must be at the same length, and rads must have one element less.")
        input_valid = 'error'
    if np.shape(Fp_val) != np.shape(B0_val):
        print("D_val, D_star_val, Fp_val, D_val must be at the same length, and rads must have one element less.")
        input_valid = 'error'
    if type(x_wid) != int or x_wid <= 0:
        print("x_wid value is invalid.")
        print("x_wid must be a positive int.")
        input_valid = 'error'
    if type(y_wid) != int or y_wid <= 0:
        print("y_wid value is invalid.")
        print("y_wid must be a positive int.")
        input_valid = 'error'
    if rads is not None:
        if not np.all(np.greater_equal(rads, 0)):
            print("rads is invalid. All elements must be non-negative.")
            input_valid = 'error'
        if len(rads) > 1:
            if not np.all(np.diff(rads) > 0):
                print("rads is invalid. rads must be a sorted vector in ascending order, "
                      "and all the elements must be different.")
                input_valid = 'error'
            if np.shape(rads) != np.shape(D_val)[0]-1:
                print(
                    "D_val, D_star_val, Fp_val, D_val must be at the same length, and rads must have one element less.")
                input_valid = 'error'
        else:
            if np.shape(D_val) == 1 and rads != [0]:
                print("rads must be 0 For 1 element in D_val/D_star_val/Fp_val/B0_val")
                input_valid = 'error'
            if np.shape(rads)[0] not in [np.shape(D_val)[0], np.shape(D_val)[0]-1]:
                print(
                    "D_val, D_star_val, Fp_val, D_val must be at the same length, and rads must have one element "
                    "less - Unless the length is 1 and in this case rads needs to be the value 0 (length 1)")
                input_valid = 'error'
    if noise_type not in ['NaN', 'gaussian', 'rayleigh', 'rice', 'non_centralized_chi2']:
        print("noise_type is invalid. Please enter a valid 'noise_type':")
        print("'gaussian or 'rayleigh' or 'rice' or 'non_centralized_chi2' or 'NaN'")
        input_valid = 'error'
    if SNR < 0:
        print("SNR is invalid. SNR must be non-negative.")
        input_valid = 'error'
    if plot_flag not in [True, False]:
        print("plot_flag value is invalid.")
        print("plot_flag must be True or False.")
        input_valid = 'error'
    if save_plot_flag not in [True, False]:
        print("save_plot_flag value is invalid.")
        print("save_plot_flag must be True or False.")
        input_valid = 'error'
    if type(save_plot_prefix) != str:
        print("save_plot_prefix value is invalid.")
        print("save_plot_prefix must be a string.")
        input_valid = 'error'
    # assert
    assert (input_valid is 'valid'), \
        "Error in IvimParamsEstimLib.phantom_simulation function. View printed comments."

    # ########################################### phantom simulation ########################################### #
    # ~~~~~~~~~~~ init phantom ~~~~~~~~~~~
    if rads is None:
        rads = [10, 20, 30]
        assert (np.shape(rads)[0] == np.shape(D_val)[0]-1), \
            "Default 'rads' length is 3 and its now being used. Please enter your own 'rads' vector, or make " \
            "sure that D_val, D_star_val, Fp_val, D_val are each in length 4."
    sx, sy, sb = x_wid, y_wid, len(b_val)

    D_star_phantom = __phantom(rads=rads, sx=sx, sy=sy)
    D_phantom      = __phantom(rads=rads, sx=sx, sy=sy)
    Fp_phantom     = __phantom(rads=rads, sx=sx, sy=sy)
    B0_phantom     = __phantom(rads=rads, sx=sx, sy=sy)

    # ~~~~~~~~~~~  create ivim parameters maps ~~~~~~~~~~~
    regions_num = len(D_val)
    for i in range(regions_num):
        D_phantom[D_phantom == i]           = D_val[i]
        D_star_phantom[D_star_phantom == i] = D_star_val[i]
        Fp_phantom[Fp_phantom == i]         = Fp_val[i]
        B0_phantom[B0_phantom == i]         = B0_val[i]

    # ~~~~~~~~~~~  create clean dw images ~~~~~~~~~~~
    # image parameter initialization
    b_image = np.ones(sx * sy).reshape([sx, sy])
    dwi_images = np.zeros((sx, sy, sb))
    for j in range(len(b_val)):
        dwi_images[:, :, j] = B0_phantom * (__ivim_to_norm_dw_signal(b_val[j] * b_image, D_phantom, D_star_phantom,
                                           Fp_phantom))

    # ~~~~~~~~~~~ add noise to dw images ~~~~~~~~~~~~~
    sig = SNR * np.max(B0_val)
    if noise_type == "gaussian":
        dwi_images = dwi_images + np.random.normal(loc=0.0, scale=sig, size=(sx, sy, sb))
    if noise_type == "rayleigh":
        meanvalue = 1
        modevalue = np.sqrt(2 / np.pi) * meanvalue
        dwi_images = dwi_images + np.random.rayleigh(scale=0.9, size=(sx, sy, sb))
    if noise_type == "rice":
        x = np.random.rand(sx, sy, sb)
        ne = 2
        dwi_images = dwi_images + __rician(6 * x, ne, sig)
    if noise_type == "non_centralized_chi2":
        df = 200  # degree of freedom
        mu = 20  # expectation
        nonc = df * (mu / sig) ** 2
        dwi_images = dwi_images + np.random.noncentral_chisquare(df, nonc, size=(sx, sy, sb))

    # ~~~~~~~~~~~ Change dw_images dimensions to be (b_num, y_wid, x_yid) ~~~~~~~~~~~~~
    dwi_images = np.moveaxis(dwi_images, -1, 0)
    dwi_images = np.abs(dwi_images)

    # ~~~~~~~~~~~ Plotting ~~~~~~~~~~~~~
    if plot_flag is True or save_plot_flag is True:
        # plot simulated dw_mri images
        __plot_ivim_maps(D_phantom, D_star_phantom, Fp_phantom, 'Phantom',
                         plot_flag, save_plot_flag, save_plot_prefix+'_phantom')
        __plot_dwmri_images(b_val, dwi_images, 'Phantom', plot_flag, save_plot_flag, save_plot_prefix+'_phantom')

    return dwi_images, B0_phantom, D_phantom, D_star_phantom, Fp_phantom


####################################################################################################################
####################################################################################################################
################################################# ivim_params_estim ################################################
####################################################################################################################
####################################################################################################################
def ivim_params_estim(dw_mri_images, b, sim_flag, estimator_mode='ALL', image_type_mode='absolute', b_tresh=200,
                      plot_flag=True, save_plot_flag=False, save_plot_prefix="001", multi_segs=8, env_size=5,
                      dbg_features=False):
    """
    :param dw_mri_images:    Type: ndarray, shape [b_num,y_wid, x_wid]
                             A 3D matrix consisting of a series of slices of DW-MRI images taken at b_num
                             different b-values.
    :param b:                Type: ndarray, shape [b_num,]
                             Vector containing the B-values the DW-MRI image was taken at.
    :param sim_flag:         Type: boolean
                             If True, the input images are from simulation meaning every pixel comes from IVIM
                             bi-exponential model. Else (if False) the input images are real clinical DWMRI images.
    :param estimator_mode:   (Optional, Default: 'absolute')
                             Type: string. Possible values: ‘SEGb’, ‘SEG’, ‘LSQ’ or ‘BSP’, ’ALL’
                             Type of IVIM parameters estimation algorithm. Choose ‘ALL’ to get estimation
                             results using all the modes: ‘SEGb’, ‘SEG’, ‘LSQ’ or ‘BSP’.
    :param image_type_mode:  (Optional, Default: 'absolute')
                             Type: string. Possible values: ‘norm’, ‘absolute’
                             DW-MRI image mode. ‘norm’ if image is normalized, ‘absolute’ if Image is
                             not normalized.
    :param b_tresh:          (Optional, Default: 200)
                             Type: integer
                             b-value threshold for SEG/SEGb estimations.
    :param plot_flag:        (Optional, Default: False)
                             Type: boolean
                             If True, plots the output figure.
    :param save_plot_flag:   (Optional, Default: False)
                             Type: boolean
                             If True, saves the output figure as a PNG file to current working 	directory.
    :param save_plot_prefix: (Optional, Default: False)
                             Type: boolean
    :param multi_segs:       (Optional, Default: 8)
                             Type: positive integer
                             For multi-processing implementation. Maximum number of multi-processes.
    :param env_size:         (Optional, Default: 5)
                             Type: positive integer
                             For ‘BSP’ mode, number of elements in each direction around a pixel.
    :param dbg_features:     (Optional, Default: False)
                             If True, prints yje number of each iteration and total running time for each
                             method except SEGb.
    :return: <estimator_mode>_estim_D :       Type: ndarray (or list for ‘ALL’ estimator)
                                              Estimated D map, using <estimator_mode> algorithm.
             <estimator_mode>_estim_D_star :  Type: ndarray (or list for ‘ALL’ estimator)
                                              Estimated D* map, using <estimator_mode> algorithm.
             <estimator_mode>_estim_f :       Type: ndarray (or list for ‘ALL’ estimator)
                                              Estimated f map, using <estimator_mode> algorithm.
             <estimator_mode>_ estim_images : Type: ndarray (or list for ‘ALL’ estimator)
                                              Reconstructed images.

    """

    # ########################################### input errors checker ########################################### #
    input_valid = 'valid'
    if not np.all(np.greater_equal(dw_mri_images, 0)):
        print("dw_mri_image is invalid. All elements must be non-negative.")
        input_valid = 'error'
    if not np.all(np.greater_equal(b, 0)):
        print("b is invalid. All elements must be non-negative.")
        input_valid = 'error'
    if sim_flag not in [True, False]:
        print("sim_flag value is invalid.")
        print("sim_flag must be True or False.")
        input_valid = 'error'
    if not np.shape(dw_mri_images)[0] == np.shape(b)[0]:
        print("'dw_mri_images' dimensions dose not match b length")
        input_valid = 'error'
    if image_type_mode not in ['norm', 'absolute']:
        print("'image_type_mode' is invalid. Please enter a valid 'image_type_mode':")
        print("'norm' or 'absolute'")
        input_valid = 'error'
    if estimator_mode not in['BSP', 'LSQ', 'SEG', 'SEGb', 'ALL']:
        print("estimator_mode is invalid. Please enter a valid 'estimator_mode':")
        print("'BSP or 'LSQ' or 'SEG' or 'SEGb' or 'ALL")
        input_valid = 'error'
    if type(b_tresh) not in [int, float]:
        print("'b_tresh' must be int or float")
        input_valid = 'error'
    if plot_flag not in [True, False]:
        print("plot_flag value is invalid.")
        print("plot_flag must be True or False.")
        input_valid = 'error'
    if save_plot_flag not in [True, False]:
        print("save_plot_flag value is invalid.")
        print("save_plot_flag must be True or False.")
        input_valid = 'error'
    if type(save_plot_prefix) != str:
        print("save_plot_prefix value is invalid.")
        print("save_plot_prefix must be a string.")
        input_valid = 'error'
    if multi_segs < 1 or (type(multi_segs) != int):
        print("multi_segs must be a natural number.")
        input_valid = 'error'
    if env_size < 0 or (type(env_size) != int):
        print("env_size must be non-negative int.")
        input_valid = 'error'
    if dbg_features not in [True, False]:
        print("dbg_features value is invalid.")
        print("dbg_features must be True or False.")
        input_valid = 'error'

    assert (
            input_valid is 'valid'), "Error in signal_parameter_estimations function. View printed comments."

    # ########################################################################################################### #
    # =============== variables preparation =============== #
    b_num = len(b)
    b = np.array(b)
    B0_image = dw_mri_images[np.argmin(b)]

    # prep images to analysis
    work_images = dw_mri_images.copy()
    work_images = work_images + 1
    work_images = np.float64(work_images)
    work_images[work_images == 0] = 0.000000001
    B0_image_for_div = B0_image.copy()
    B0_image_for_div = B0_image_for_div + 1
    B0_image_for_div = np.float64(B0_image_for_div)
    B0_image_for_div[B0_image_for_div == 0] = 0.000000001
    for b_id in range(b_num):
        work_images[b_id, :, :] = work_images[b_id, :, :] / B0_image_for_div
    work_images[work_images > 1] = 1
    work_images[work_images == 0] = 0.000000001
    # main properties
    x_wid = work_images.shape[2]
    y_wid = work_images.shape[1]
    num_of_pixels = x_wid * y_wid
    # prep images to analysis
    work_B0_image = np.reshape(B0_image.copy(), (x_wid * y_wid))
    work_images = np.reshape(work_images, (b_num, x_wid * y_wid))
    if sim_flag is True:
        B0_image_min_real_val = -1
    else:
        B0_image_min_real_val = 50
    sub_tresh_b_values = b[b <= b_tresh]
    sub_tresh_images_values = work_images[b <= b_tresh]
    sup_tresh_b_values = b[b > b_tresh]
    sup_tresh_images_values = work_images[b > b_tresh]

    # ################################################### SEGb ################################################### #
    # =============== for b>b_tresh: =============== #
    # dw_mri_signal = (1-f)*exp(-b*D)
    # ln(dw_mri_signal) = ln[(1-f)*exp(-b*D)]
    # ln(dw_mri_signal) = ln(1-f)+ln[exp(-b*D)]
    # ln(dw_mri_signal) = -b*D+ln(1-f)
    # so we can find the line that best fits to ln(dw_mri_signal) vs b:
    sup_p = np.polyfit(x=sup_tresh_b_values, y=np.log(sup_tresh_images_values), deg=1)
    sup_tresh_sloops = sup_p[0]
    sup_tresh_y_intersections = sup_p[1]

    # from the above sloop and intersection we will find:
    #   D : from the sloop: the neg of the sloop
    #   f:  from the intersection  :
    #       y_intersection = ln(1-f)
    #       exp(y_intersection) = 1-f
    #       f = 1-exp(y_intersection)
    SEGb_estim_D = np.maximum(0, -sup_tresh_sloops)
    SEGb_estim_f = np.maximum(0, 1 - np.exp(sup_tresh_y_intersections))
    # =============== for b<tresh: =============== #
    # dw_mri_signal = f*exp[-b(D_star+D)]
    # ln(dw_mri_signal) = ln(f*exp[-b(D_star+D)])
    # ln(dw_mri_signal) =  - b(D_star + D) + ln(f)
    # so we can find the line that best fits to ln(dw_mri_signal) vs b:
    sub_p = np.polyfit(x=sub_tresh_b_values, y=np.log(sub_tresh_images_values), deg=1)
    sub_tresh_sloops = sub_p[0]

    # from the above sloop and intersection we will find:
    # D_star : from the sloop:
    #          sloop =  -(D_star+D)
    #          D_star = -D-sloop
    SEGb_estim_D_star = np.maximum(0, -SEGb_estim_D - sub_tresh_sloops)

    # =============== roughly rm background =============== #
    SEGb_estim_D[work_B0_image < B0_image_min_real_val] = 0
    SEGb_estim_f[work_B0_image < B0_image_min_real_val] = 0
    SEGb_estim_D_star[work_B0_image < B0_image_min_real_val] = 0

    # =============== reconstruct image =============== #
    SEGb_estim_images = __ivim_to_reconstruct_image(num_of_pixels, b, b_num, SEGb_estim_D, SEGb_estim_D_star, SEGb_estim_f)

    # =============== plot and return =============== #
    # ~~~~~ save spread form ~~~~~ #
    if estimator_mode is not 'SEGb':
        SEGb_estim_spread_f         = SEGb_estim_f
        SEGb_estim_spread_D         = SEGb_estim_D
        SEGb_estim_spread_D_star    = SEGb_estim_D_star
        SEGb_estim_spread_images    = SEGb_estim_images
    if estimator_mode is 'SEGb' or estimator_mode is 'ALL':
        # ~~~~~ images reshape ~~~~~ #
        # Reshaping everything back to real images dimensions. The result is 3D matrix - A series of N estimated images
        SEGb_estim_f = np.reshape(SEGb_estim_f, (y_wid, x_wid))
        SEGb_estim_D = np.reshape(SEGb_estim_D, (y_wid, x_wid))
        SEGb_estim_D_star = np.reshape(SEGb_estim_D_star, (y_wid, x_wid))
        SEGb_estim_images = np.reshape(SEGb_estim_images, (b_num, y_wid, x_wid))
        for b_id in range(b_num):
            SEGb_estim_images[b_id, :, :] = SEGb_estim_images[b_id, :, :] * B0_image
        # ~~~~~ plotting ~~~~~ #
        if plot_flag is True or save_plot_flag is True:
            __plot_ivim_maps(SEGb_estim_D, SEGb_estim_D_star, SEGb_estim_f, 'SEGb', plot_flag, save_plot_flag,
                             save_plot_prefix+'_SEGb_estim')
            __plot_dwmri_images(b, SEGb_estim_images, 'SEGb', plot_flag, save_plot_flag, save_plot_prefix+'_SEGb_rec')
    # ~~~~~ return ~~~~~ #
    if estimator_mode is 'SEGb':
        return SEGb_estim_D, SEGb_estim_D_star, SEGb_estim_f, SEGb_estim_images

    # ################################################### SEG #################################################### #
    # =============== for b>b_tresh: =============== #
    # polyfit - Identical to SEGb
    SEG_estim_D = SEGb_estim_spread_D
    SEG_estim_f = SEGb_estim_spread_f

    # =============== for b<b_tresh - with multiprocessing =============== #
    # dw_mri_signal = f * exp[-b_i(D_star + D)] + (1 - f) * exp(-b_i * D)
    # when now D and f are known.
    # We'll find D_star by LSQ.
    seg_length = num_of_pixels // multi_segs
    seg_length_res = np.remainder(num_of_pixels, multi_segs)
    SEG_estim_D_star_segs = []
    p_seg = []
    if dbg_features is True:
        SEG_start_time = time.time()
        SEG_local_start_time = time.ctime(SEG_start_time)
        print("Local time before loop: ", SEG_local_start_time)
    for i in range(multi_segs):
        if i == multi_segs - 1:
            SEG_estim_D_star_segs.append(multiprocessing.Array('d', np.empty([seg_length + seg_length_res])))
            p_seg.append(multiprocessing.Process(target=__seg_multi_,
                                                 args=(seg_length + seg_length_res,
                                                       SEGb_estim_spread_D_star[i * seg_length:num_of_pixels],
                                                       sub_tresh_images_values[:, i * seg_length:num_of_pixels],
                                                       sub_tresh_b_values,
                                                       SEG_estim_D[i * seg_length:num_of_pixels],
                                                       SEG_estim_f[i * seg_length:num_of_pixels],
                                                       SEG_estim_D_star_segs[i], dbg_features)))
            p_seg[i].start()
        else:
            SEG_estim_D_star_segs.append(multiprocessing.Array('d', np.empty([seg_length])))
            p_seg.append(multiprocessing.Process(target=__seg_multi_,
                                                 args=(
                                                     seg_length,
                                                     SEGb_estim_spread_D_star[i * seg_length:(i + 1) * seg_length],
                                                     sub_tresh_images_values[:, i * seg_length:(i + 1) * seg_length],
                                                     sub_tresh_b_values,
                                                     SEG_estim_D[i * seg_length:(i + 1) * seg_length],
                                                     SEG_estim_f[i * seg_length:(i + 1) * seg_length],
                                                     SEG_estim_D_star_segs[i], dbg_features)))
            p_seg[i].start()

    for i in range(multi_segs):
        p_seg[i].join()
    if dbg_features is True:
        SEG_end_time = time.time()
        SEG_local_end_time = time.ctime(SEG_end_time)
        SEG_total_loop_time = (SEG_end_time - SEG_start_time) / 60
        print("Total SEG loop time: " + "{:.2f}".format(SEG_total_loop_time), " minutes")

    # =============== arrange results =============== #
    SEG_estim_D_star = []
    for i in range(multi_segs):
        SEG_estim_D_star = np.concatenate([np.array([SEG_estim_D_star]), np.array([SEG_estim_D_star_segs[i]])],
                                          axis=None)

    # =============== roughly rm background =============== #
    SEG_estim_D_star[work_B0_image < B0_image_min_real_val] = 0

    # =============== reconstruct image =============== #
    SEG_estim_images = __ivim_to_reconstruct_image(num_of_pixels, b, b_num, SEG_estim_D, SEG_estim_D_star, SEG_estim_f)

    # =============== plots and return =============== #
    # ~~~~~ save spread form ~~~~~ #
    if estimator_mode is not 'SEG':
        SEG_estim_spread_f         = SEG_estim_f
        SEG_estim_spread_D         = SEG_estim_D
        SEG_estim_spread_D_star    = SEG_estim_D_star
        SEG_estim_spread_images    = SEG_estim_images
    if estimator_mode is 'SEG' or estimator_mode is 'ALL':
        # ~~~~~ images reshape ~~~~~ #
        # Reshaping everything back to real images dimensions. The result is 3D matrix - A series of N estimated images
        SEG_estim_f = np.reshape(SEG_estim_f, (y_wid, x_wid))
        SEG_estim_D = np.reshape(SEG_estim_D, (y_wid, x_wid))
        SEG_estim_D_star = np.reshape(SEG_estim_D_star, (y_wid, x_wid))
        SEG_estim_images = np.reshape(SEG_estim_images, (b_num, y_wid, x_wid))
        for b_id in range(b_num):
            SEG_estim_images[b_id, :, :] = SEG_estim_images[b_id, :, :] * B0_image
        # ~~~~~ plotting ~~~~~ #
        if plot_flag is True or save_plot_flag is True:
            __plot_ivim_maps(SEG_estim_D, SEG_estim_D_star, SEG_estim_f, 'SEG', plot_flag, save_plot_flag,
                             save_plot_prefix+'_SEG_estim')
            __plot_dwmri_images(b, SEG_estim_images, 'SEG', plot_flag, save_plot_flag, save_plot_prefix+'_SEG_rec')
    # ~~~~~ return ~~~~~ #
    if estimator_mode is 'SEG':
        return SEG_estim_D, SEG_estim_D_star, SEG_estim_f, SEG_estim_images


    # ################################################### LSQ #################################################### #
    # =============== finding ivim params by LSQ - with multiprocessing =============== #
    LSQ_estim_D_segs = []
    LSQ_estim_D_star_segs = []
    LSQ_estim_f_segs = []
    p_seg = []
    if dbg_features is True:
        LSQ_start_time = time.time()
        LSQ_local_start_time = time.ctime(LSQ_start_time)
        print("Local time before loop: ", LSQ_local_start_time)
    for i in range(multi_segs):
        if i == multi_segs - 1:
            LSQ_estim_D_segs.append(multiprocessing.Array('d', np.empty([seg_length + seg_length_res])))
            LSQ_estim_D_star_segs.append(multiprocessing.Array('d', np.empty([seg_length + seg_length_res])))
            LSQ_estim_f_segs.append(multiprocessing.Array('d', np.empty([seg_length + seg_length_res])))
            p_seg.append(multiprocessing.Process(target=__lsq_multi_,
                                                 args=(seg_length + seg_length_res,
                                                       SEG_estim_spread_D[(i * seg_length):num_of_pixels],
                                                       SEG_estim_spread_D_star[(i * seg_length):num_of_pixels],
                                                       SEG_estim_spread_f[(i * seg_length):num_of_pixels],
                                                       work_images[:, (i * seg_length):num_of_pixels],
                                                       b, LSQ_estim_D_segs[i], LSQ_estim_D_star_segs[i],
                                                       LSQ_estim_f_segs[i], dbg_features)))
            p_seg[i].start()
        else:
            LSQ_estim_D_segs.append(multiprocessing.Array('d', np.empty([seg_length])))
            LSQ_estim_D_star_segs.append(multiprocessing.Array('d', np.empty([seg_length])))
            LSQ_estim_f_segs.append(multiprocessing.Array('d', np.empty([seg_length])))
            p_seg.append(multiprocessing.Process(target=__lsq_multi_,
                                                 args=(seg_length,
                                                       SEG_estim_spread_D[(i * seg_length):((i + 1) * seg_length)],
                                                       SEG_estim_spread_D_star[(i * seg_length):((i + 1) * seg_length)],
                                                       SEG_estim_spread_f[(i * seg_length):((i + 1) * seg_length)],
                                                       work_images[:, (i * seg_length):((i + 1) * seg_length)],
                                                       b, LSQ_estim_D_segs[i], LSQ_estim_D_star_segs[i],
                                                       LSQ_estim_f_segs[i], dbg_features)))
            p_seg[i].start()

    for i in range(multi_segs):
        p_seg[i].join()
    if dbg_features is True:
        LSQ_end_time = time.time()
        LSQ_local_end_time = time.ctime(LSQ_end_time)
        print("Local time after loop: ", LSQ_local_end_time)
        LSQ_total_loop_time = (LSQ_end_time - LSQ_start_time) / 60
        print("Total SEG loop time: " + "{:.2f}".format(SEG_total_loop_time), " minutes")
        print("Total LSQ loop time: " + "{:.2f}".format(LSQ_total_loop_time), " minutes")

    # =============== arrange results =============== #
    LSQ_estim_D = []
    LSQ_estim_D_star = []
    LSQ_estim_f = []
    for i in range(multi_segs):
        LSQ_estim_D = np.concatenate([np.array([LSQ_estim_D]), np.array([LSQ_estim_D_segs[i]])],
                                     axis=None)
        LSQ_estim_D_star = np.concatenate([np.array([LSQ_estim_D_star]), np.array([LSQ_estim_D_star_segs[i]])],
                                          axis=None)
        LSQ_estim_f = np.concatenate([np.array([LSQ_estim_f]), np.array([LSQ_estim_f_segs[i]])],
                                     axis=None)

    # =============== roughly rm background =============== #
    LSQ_estim_D[work_B0_image < B0_image_min_real_val] = 0
    LSQ_estim_D_star[work_B0_image < B0_image_min_real_val] = 0
    LSQ_estim_f[work_B0_image < B0_image_min_real_val] = 0

    # =============== reconstruct image =============== #
    LSQ_estim_images = __ivim_to_reconstruct_image(num_of_pixels, b, b_num, LSQ_estim_D, LSQ_estim_D_star, LSQ_estim_f)

    # =============== plots and return =============== #
    if estimator_mode is not 'LSQ':
        LSQ_estim_spread_f         = np.copy(LSQ_estim_f)
        LSQ_estim_spread_D         = np.copy(LSQ_estim_D)
        LSQ_estim_spread_D_star    = np.copy(LSQ_estim_D_star)
        LSQ_estim_spread_images    = np.copy(LSQ_estim_images)
    if estimator_mode is 'LSQ' or estimator_mode is 'ALL':
        # ~~~~~ images reshape ~~~~~ #
        # Reshaping everything back to real images dimensions. The result is 3D matrix - A series of N estimated images
        LSQ_estim_f = np.reshape(LSQ_estim_f, (y_wid, x_wid))
        LSQ_estim_D = np.reshape(LSQ_estim_D, (y_wid, x_wid))
        LSQ_estim_D_star = np.reshape(LSQ_estim_D_star, (y_wid, x_wid))
        LSQ_estim_images = np.reshape(LSQ_estim_images, (b_num, y_wid, x_wid))
        for b_id in range(b_num):
            LSQ_estim_images[b_id, :, :] = LSQ_estim_images[b_id, :, :] * B0_image
        # ~~~~~ plotting ~~~~~ #
        if plot_flag is True or save_plot_flag is True:
            __plot_ivim_maps(LSQ_estim_D, LSQ_estim_D_star, LSQ_estim_f, 'LSQ', plot_flag, save_plot_flag,
                             save_plot_prefix+'_LSQ_estim')
            __plot_dwmri_images(b, LSQ_estim_images, 'LSQ', plot_flag, save_plot_flag, save_plot_prefix+'_LSQ_rec')
    # ~~~~~ return ~~~~~ #
    if estimator_mode is 'LSQ':
        return LSQ_estim_D, LSQ_estim_D_star, LSQ_estim_f, LSQ_estim_images


    # ################################################### BSP ################################################### #
    # Parameters of the method:
    #   M - an environment of M pixels around the target pixel (square environment).
    #   r - random parameter for threshold for alpha, that will determine if we reject or accept.

    # steps:
    # 1. Run LSQ on each pixel (already done).
    # 2. Transform IVIM parameters into log form:
    #     D      --> d_log=log(D)
    #     D_star --> d_star_log=log(D_star)
    #     f      --> f_log=log(f)-log(1-f)=log(f/(1-f))
    # 3. Iterative algorithm:
    #    For each pixel:
    #    3.1. Calculating the average (mu) and co-variance matrix (sigma) for the current log_IVIM parameters, of
    #         the M-size-environment of each pixel.
    #    3.2. From a multivariate Gaussian distribution with mu and sigma: randomly chose the log_IVIM parameters
    #         values for each pixel.
    #    3.3. Now we need to chose if we want to update the log_IVIM parameters from the previous iteration to
    #         the ones we have randomly chosen in current iteration.
    #         So for both of them we will calculate the estimated images for all the b-values:
    #         3.3.1. Finding the new IVIM parameters maps:
    #                   d_log       --> D=exp(d_log)
    #                   d_star_log  --> D_star=exp(d_star_log)
    #                   f_log       --> f=exp(f_log)/(1+exp(f_log))
    #         3.3.2. Finding the estimated images, g:
    #                   g = f * exp[-b(D_star + D)] + (1 - f) * exp(-b*D)
    #         3.3.3. Calculating the distribution p(theta|y) for each pixel:
    #                   y - the original images values. Column vector in size b_num.
    #                       yt == y transpose
    #                   g - Calculated images values. Column vector in size b_num.
    #                       gt == g transpose
    #                   b_num - number of b-values
    #                   p(theta|y) ~ (yt*y-((yt*g)^2)/(gt*g))^(-b_num/2) [This value is a scalar!]
    #    3.4. Calculating alpha:
    #         alpha = min(1, [p(theta_prev_iteration|y)/p(theta_curr_rand|y)] )
    #    3.5. Randomly chose r from a uniform distribution in [0,1].
    #    3.6. If alpha>r -> reject and don't update theta.
    #                       Else, updating theta to the current randomly chosen value.
    #    3.7. Check stopping condition:
    #         If we have reached 100 iterations, or if we have reached a convergence condition.

    # ~~~~~~~~~~~~~~~~~~ Step 1: init bsp results ~~~~~~~~~~~~~~~~~~
    tmp_BSP_D = LSQ_estim_spread_D
    tmp_BSP_D_star = LSQ_estim_spread_D_star
    tmp_BSP_f = LSQ_estim_spread_f
    tmp_BSP_D[tmp_BSP_D == 0]           = 0.0000001
    tmp_BSP_D_star[tmp_BSP_D_star == 0] = 0.0000001
    tmp_BSP_f[tmp_BSP_f == 0]           = 0.0000001
    BSP_cnvrt_tresh = 0.0000001
    # ~~~~~~~~~~~~~~~~~~ Step 2: log transform ~~~~~~~~~~~~~~~~~~
    tmp_BSP_d_log = np.log(tmp_BSP_D)
    tmp_BSP_d_star_log = np.log(tmp_BSP_D_star)
    tmp_BSP_f_log = np.log(tmp_BSP_f) - np.log(1 - tmp_BSP_f)

    # ~~~~~~~~~~~~~~~~~~ Step 3: main bsp loop ~~~~~~~~~~~~~~~~~~
    bsp_theta = np.concatenate(
        [np.array([tmp_BSP_d_log]), np.array([tmp_BSP_d_star_log]), np.array([tmp_BSP_f_log])])
    bsp_ivim = np.concatenate(
        [np.array([tmp_BSP_D]), np.array([tmp_BSP_D_star]), np.array([tmp_BSP_f])])
    bsp_re_image = np.copy(LSQ_estim_spread_images)
    bsp_prob = np.zeros(num_of_pixels)
    done_mask = np.zeros(num_of_pixels)
    done_mask[work_B0_image < B0_image_min_real_val] = 1
    for j in range(num_of_pixels):
        if done_mask is 0:
            bsp_prob_1 = np.dot(work_images[:, j].T, work_images[:, j]) - np.dot(work_images[:, j].T, bsp_re_image[:, j])
            if bsp_prob_1 < BSP_cnvrt_tresh: # if np.abs(bsp_prob_1) < BSP_cnvrt_tresh:
                done_mask[j] = 1
                continue
            bsp_prob_2 = np.dot(bsp_re_image[:, j].T, bsp_re_image[:, j])
            if bsp_prob_2 == 0:
                bsp_prob_2 = bsp_prob_2 + 0.0001
            bsp_prob[j] = (bsp_prob_1 ** 2 / bsp_prob_2) ** (-b_num / 2)
    if dbg_features is True:
        BSP_start_time = time.time()
        BSP_local_start_time = time.ctime(BSP_start_time)
        print("Local time before loop: ", BSP_local_start_time)
        dbg_alpha_list = []
        dbg_r_list = []
    BSP_max_iterations = 100
    for i in range(BSP_max_iterations):
        if dbg_features is True:
            print("BSP: num of done pixels before", i, "th iteration: ", np.sum(done_mask))
        for j in range(num_of_pixels):
            if done_mask[j] == 1:
                continue
            tmp_bsp_rnd_guess = __evn_gauss_func(pixel_index=j, x_wid=x_wid, y_wid=y_wid, M=env_size, theta=bsp_theta)
            tmp_bsp_rnd_ivim = np.copy(tmp_bsp_rnd_guess)
            tmp_bsp_rnd_ivim[:, 0] = np.exp(tmp_bsp_rnd_ivim[:, 0])  # D
            tmp_bsp_rnd_ivim[:, 1] = np.exp(tmp_bsp_rnd_ivim[:, 1])  # D_star
            tmp_bsp_rnd_ivim[:, 2] = np.exp(tmp_bsp_rnd_ivim[:, 2]) / (1 + np.exp(tmp_bsp_rnd_ivim[:, 2]))  # f
            tmp_bsp_rnd_rec_pxl = tmp_bsp_rnd_ivim[:, 2] * np.exp(
                -b * (tmp_bsp_rnd_ivim[:, 1] + tmp_bsp_rnd_ivim[:, 0])) + (
                                          1 - tmp_bsp_rnd_ivim[:, 2]) * np.exp(-b * tmp_bsp_rnd_ivim[:, 0])

            tmp_bsp_prob_1 = np.dot(work_images[:, j].T, work_images[:, j]) - np.dot(work_images[:, j].T,
                                                                                         tmp_bsp_rnd_rec_pxl)
            if tmp_bsp_prob_1 < BSP_cnvrt_tresh: # if np.abs(tmp_bsp_prob_1) < BSP_cnvrt_tresh:
                done_mask[j] = 1
                continue
            tmp_bsp_prob_2 = np.dot(tmp_bsp_rnd_rec_pxl.T, tmp_bsp_rnd_rec_pxl)
            if tmp_bsp_prob_2 == 0:
                tmp_bsp_prob_2 = tmp_bsp_prob_2 + 0.0001

            tmp_bsp_rnd_prob = (tmp_bsp_prob_1 ** 2 / tmp_bsp_prob_2) ** (-b_num / 2)
            tmp_bsp_alpha = min(1, (bsp_prob[j] / tmp_bsp_rnd_prob))
            if dbg_features is True:
                dbg_alpha_list = dbg_alpha_list + [tmp_bsp_alpha]
            tmp_bsp_r = np.random.uniform(0, 1, 1)
            if dbg_features is True:
                dbg_r_list = dbg_r_list + [tmp_bsp_r]

            if tmp_bsp_alpha <= tmp_bsp_r:
                # update choice
                bsp_theta[:, j] = tmp_bsp_rnd_guess
                bsp_ivim[:, j] = tmp_bsp_rnd_ivim
                bsp_re_image[:, j] = tmp_bsp_rnd_rec_pxl
                bsp_prob[j] = tmp_bsp_rnd_prob
        if dbg_features is True:
            print("BSP Iterat num is ", i)
        if np.all (done_mask == 1):
            if dbg_features is True:
                print("BSP converts in ", i, "th iteration out of 100 - Get out of loop")
            break

    if dbg_features is True:
        BSP_end_time = time.time()
        BSP_local_end_time = time.ctime(BSP_end_time)
        print("Local time after loop: ", BSP_local_end_time)
        BSP_total_loop_time = (BSP_end_time - BSP_start_time) / 60
        print("Total SEG loop time: " + "{:.2f}".format(SEG_total_loop_time), " minutes")
        print("Total LSQ loop time: " + "{:.2f}".format(LSQ_total_loop_time), " minutes")
        print("Total BSP loop time: " + "{:.2f}".format(BSP_total_loop_time), " minutes")

    # =============== arrange results =============== #
    BSP_estim_D = bsp_ivim[0, :]
    BSP_estim_D_star = bsp_ivim[1, :]
    BSP_estim_f = bsp_ivim[2, :]

    # =============== roughly rm background =============== #
    BSP_estim_D[work_B0_image < B0_image_min_real_val] = 0
    BSP_estim_D_star[work_B0_image < B0_image_min_real_val] = 0
    BSP_estim_f[work_B0_image < B0_image_min_real_val] = 0

    # =============== reconstruct image =============== #
    BSP_estim_images = __ivim_to_reconstruct_image(num_of_pixels, b, b_num, BSP_estim_D, BSP_estim_D_star, BSP_estim_f)

    # =============== plots and return =============== #
    if estimator_mode is 'BSP' or estimator_mode is 'ALL':
        # ~~~~~ images reshape ~~~~~ #
        # Reshaping everything back to real images dimensions. The result is 3D matrix - A series of N estimated images
        BSP_estim_f = np.reshape(BSP_estim_f, (y_wid, x_wid))
        BSP_estim_D = np.reshape(BSP_estim_D, (y_wid, x_wid))
        BSP_estim_D_star = np.reshape(BSP_estim_D_star, (y_wid, x_wid))
        BSP_estim_images = np.reshape(BSP_estim_images, (b_num, y_wid, x_wid))
        for b_id in range(b_num):
            BSP_estim_images[b_id, :, :] = BSP_estim_images[b_id, :, :] * B0_image
        # ~~~~~ plotting ~~~~~ #
        if plot_flag is True or save_plot_flag is True:
            # plot simulated dw_mri images
            __plot_ivim_maps(BSP_estim_D, BSP_estim_D_star, BSP_estim_f, 'BSP', plot_flag, save_plot_flag,
                             save_plot_prefix+'_BSP_estim')
            __plot_dwmri_images(b, BSP_estim_images, 'BSP', plot_flag, save_plot_flag, save_plot_prefix+'_BSP_rec')
    # ~~~~~ return ~~~~~ #
    if estimator_mode is 'BSP':
        return BSP_estim_D, BSP_estim_D_star, BSP_estim_f, BSP_estim_images


    # ################################################### ALL ################################################### #
    if estimator_mode is 'ALL':
        ALL_estim_D        = [SEGb_estim_D     , SEG_estim_D       , LSQ_estim_D       , BSP_estim_D       ]
        ALL_estim_D_star   = [SEGb_estim_D_star, SEG_estim_D_star  , LSQ_estim_D_star  , BSP_estim_D_star  ]
        ALL_estim_f        = [SEGb_estim_f     , SEG_estim_f       , LSQ_estim_f       , BSP_estim_f       ]
        ALL_estim_images   = [SEGb_estim_images, SEG_estim_images  , LSQ_estim_images  , BSP_estim_images  ]
        return ALL_estim_D, ALL_estim_D_star, ALL_estim_f, ALL_estim_images

    # Should never get here
    return


####################################################################################################################
####################################################################################################################
############################################### error_estim_ivim_maps ##############################################
####################################################################################################################
####################################################################################################################
def error_estim_ivim_maps(dw_mri_images, b_val, known_D, known_D_star, known_f, estim_D, estim_D_star, estim_f,
                          error_type='perc'):
    """
    :param dw_mri_images:    Type: ndarray, shape [b_num,y_wid, x_wid]
                             A 3D matrix consisting of a series of slices of DW-MRI images taken at b_num
                             different b-values.
    :param b_val:            Type: ndarray, shape [b_num,]
                             Vector containing the B-values the DW-MRI image was taken at.
    :param known_D:          Type: ndarray, shape [y_wid, x_wid] [(mm^2)/s]
                             Simulated D map.
    :param known_D_star:     Type: ndarray, shape [y_wid, x_wid] [(mm^2)/s]
                             Simulated D* map.
    :param known_f:          Type: ndarray, shape [y_wid, x_wid] [a.u.]
                             Simulated f map.
    :param estim_D:          Type: ndarray, shape [y_wid, x_wid] [(mm^2)/s]
                             Estimated D map.
    :param estim_D_star:     Type: ndarray, shape [y_wid, x_wid] [(mm^2)/s]
                             Estimated  D* map.
    :param estim_f:          Type: ndarray, shape [y_wid, x_wid] [a.u.]
                             Estimated  f map.
    :param error_type:       Type: string. Possible values: ‘l1’, ‘l2’, ‘perc’
                             Type of error calculation.
    :return: <error_type>_D_error :      Type: float
                                         Calculated error between known (e.g. simulated) D map and estimated D map.
             <error_type>_D_star_error : Type: float
                                         Calculated error between known (e.g. simulated) D* map and estimated D* map.
             <error_type>_f_error:       Type: float
                                         Calculated error between known (e.g. simulated) f map and estimated f map.
             <error_type>_mean_error :   Type: float
                                         Mean of calculated IVIM parameters maps errors.
    """
    # ########################################### input errors checker ########################################### #
    input_valid = 'valid'
    if not np.all(np.greater_equal(dw_mri_images, 0)):
        print("dw_mri_image is invalid. All elements must be non-negative.")
        input_valid = 'error'
    if not np.all(np.greater_equal(b_val, 0)):
        print("b_val is invalid. All elements must be non-negative.")
        input_valid = 'error'
    if not np.all(np.greater_equal(known_D, 0)):
        print("IVIM parameter 'known_D' is invalid. Please enter a valid 'known_D'")
        print("All 'known_D' elements must be larger or equal to 0")
        input_valid = 'error'
    if not np.all(np.greater_equal(known_D_star, 0)):
        print("IVIM parameter 'known_D_star' is invalid. Please enter a valid 'known_D_star'")
        print("All 'known_D_star' elements must be larger or equal to 0")
        input_valid = 'error'
    if not np.all(np.greater_equal(known_f, 0)) and np.all(np.less_equal(known_f, 1)):
        print("IVIM parameter 'known_f' is invalid. Please enter a valid 'known_f'")
        print("All 'known_f' elements must be between 0 to 1")
        input_valid = 'error'
    if not np.all(np.greater_equal(estim_D, 0)):
        print("IVIM parameter 'estim_D' is invalid. Please enter a valid 'estim_D'")
        print("All 'estim_D' elements must be larger or equal to 0")
        input_valid = 'error'
    if not np.all(np.greater_equal(estim_D_star, 0)):
        print("IVIM parameter 'estim_D_star' is invalid. Please enter a valid 'estim_D_star'")
        print("All 'estim_D_star' elements must be larger or equal to 0")
        input_valid = 'error'
    if not np.all(np.greater_equal(estim_f, 0)) and np.all(np.less_equal(estim_f, 1)):
        print("IVIM parameter 'estim_f' is invalid. Please enter a valid 'estim_f'")
        print("All 'estim_f' elements must be between 0 to 1")
        input_valid = 'error'
    if error_type not in ['perc', 'l1', 'l2']:
        print("'error_type' is invalid. Please enter a valid 'error_type':")
        print("'perc' or 'l1' or 'l2'")
        input_valid = 'error'

    assert (input_valid is 'valid'), "Error in ivim_parameters_error function. View printed comments above."

    # ########################################################################################################### #
    B0_image = dw_mri_images[np.argmin(b_val)]
    x_wid = dw_mri_images.shape[2]
    y_wid = dw_mri_images.shape[1]
    b_val = np.array(b_val)

    # prep images to analysis
    B0_image_spread     = np.reshape(B0_image.copy(), (x_wid * y_wid))
    known_D_spread      = np.reshape(known_D.copy(), (x_wid * y_wid))
    known_D_star_spread = np.reshape(known_D_star.copy(), (x_wid * y_wid))
    known_f_spread      = np.reshape(known_f.copy(), (x_wid * y_wid))
    estim_D_spread      = np.reshape(estim_D.copy(), (x_wid * y_wid))
    estim_D_star_spread = np.reshape(estim_D_star.copy(), (x_wid * y_wid))
    estim_f_spread      = np.reshape(estim_f.copy(), (x_wid * y_wid))

    # =============== diff arrays =============== #
    D_diff      = np.absolute(known_D_spread - estim_D_spread)
    D_star_diff = np.absolute(known_D_star_spread - estim_D_star_spread)
    f_diff      = np.absolute(known_f_spread - estim_f_spread)

    # =============== calc error =============== #
    if error_type is 'l2':
        l2_D_error      = np.mean(D_diff ** 2)
        l2_D_star_error = np.mean(D_star_diff ** 2)
        l2_f_error      = np.mean(f_diff ** 2)
        l2_mean_error   = np.mean([l2_D_error, l2_D_star_error, l2_f_error])
        return l2_D_error, l2_D_star_error, l2_f_error, l2_mean_error

    if error_type is 'l1':
        l1_D_error      = np.median(D_diff)
        l1_D_star_error = np.median(D_star_diff)
        l1_f_error      = np.median(f_diff)
        l1_mean_error   = np.mean([l1_D_error, l1_D_star_error, l1_f_error])
        return l1_D_error, l1_D_star_error, l1_f_error, l1_mean_error

    if error_type is 'perc':
        # ~~~ get rel arrays ~~~ #
        rel_D       = np.copy(D_diff)
        rel_D_star  = np.copy(D_star_diff)
        rel_f       = np.copy(f_diff)
        rel_D[known_D_spread != 0]             = rel_D[known_D_spread != 0] / known_D_spread[known_D_spread != 0]
        rel_D_star[known_D_star_spread != 0]   = rel_D_star[known_D_star_spread != 0] / known_D_star_spread[known_D_star_spread != 0]
        rel_f[known_f_spread != 0]             = rel_f[known_f_spread != 0] / known_f_spread[known_f_spread != 0]
        # ~~~ calc perc ~~~ #
        perc_D_error      = 100 * np.mean(rel_D)
        perc_D_star_error = 100 * np.mean(rel_D_star)
        perc_f_error      = 100 * np.mean(rel_f)
        perc_mean_error   = np.mean([perc_D_error, perc_D_star_error, perc_f_error])
        return perc_D_error, perc_D_star_error, perc_f_error, perc_mean_error

    # Should never get here
    return


####################################################################################################################
####################################################################################################################
############################################### error_estim_dw_images ##############################################
####################################################################################################################
####################################################################################################################
def error_estim_dw_images(orig_dw_mri_images, rec_dw_mri_images, b_val, sim_flag, error_type='perc'):
    """
    :param orig_dw_mri_images: Type: ndarray, shape [b_num,y_wid, x_wid]
                               A 3D matrix consisting of a series of slices of DW-MRI images taken at
                               b_num different b-values.
    :param rec_dw_mri_images:  Type: ndarray, shape [b_num,y_wid, x_wid]
                               A 3D matrix consisting of a series of slices of reconstructed DW-MRI images
                               taken at b_num different b-values.
    :param b_val:              Type: ndarray, shape [b_num,]
                               Vector containing the B-values the DW-MRI image was taken at.
    :param sim_flag:           Type: boolean
                               If True, the input images are from simulation meaning every pixel comes from IVIM
                               bi-exponential model. Else (if False) the input images are real clinical DWMRI images.
    :param error_type:         Type: string. Possible values: ‘l1’, ‘l2’, ‘perc’
                               Type of error calculation.
    :return: <error_type>_error : Type: float
                                  Calculated error between known DW-MRI images and reconstructed DW-MRI
                                images obtained from IVIM estimation function. (Phantom_Simulation_DWMRI)
    """

    # ########################################### input errors checker ########################################### #
    input_valid = 'valid'
    if not np.all(np.greater_equal(orig_dw_mri_images, 0)):
        print("Orig dw_mri_image is invalid. All elements must be non-negative.")
        input_valid = 'error'
    if not np.all(np.greater_equal(rec_dw_mri_images, 0)):
        print("Rect dw_mri_image is invalid. All elements must be non-negative.")
        input_valid = 'error'
    if not np.all(np.greater_equal(b_val, 0)):
        print("b_val is invalid. All elements must be non-negative.")
        input_valid = 'error'
    if sim_flag not in [True, False]:
        print("sim_flag value is invalid.")
        print("sim_flag must be True or False.")
        input_valid = 'error'
    if error_type not in ['perc', 'l1', 'l2']:
        print("'error_type' is invalid. Please enter a valid 'error_type':")
        print("'perc' or 'l1' or 'l2'")
        input_valid = 'error'

    assert (input_valid is 'valid'), "Error in ivim_parameters_error function. View printed comments above."

    # ########################################################################################################### #
    B0_image = orig_dw_mri_images[np.argmin(b_val)]
    x_wid = orig_dw_mri_images.shape[2]
    y_wid = orig_dw_mri_images.shape[1]
    b_val = np.array(b_val)
    b_num = len(b_val)
    if sim_flag is True:
        B0_image_min_real_val = -1
    else:
        B0_image_min_real_val = 50
    # prep images to analysis
    B0_image_spread     = np.reshape(B0_image.copy(), (x_wid * y_wid))

    l2_error_list = []
    l1_error_list = []
    perc_error_list = []
    for i in range(b_num):
        tmp_orig = np.reshape(orig_dw_mri_images[i,:,:], (x_wid * y_wid))
        tmp_rec = np.reshape(rec_dw_mri_images[i,:,:], (x_wid * y_wid))
        tmp_orig [B0_image_spread < B0_image_min_real_val] = 0
        tmp_rec [B0_image_spread < B0_image_min_real_val] = 0
        tmp_diff = np.absolute(tmp_orig - tmp_rec)
        if error_type is 'l2':
            l2_error_list      = l2_error_list+[np.mean(tmp_diff ** 2)]
        if error_type is 'l1':
            l1_error_list      = l1_error_list+[np.median(tmp_diff)]
        if error_type is 'perc':
            # ~~~ get rel arrays ~~~ #
            rel  = np.copy(tmp_diff)
            rel[tmp_orig != 0] = rel[tmp_orig != 0] / tmp_orig[tmp_orig != 0]
            # ~~~ calc perc ~~~ #
            perc_error_list = perc_error_list+[100 * np.mean(rel)]


    # =============== calc error =============== #
    if error_type is 'l2':
        l2_error      = np.mean(l2_error_list)
        return l2_error

    if error_type is 'l1':
        l1_error      = np.median(l1_error_list)
        return l1_error

    if error_type is 'perc':
        perc_error = np.mean(perc_error_list)
        return perc_error

    # Should never get here
    return
