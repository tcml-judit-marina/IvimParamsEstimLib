Last Update: August 14, 2020  
Authors: Judit Ben Ami & Marina Khizgilov  
Created during 2019-2020 as part of a final project towards BSc in biomedical engineering.  
TCML lab at the Department of Biomedical Engineering, Technion - IIT, Haifa, Israel  

Under the supervision of Dr. Moti Frieman  
Special thanks for Elad Rotman for phantom_simulation function.  

# IvimParamsEstimLib 

IvimParamsEstimLib is a multi-purpose Python library for IVIM parameters estimation.  
The model used for IVIM–based biexponential analysis is:  
s_i=s_0 (f⋅e^├ -b_i (D^*+D) +(1-f)e^(-b_i D) )  
Where D is the diffusion coefficient, which reflects tissue diffusivity; D\* is the pseudo-diffusion coefficient, which reflects microcapillary perfusion; and f is the perfusion fraction.  
This library includes:  
 - 4 IVIM parameters estimation algorithms: SEGb, SEG, LSQ, BSP.  
 - Phantom creating function.  
 - Error calculation functions.  
 <a/>
In addition, for efficiency and higher performance, the library uses vectorize programming, multiprocessing, assertions and exceptions.  

### IvimParamsEstimLib.dwmri_images_to_ndarray

**dwmri_images_to_ndarray(file_path, b_value, slice_num=0, plot_flag=False, save_plot_flag=False, save_plot_prefix=’001’)**

Converts input DW-MRI images to a ndarray.  
###### Parameters:  
**file_path : List of PathLike**  
Paths of the DW-MRI images to convert.  
**b-values : ndarray**  
Vector containing the B-values the DW-MRI image was taken at.  
**slice_num : integer, optional (0 by default)**  
For 3D Images, choose a specific 2D slice.  
**plot_flag : boolean, optional (False by default)**  
If True, plots the output figure.  
**save_plot_flag : boolean, optional (False by default)**  
If True, saves the output figure as a PNG file to current working 	directory.  
**save_plot_prefix: str, optional (‘001’ by default)**  
Prefix of the name of the file the figure will be saved as.  

# Returns:
** slice: ndarray, shape [b_num, y_wid, x_wid]**  
Conversion result matrix.  


### IvimParamsEstimLib.ivim_params_estim   

**ivim_params_estim(dw_mri_images, b, estimator_mode='ALL', image_type_mode='norm', b_tresh=200, plot_flag=True, save_plot_flag=False, save_plot_prefix="001", multi_segs=1, env_size=5, dbg_features=False)**

Evaluates the IVIM parameters, D, D* and f. Reconstructs the DW-MRI imaged using the estimated IVIM parameters.

###### Parameters:  
**dw_mri_images : ndarray, shape [b_num,y_wid, x_wid]**  
A 3D matrix consisting of a series of slices of DW-MRI images taken at b_num different b-values.  
**b : ndarray**  
Vector containing the B-values the DW-MRI image was taken at.  
**estimator_mode ‘SEGb’, ‘SEG’, ‘LSQ’ or ‘BSP’ ,’ALL’ optional (‘ALL’ by default)**  
Type of IVIM parameters estimation algorithm. Choose ‘ALL’ to get estimation results using all the modes: ‘SEGb’, ‘SEG’, ‘LSQ’ or ‘BSP’.  
**image_type_mode : ‘norm’ or ‘absolute’, optional (‘absolute’’ by default)**  
DW-MRI image mode. ‘norm’ if image is normalized, ‘absolute’ if Image is not normalized.  
**b_tresh : integer, optional (200 by default)**  
b-value threshold for SEG/SEGb estimations.  
**plot_flag : boolean, optional (False by default)**  
If True, plots the output figures - maps of IVIM parameters and  the reconstructed DW-MRI images.  
**save_plot_flag : boolean, optional (False by default)**  
If True, saves output figures as a PNG files to current working directory.  
**save_plot_prefix: str, optional (‘001’ by default)**  
Prefix of the name of the files the figures were saved as.  
**multi_segs: integer, optional (8 by default)**  
For multi-processing implementation. Maximum number of multi-processes.  
**env_size: integer, optional (5 by default)**  
For ‘BSP’ mode, number of elements in each direction around a pixel. 

# Returns :
**<estimator_mode>_estim_D : ndarray (or list for ‘ALL’ estimator)**  
Estimated D map, using <estimator_mode> algorithm.  
**<estimator_mode>_estim_D_star : ndarray (or list for ‘ALL’ estimator)**  
Estimated D* map, using <estimator_mode> algorithm.  
**<estimator_mode>_estim_f : ndarray (or list for ‘ALL’ estimator)**  
Estimated f map, using <estimator_mode> algorithm.  
**<estimator_mode>_ estim_images : ndarray (or list for ‘ALL’ estimator)**  
Reconstructed images.  

Notes:  
	For shorter running time, more than 1 multi-process is favoured.
	For ‘ALL’ estimator mode, each return is a list consists the results of:  ‘SEGb’, ‘SEG’, ‘LSQ’, ‘BSP’ in that particular order.
  
  
### IvimParamsEstimLib.Phantom_Simulation_DWMRI

**phantom_simulation(b_val, D_val, D_star_val, Fp_val, B0_val, x_wid=64, y_wid=64, rads=None, noise_type='NaN', SNR=0.1, plot_flag=False, save_plot_flag=False, save_plot_prefix=’001’)**  

Creates IVIM parameters maps and simulates DW-MRI images. The phantoms created consists of N concentric circles.

###### Parameters:  
**b-values : ndarray**  
Vector containing the b-values for the simulated image.  
**D_val : ndarray , shape [N+1,]**  
Vector containing the D values of the simulated image.  
**D_star_val : ndarray, shape [N+1,]**  
Vector containing the D* values of the simulated image.  
**Fp_val : ndarray, shape [N+1,]**  
Vector containing the f values of the simulated image.  
**B0_val : ndarray, shape [N+1,]**  
Vector containing the B0 values of the simulated image.  
**x_wid : integer, optional (64 by default)**  
Number of pixels at x-axis of simulated image.  
**y_wid : integer, optional (64 by default)**
Number of pixels at y-axis of simulated image.  
**rads : ndarray, shape [N,], optional ([10,20,30] by default)**  
Vector containing the radii of the circles of simulate image. The radii are in ascending order of length, where the value of the first index of the vector is the radius of the inner circle, and value of the last index of the vector is the radius of the external circle.  
**noise_type : ‘NaN’, ‘gaussian’, ‘rayleigh’, ‘rice’ or ‘non_centralized_chi2’, optional (‘NaN’ by default)**  
Type of noise to be added to the simulated DW-MRI images.  
**SNR : float, optional (0.1 by default)**  
SNR value for gaussian noise.  
**plot_flags : boolean, optional (False by default)**  
If True, plots the output figures – IVIM parameters maps and simulated DW-MRI images.  
**save_plot_flag : boolean, optional (False by default)**  
If True, saves the output figure as a PNG file to current working directory.  
**save_plot_prefix: str, optional (‘001’ by default)**
Name of the file the figure was saved as.  

###### Returns:  
**dwi_images : ndarray, shape [b_num,y_wid, x_wid]**  
Simulated DW-MRI images.  
**B0_phantom : ndarray, shape [y_wid, x_wid]**  
Initial simulated DW-MRI image taken at b=0.  
**D_phantom : ndarray, shape [y_wid, x_wid]**  
Simulated D map.  
**D_star_phantom: ndarray, shape [y_wid, x_wid]**  
Simulated D* map.  
**Fp_phantom : ndarray, shape [y_wid, x_wid]**  
Simulated f map.  

Notes: 
As mentioned, D_val, D_star_val,  Fp_val, B0_val vectors are the size of N+1. The elements of each vector are ordered according to circles locations: The first element of each vector represents the inner circle and each element that comes afterwards represents the next circle. Last element of the vectors represents the background of the phantom.

### IvimParamsEstimLib.error_estim_ivim_maps

**error_estim_ivim_maps(dw_mri_images, b_val, known_D, known_D_star, known_f, estim_D, estim_D_star, estim_f, error_type='perc')**

Computes the error between simulated IVIM parameters maps and estimated IVIM parameters maps. 

###### Parameters:  
**dw_mri_images: ndarray, shape [b_num,y_wid, x_wid]**  
A 3D matrix consisting of a series of slices of DW-MRI images taken at b_num different b-values.  
**b-val : ndarray**  
Vector containing the B-values the DW-MRI image was taken at.  
**known_D : ndarray, shape [y_wid, x_wid]**  
Simulated D map.  
**known_D_star : ndarray, shape [y_wid, x_wid]**  
Simulated D* map.  
**known_f : ndarray, shape [y_wid, x_wid]**  
Simulated f map.  
**estim_D: ndarray, shape [y_wid, x_wid]**  
Estimated D map.  
**estim_D_star: ndarray, shape [y_wid, x_wid]**  
Estimated D* map.  
**estim_f: ndarray, shape [y_wid, x_wid]**  
Estimated f map.  
**error_type: ‘l1’, ‘l2’ or ‘perc’ (‘perc’ by default)**
Type of error calculation.  
###### Returns:  	
**D_error : float**  
Calculated error between known (e.g. simulated) D map and estimated D map.  
**D_star_error : float**  
Calculated error between known (e.g. simulated) D* map and estimated D* map.  
**f_error: float**  
Calculated error between known (e.g. simulated) f map and estimated f map.  
**mean_error : float**  
 Mean of calculated IVIM parameters maps errors.  
 
Notes:

### IvimParamsEstimLib.error_estim_dw_images

**error_estim_dw_images(orig_dw_mri_images, rec_dw_mri_images, b_val, error_type='perc')**

Computes the error between original DW-MRI images and  reconstructed DW-MRI images obtained from IVIM estimation function.

###### Parameters:  
**orig_dw_mri_images : ndarray, shape [b_num,y_wid, x_wid]**  
A 3D matrix consisting of a series of slices of DW-MRI images taken at b_num different b-values.  
**rec_dw_mri_images : ndarray, shape [b_num,y_wid, x_wid]**  
A 3D matrix consisting of a series of slices of reconstructed DW-MRI images taken at b_num different b-values.  
**b-val : ndarray**  
Vector containing the B-values the DW-MRI image was taken at.  
**error_type: ‘l1’, ‘l2’ or ‘perc’ (‘perc’ by default)**  
Type of error calculation.  
###### Returns:  
**< error_type >_error : float**  
Calculated error between known DW-MRI images and reconstructed DW-MRI images obtained from IVIM estimation function. (Phantom_Simulation_DWMRI)  




