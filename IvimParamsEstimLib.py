"""
@author: Marina Khizgilov, Judit Ben Ami

@description: IvimParamsEstimLib run examples

"""

import IvimParamsEstimLib as our_lib
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # TODO to user: to run this code put your images paths on line 105 !!!
    # ################################################### Simulation ###################################################
    # =============== Sim =============== #
    # IVIM parameters initialization for phantom simulation
    b_val = np.array([0, 20, 40, 100, 250, 350, 500, 800])
    D_val = [0.0020, 0.0015, 0.0010, 0.0005]
    D_star_val = [0.02, 0.04, 0.06, 0.08]
    Fp_val = [0.1, 0.15, 0.3, 0.45]
    # B0_val = [1500, 1300, 1600, 1800]
    B0_val = [500, 500, 1000, 6000]
    # call sim func
    # np.random.seed(42)
    phantoms_images, phantom_B0, phantom_D, phantom_D_star, phantom_f = \
        our_lib.phantom_simulation(b_val=b_val, D_val=D_val, D_star_val=D_star_val, Fp_val=Fp_val, B0_val=B0_val,
                                   noise_type='gaussian', SNR=0.001, plot_flag=True, save_plot_flag=True,
                                   save_plot_prefix="mj") # noise_type: gaussian / NaN
                                                          # SNR: Big SNR is a lot pf noise
                                                          # SNR: 0.001

    # =============== estim =============== #
    estim_D_list, estim_D_star_list, estim_f_list, estim_images_list = \
        our_lib.ivim_params_estim(dw_mri_images=phantoms_images, b=b_val, sim_flag=True, estimator_mode='ALL',
                                  image_type_mode='absolute', b_tresh=200, plot_flag=True, save_plot_flag=True,
                                  save_plot_prefix="mj_phantom", multi_segs=8, env_size=5, dbg_features=True)

    # =============== error estim =============== #
    # based on orig IVIM maps
    error_D_SEGb, error_D_star_SEGb, error_f_SEGb, error_avg_SEGb = \
        our_lib.error_estim_ivim_maps(dw_mri_images=phantoms_images, b_val=b_val,
                                      known_D=phantom_D, known_D_star=phantom_D_star, known_f=phantom_f,
                                      estim_D=estim_D_list[0], estim_D_star=estim_D_star_list[0],
                                      estim_f=estim_f_list[0], error_type='perc')
    error_D_SEG, error_D_star_SEG, error_f_SEG, error_avg_SEG = \
        our_lib.error_estim_ivim_maps(dw_mri_images=phantoms_images, b_val=b_val,
                                      known_D=phantom_D, known_D_star=phantom_D_star, known_f=phantom_f,
                                      estim_D=estim_D_list[1], estim_D_star=estim_D_star_list[1],
                                      estim_f=estim_f_list[1], error_type='perc')
    error_D_LSQ, error_D_star_LSQ, error_f_LSQ, error_avg_LSQ = \
        our_lib.error_estim_ivim_maps(dw_mri_images=phantoms_images, b_val=b_val,
                                      known_D=phantom_D, known_D_star=phantom_D_star, known_f=phantom_f,
                                      estim_D=estim_D_list[2], estim_D_star=estim_D_star_list[2],
                                      estim_f=estim_f_list[2], error_type='perc')
    error_D_BSP, error_D_star_BSP, error_f_BSP, error_avg_BSP = \
        our_lib.error_estim_ivim_maps(dw_mri_images=phantoms_images, b_val=b_val,
                                      known_D=phantom_D, known_D_star=phantom_D_star, known_f=phantom_f,
                                      estim_D=estim_D_list[3], estim_D_star=estim_D_star_list[3],
                                      estim_f=estim_f_list[3], error_type='perc')
    # based on rec images
    sim_images_error_SEGb = our_lib.error_estim_dw_images(orig_dw_mri_images=phantoms_images,
                                                          rec_dw_mri_images=estim_images_list[0],
                                                          b_val=b_val, sim_flag=True, error_type='perc')
    sim_images_error_SEG = our_lib.error_estim_dw_images(orig_dw_mri_images=phantoms_images,
                                                         rec_dw_mri_images=estim_images_list[1],
                                                         b_val=b_val, sim_flag=True, error_type='perc')
    sim_images_error_LSQ = our_lib.error_estim_dw_images(orig_dw_mri_images=phantoms_images,
                                                         rec_dw_mri_images=estim_images_list[2],
                                                         b_val=b_val, sim_flag=True, error_type='perc')
    sim_images_error_BSP = our_lib.error_estim_dw_images(orig_dw_mri_images=phantoms_images,
                                                         rec_dw_mri_images=estim_images_list[3],
                                                         b_val=b_val, sim_flag=True, error_type='perc')

    # =============== plot one pixel error estim =============== #
    plt.figure()
    j = 40
    i = 40
    plt.plot(b_val, np.log(phantoms_images[:, i, j, ]/phantoms_images[0, i, j])     , 'k.' , label="Orig")
    plt.plot(b_val, np.log(estim_images_list[0][:, i, j, ]/phantoms_images[0, i, j]), 'b--' , label="SEGb")
    plt.plot(b_val, np.log(estim_images_list[1][:, i, j, ]/phantoms_images[0, i, j]), 'g--' , label="SEG")
    plt.plot(b_val, np.log(estim_images_list[2][:, i, j, ]/phantoms_images[0, i, j]), 'r--' , label="LSQ")
    plt.plot(b_val, np.log(estim_images_list[3][:, i, j, ]/phantoms_images[0, i, j]), 'c--' , label="BSP")
    plt.xlabel("b values [sec/(mm^2)]")
    plt.ylabel("DW-MRI signal [a.u.]")
    plt.title("DW-MRI simulated signal vs b values")
    plt.legend(loc="upper right")
    plt.savefig('phantom_signal_vs_b.png', transparent=True)
    plt.show(block=False)

    # # =============== ivim maps - mean and std ===============
    # # ERRORS - LSQ vs BSP
    print(f"Sim images: D_star_ref: {phantom_D_star.mean()}")
    print(f"Sim images: D_star_hat_lsq (mu,sigma): {estim_D_star_list[2].mean()}, {estim_D_star_list[2].std()}")
    print(f"Sim images: D_star_hat_bsp (mu,sigma): {estim_D_star_list[3].mean()}, {estim_D_star_list[3].std()}")

    print(f"Sim images: f_ref: {phantom_f.mean()}")
    print(f"Sim images: f_hat_lsq (mu,sigma): {estim_f_list[2].mean()}, {estim_f_list[2].std()}")
    print(f"Sim images: f_hat_bsp (mu,sigma): {estim_f_list[3].mean()}, {estim_f_list[3].std()}")



    # ############################################### Real DW-MRI images ###############################################
    # =============== images to ndarray =============== #
    # TODO TO USER: to run this put your images paths here!!!
    # real_images_paths =     ["image_full_path_bval_0.vtk"   ,
    #                          "image_full_path_bval_50.vtk"  ,
    #                          "image_full_path_bval_100.vtk" ,
    #                          "image_full_path_bval_200.vtk" ,
    #                          "image_full_path_bval_400.vtk" ,
    #                          "image_full_path_bval_600.vtk" ,
    #                          "image_full_path_bval_800.vtk"  ]
    real_images_paths =     [] # TODO TO USER: delete this line

    real_b_values = np.array([0, 50, 100, 200, 400, 600, 800])
    real_images = our_lib.dwmri_images_to_ndarray(file_path=real_images_paths, b_value=real_b_values,
                                                  slice_num=23, plot_flag=True, save_plot_flag=True,
                                                  save_plot_prefix="case 001 plot")

    # =============== estim =============== #
    real_estim_D_list, real_estim_D_star_list, real_estim_f_list, real_estim_images_list = \
        our_lib.ivim_params_estim(dw_mri_images=real_images, b=real_b_values, sim_flag=False, estimator_mode='ALL',
                                  image_type_mode='absolute', b_tresh=200, plot_flag=True, save_plot_flag=True,
                                  save_plot_prefix="case001", multi_segs=8, env_size=10, dbg_features=True)

    # =============== error estim =============== #
    real_images_error_SEGb = our_lib.error_estim_dw_images(orig_dw_mri_images=real_images,
                                                           rec_dw_mri_images=real_estim_images_list[0],
                                                           b_val=real_b_values, sim_flag=False, error_type='perc')
    real_images_error_SEG = our_lib.error_estim_dw_images(orig_dw_mri_images=real_images,
                                                          rec_dw_mri_images=real_estim_images_list[1],
                                                          b_val=real_b_values, sim_flag=False, error_type='perc')
    real_images_error_LSQ = our_lib.error_estim_dw_images(orig_dw_mri_images=real_images,
                                                          rec_dw_mri_images=real_estim_images_list[2],
                                                          b_val=real_b_values, sim_flag=False, error_type='perc')
    real_images_error_BSP = our_lib.error_estim_dw_images(orig_dw_mri_images=real_images,
                                                          rec_dw_mri_images=real_estim_images_list[3],
                                                          b_val=real_b_values, sim_flag=False, error_type='perc')

    # =============== plot one pixel error estim =============== #
    plt.figure()
    j = 120
    i = 80
    plt.plot(real_b_values, np.log(real_images[:, i, j, ]/real_images[0, i, j, ])          , 'k.' , label="Orig")
    plt.plot(real_b_values, np.log(real_estim_images_list[0][:, i, j, ]/real_images[0, i, j, ]), 'b--', label="SEGb")
    plt.plot(real_b_values, np.log(real_estim_images_list[1][:, i, j, ]/real_images[0, i, j, ]), 'g--', label="SEG")
    plt.plot(real_b_values, np.log(real_estim_images_list[2][:, i, j, ]/real_images[0, i, j, ]), 'r--', label="LSQ")
    plt.plot(real_b_values, np.log(real_estim_images_list[3][:, i, j, ]/real_images[0, i, j, ]), 'c--', label="BSP")
    plt.xlabel("b values [sec/(mm^2)]")
    plt.ylabel("DW-MRI signal [a.u.]")
    plt.title("DW-MRI signal vs b values")
    plt.legend(loc="upper right")
    plt.savefig('real_signal_vs_b.png', transparent=True)
    plt.show(block=False)

    # # =============== ivim maps - mean and std ===============
    # prints - lsq vs bsp
    print(f"Real images: D_hat_lsq (mu,sigma): {real_estim_D_list[2].mean()}, {real_estim_D_list[2].std()}")
    print(f"Real images: D_hat_bsp (mu,sigma): {real_estim_D_list[3].mean()}, {real_estim_D_list[3].std()}")

    print(f"Real images: D_star_hat_lsq (mu,sigma): {real_estim_D_star_list[2].mean()}, {real_estim_D_star_list[2].std()}")
    print(f"Real images: D_star_hat_bsp (mu,sigma): {real_estim_D_star_list[3].mean()}, {real_estim_D_star_list[3].std()}")

    print(f"Real images: f_hat_lsq (mu,sigma): {real_estim_f_list[2].mean()}, {real_estim_f_list[2].std()}")
    print(f"Real images: f_hat_bsp (mu,sigma): {real_estim_f_list[3].mean()}, {real_estim_f_list[3].std()}")


    mj = 3
