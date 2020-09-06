import os
import pickle
import numpy as np
import cv2


def run(args, save_dir, camera_mtx_dir):

    left_cam_num = args.left_cam_num
    right_cam_num = args.right_cam_num

    CamL= cv2.VideoCapture(left_cam_num)
    CamR= cv2.VideoCapture(right_cam_num)

    cam_mtx_path = os.path.join(save_dir, camera_mtx_dir, args.load_cam_mtx)
    with open(cam_mtx_path, 'rb') as f:
        total_mtx = pickle.load(f)

    MLS = total_mtx['left_cam_mtx']
    dLS = total_mtx['left_cam_dist']
    MRS = total_mtx['right_cam_mtx']
    dRS = total_mtx['right_cam_dist']
    R = total_mtx['rotation_mtx']
    T = total_mtx['translation_mtx']

    img_shape = (int(CamL.get(3)), int(CamL.get(4)))
    kernel = np.ones((3, 3), np.uint8)

    RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS, img_shape, R, T)

    Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                               img_shape,
                                               cv2.CV_16SC2)
    Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                               img_shape, cv2.CV_16SC2)

    window_size = 3
    min_disp = 2
    num_disp = 130 - min_disp
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=window_size,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32,
                                   disp12MaxDiff=5,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2)

    # Used for the filtered image
    stereoR = cv2.ximgproc.createRightMatcher(stereo)  # Create another stereo for right this time

    # WLS FILTER Parameters
    lmbda = 80000
    sigma = 1.8

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    window_start = (640, 60)

    cv2.namedWindow('left_img')
    cv2.moveWindow('left_img', *window_start)

    cv2.namedWindow('right_img')
    cv2.moveWindow('right_img', int(window_start[0] + CamL.get(3)), window_start[1])

    while True:
        retR, frameR = CamR.read()
        retL, frameL = CamL.read()

        # Rectify the images on rotation and alignement
        Left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT,
                              0)  # Rectify the image using the kalibration parameters founds during the initialisation
        Right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4,
                               cv2.BORDER_CONSTANT, 0)

        grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
        grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)

        disp = stereo.compute(grayL, grayR)
        dispL = disp
        dispR = stereoR.compute(grayR, grayL)
        dispL = np.int16(dispL)
        dispR = np.int16(dispR)

        # Using the WLS filter
        filteredImgL = wls_filter.filter(dispL, grayL, None, dispR)
        filteredImgL = cv2.normalize(src=filteredImgL, dst=filteredImgL, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        filteredImgL = np.uint8(filteredImgL)

        filteredImgR = wls_filter.filter(dispR, grayR, None, dispL)
        filteredImgR = cv2.normalize(src=filteredImgR, dst=filteredImgR, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        filteredImgR = np.uint8(filteredImgR)

        # Colors map
        filt_ColorL = cv2.applyColorMap(filteredImgL, cv2.COLORMAP_OCEAN)
        filt_ColorR = cv2.applyColorMap(filteredImgR, cv2.COLORMAP_OCEAN)

        # Show the result for the Depth_image
        cv2.imshow('left_img', filt_ColorL)
        cv2.imshow('right_img', filt_ColorR)

        # End the Programme
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    CamR.release()
    CamL.release()
    cv2.destroyAllWindows()
