import numpy as np
import cv2
import os
import pickle
import time
import glob

def by_cam(args, save_dir, camera_mtx_dir):

    n_cap = args.n_cap
    capture_img_path = args.capture_img_path
    left_cam_num = args.left_cam_num
    right_cam_num = args.right_cam_num

    CamL= cv2.VideoCapture(left_cam_num)
    CamR= cv2.VideoCapture(right_cam_num)

    count = 0
    left_imgs=[]
    right_imgs=[]

    window_start = (640, 60)

    cv2.namedWindow('left_img')
    cv2.moveWindow('left_img', *window_start)

    cv2.namedWindow('right_img')
    cv2.moveWindow('right_img', int(window_start[0] + CamL.get(3)), window_start[1])

    while True:
        retL, frameL= CamL.read()
        retR, frameR= CamR.read()

        if not ((retR == True) and (retL == True)):
            continue

        cv2.imshow('left_img',frameL)
        cv2.imshow('right_img',frameR)


        if cv2.waitKey(1) & 0xFF == ord('s'):   # Push "s" to save the images

            left_imgs.append(frameL)
            right_imgs.append(frameR)
            print('[{:2d}/{}] capture right and left camera images'.format(count+1, n_cap))

            if capture_img_path is not None:
                left_dir = os.path.join(capture_img_path, 'capture_left')
                right_dir = os.path.join(capture_img_path, 'capture_right')

                if not os.path.exists(left_dir):
                    os.mkdir(left_dir)

                if not os.path.exists(right_dir):
                    os.mkdir(right_dir)

                cv2.imwrite(os.path.join(right_dir,'chessboard{:02d}.png'.format(count)), frameR)
                cv2.imwrite(os.path.join(left_dir,'chessboard{:02d}.png'.format(count)), frameL)
            count = count+1

        if (cv2.waitKey(1) & 0xFF == ord('q')):   # Push 'q' to exit this Programm
            return

        if (count >= n_cap):
            break

    CamR.release()
    CamL.release()
    cv2.destroyAllWindows()

    by_imgs(args, save_dir, camera_mtx_dir, left_imgs, right_imgs)

def by_imgs(args , save_dir, camera_mtx_dir, left_imgs=[], right_imgs=[]):
    if (left_imgs==[]) and (right_imgs==[]):
        left_img_paths = glob.glob(args.left_img_path+'/*.'+args.suffix)
        right_img_paths = glob.glob(args.right_img_path+'/*.'+args.suffix)

        assert len(left_img_paths) == len(right_img_paths)

        for left_path, right_path in zip(left_img_paths, right_img_paths):
            left_imgs.append(cv2.imread(left_path, 1))
            right_imgs.append(cv2.imread(right_path, 1))

    chess_rows = args.chess_rows
    chess_cols = args.chess_cols
    result_img_path = args.result_img_path

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    objp = np.zeros((chess_rows * chess_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:chess_rows, 0:chess_cols].T.reshape(-1, 2)

    # Arrays to store object points and image points from all images
    objpoints = []  # 3d points in real world space
    imgpointsR = []  # 2d points in image plane
    imgpointsL = []

    print('Starting calibration for the 2 cameras... ')

    count =0
    n_imgs = len(left_imgs)
    find_chessboardL=[]
    find_chessboardR=[]
    for imgL, imgR in zip(left_imgs, right_imgs):

        gray_imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        gray_imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        retL, cornersL = cv2.findChessboardCorners(gray_imgL, (chess_rows, chess_cols), None)
        retR, cornersR = cv2.findChessboardCorners(gray_imgR, (chess_rows, chess_cols), None)

        if retL and retR:
            count +=1
            print("{}/{} chessboard find".format(count, n_imgs))
            find_chessboardL.append([gray_imgL, imgL, cornersL])
            find_chessboardR.append([gray_imgR, imgR, cornersR])

    image_shape= right_imgs[0].shape[:2][::-1]
    disp_size_min = 600
    w, h = image_shape
    if (h > w):
        disp_h = int(h / w * disp_size_min)
        disp_w = int(disp_size_min)
    else:
        disp_w = int(w / h * disp_size_min)
        disp_h = int(disp_size_min)
    disp_shape = (disp_w, disp_h)

    left_window_start = (640, 60)
    right_window_start = (int(left_window_start[0] + disp_shape[0]), left_window_start[1])

    cv2.namedWindow("Left detect corner", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Left detect corner", *disp_shape)
    cv2.moveWindow("Left detect corner", *left_window_start)

    cv2.namedWindow("Right detect corner", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Right detect corner", *disp_shape)
    cv2.moveWindow("Right detect corner", *right_window_start)

    count=0
    for (gray_imgL, imgL, cornersL), (gray_imgR, imgR, cornersR) in zip(find_chessboardL, find_chessboardR):
        cornL = cv2.cornerSubPix(gray_imgL, cornersL, (11, 11), (-1, -1), criteria)
        cornR = cv2.cornerSubPix(gray_imgR, cornersR, (11, 11), (-1, -1), criteria)
        draw_cornL = cv2.drawChessboardCorners(imgL, (chess_rows, chess_rows), cornL, retL)
        draw_cornR = cv2.drawChessboardCorners(imgR, (chess_rows, chess_rows), cornR, retR)

        cv2.imshow("Left detect corner", draw_cornL)
        cv2.imshow("Right detect corner", draw_cornR)

        if cv2.waitKey(0)& 0xFF == ord('s'):
            objpoints.append(objp)
            imgpointsL.append(cornL)
            imgpointsR.append(cornR)

            count +=1
            print("Add corner", count)

        elif cv2.waitKey(0)& 0xFF == ord('q'):   # Push 'q' to exit this Programm
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    if count==0:
        print("Can't find any chessboard")
        print("Finish")
        return


    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                            imgpointsL,
                                                            image_shape, None, None)

    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                            imgpointsR,
                                                            image_shape, None, None)

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_ASPECT_RATIO
    # flags |= cv2.CALIB_ZERO_TANGENT_DIST
    # flags |= cv2.CALIB_RATIONAL_MODEL
    # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_K3
    # flags |= cv2.CALIB_FIX_K4
    # flags |= cv2.CALIB_FIX_K5

    retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                               imgpointsL,
                                                               imgpointsR,
                                                               mtxL,
                                                               distL,
                                                               mtxR,
                                                               distR,
                                                               image_shape,
                                                               criteria_stereo,
                                                               flags = flags)

    total_mtx = {'error' : retS,
                 'left_cam_mtx' : MLS,
                 'left_cam_dist' : dLS,
                 'right_cam_mtx' : MRS,
                 'right_cam_dist' : dRS,
                 'rotation_mtx' : R,
                 'translation_mtx': T,
                 'essential_mtx': E,
                 'fundamental_mtx': F}

    for key, value in total_mtx.items():
        print('\n'+key)
        print(value)

    time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime((time.time())))[2:]
    file_name = 'mtx_' + time_stamp + '_{:.2f}_'.format(retS)+'.sav'
    pickle.dump(total_mtx, open(os.path.join(save_dir, camera_mtx_dir, file_name), 'wb'))


    if result_img_path is not None:
        RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS, image_shape, R, T)

        xmapL, ymapL= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                                      image_shape , cv2.CV_16SC2)  # cv2.CV_16SC2 this format enables us the programme to work faster
        xmapR, ymapR = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                                       image_shape, cv2.CV_16SC2)

        left_dir = os.path.join(result_img_path, 'rectified_left')
        right_dir = os.path.join(result_img_path, 'rectified_right')

        if not os.path.exists(left_dir):
            os.mkdir(left_dir)

        if not os.path.exists(right_dir):
            os.mkdir(right_dir)

        for i, (left_img, right_img) in enumerate(zip(left_imgs, right_imgs)):
            left_img_rectified = cv2.remap(left_img, xmapL, ymapL, cv2.INTER_LINEAR)
            right_img_rectified = cv2.remap(right_img, xmapR, ymapR, cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(left_dir, 'rectified{:02d}.png'.format(i)), left_img_rectified)
            cv2.imwrite(os.path.join(right_dir, 'rectified{:02d}.png'.format(i)), right_img_rectified)
