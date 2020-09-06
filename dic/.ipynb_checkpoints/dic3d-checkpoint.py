from . import dic
import glob
import numpy as np
import pickle
import time
import os
import cv2
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def save_mesh(mesh, subset_radius, save_dir, mesh_dir):
    mesh_info = {'mesh': mesh, 'subset_radius': subset_radius}
    time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime((time.time())))[2:]
    file_name = 'mesh_' + time_stamp + '.sav'
    pickle.dump(mesh_info, open(os.path.join(save_dir, mesh_dir, file_name), 'wb'))

def get_new_mesh(img_path):
    pmd = dic.PolygonMeshDrawer(img_path)
    ret = False
    while (not ret):
        print('Draw new mesh')
        pmd.set_polygon()
        mesh = pmd.get_mesh()
        ret = pmd.show_mesh()
    mesh = np.array(mesh, dtype='float32')
    subset_radius = pmd.get_subset_radius()
    return mesh, subset_radius

def load_mesh(mesh_name, save_dir, mesh_dir):
    with open(os.path.join(save_dir, mesh_dir, mesh_name), 'rb') as f:
        mesh_info = pickle.load(f)
    return mesh_info['mesh'], mesh_info['subset_radius']

def get_projection_mtx(mtx_name, save_dir, camera_mtx_dir):
    with open(os.path.join(save_dir, camera_mtx_dir, mtx_name), 'rb') as f:
        cam_mtx = pickle.load(f)
    Kl = cam_mtx['left_cam_mtx']
    Kr =cam_mtx['right_cam_mtx']
    R = cam_mtx['rotation_mtx']
    T = cam_mtx['translation_mtx']

    P1 = Kl@np.eye(3,4)
    P2 = Kr@np.concatenate([R,T], axis=1)
    return P1, P2

def get_find_points(result_graphL, result_graphR):
    find_points = np.logical_not(np.any(np.isnan(result_graphL.nodes[1]['POIs']), axis=1))
    for i in range(2, len(result_graphL.nodes)):
        points = np.logical_not(np.any(np.isnan(result_graphL.nodes[i]['POIs']), axis=1))
        find_points = np.logical_and(find_points, points)
    for i in range(1, len(result_graphR.nodes)):
        points = np.logical_not(np.any(np.isnan(result_graphR.nodes[i]['POIs']), axis=1))
        find_points = np.logical_and(find_points, points)
    return find_points

def reconstruct_3d(result_graphL, result_graphR, find_points, P1, P2):
    points3d=[]
    for i in range(len(result_graphL.nodes)):
        pointsL = result_graphL.nodes[i]['POIs'][find_points]
        pointsR = result_graphR.nodes[i+1]['POIs'][find_points]
        reconstruct_point = cv2.triangulatePoints(P1, P2, np.expand_dims(pointsL, 1), np.expand_dims(pointsR, 1))
        reconstruct_point /= reconstruct_point[3,:]
        points3d.append(reconstruct_point[:3].T)
    return points3d

def show_3d_plot(points3d):
    fig = plt.figure()
    ax = Axes3D(fig)
    for i, points in enumerate(points3d):
        X = points[:,0]
        Y = points[:,1]
        Z = points[:,2]
        ax.scatter(X,Y,Z, label=i)
    ax.legend()
    plt.show()

def save_disp_fig(result_graph, reader, points3d, find_points, result_path, suffix):
    idx2d=0
    idx3d=0
    if suffix == 1:
        idx2d += 1

    ref_points2d = result_graph.nodes[idx2d]['POIs'][find_points]
    ref_x = ref_points2d[:, 0]
    ref_y = ref_points2d[:, 1]
    ref_z = points3d[idx3d][:, 2]

    idx2d += 1
    idx3d += 1
    while(idx2d < len(result_graph.nodes)):
        cur_points2d = result_graph.nodes[idx2d]['POIs'][find_points]
        cur_x = cur_points2d[:, 0]
        cur_y = cur_points2d[:, 1]
        cur_z = points3d[idx3d][:, 2]

        u =  cur_x - ref_x
        v = cur_y - ref_y
        z_disp = cur_z - ref_z

        Z_disp = z_disp.copy()
        Z_disp_min, Z_disp_max = np.quantile(z_disp, [0.05, 0.95])
        Z_disp[Z_disp < Z_disp_min] = Z_disp_min
        Z_disp[Z_disp > Z_disp_max] = Z_disp_max

        fig = plt.figure()
        fig.set_size_inches(20, 20)
        plt.imshow(reader[suffix], cmap='gray')
        qq = plt.quiver(ref_x, ref_y, u, v, Z_disp, cmap=plt.cm.jet)
        plt.colorbar(qq, cmap=plt.cm.jet)
        plt.title('frame index {}'.format(idx3d))
        plt.savefig('{}/frame_{:02d}_{}.png'.format(result_path, idx3d, suffix),
                    pad_inches=0, bbox_inches='tight', dpi=200)
        idx2d += 1
        idx3d += 1


def run(args, save_dir, camera_mtx_dir, mesh_dir):
    left_img_path = args.left_img_path
    right_img_path = args.right_img_path
    suffix = args.suffix

    subpixel_method = args.subpixel_method
    ZNSSD_limit = args.ZNSSD_limit
    start_index = args.start_index
    end_index = args.end_index
    result_path = args.result_path

    pathsL = glob.glob(left_img_path+'/*.'+suffix)
    pathsR_ = glob.glob(right_img_path+'/*.'+suffix)
    assert len(pathsL) == len(pathsR_)
    print("Find {} image pairs".format(len(pathsL)))
    pathsR = pathsL[:1]+pathsR_

    print('Load Camera matrix : {}'.format(args.load_cam_mtx))
    P1, P2 = get_projection_mtx(args.load_cam_mtx, save_dir, camera_mtx_dir)

    if args.new_mesh:
        mesh, subset_radius = get_new_mesh(pathsL[0])
        save_mesh(mesh, subset_radius, save_dir, mesh_dir)
    else:
        print('Load mesh : {}'.format(args.load_mesh))
        mesh, subset_radius =load_mesh(args.load_mesh, save_dir, mesh_dir)

    print("number of poits : {}\nsubset radius : {}".format(len(mesh), subset_radius))

    readerL = dic.image_loader(pathsL, suffix=suffix)
    readerR = dic.image_loader(pathsR, suffix=suffix)

    output_frame_listL = np.arange(1, end_index + 1, dtype=np.int64)
    output_frame_listR = np.arange(1, end_index + 2, dtype=np.int64)


    t1 = time.time()
    print("============= Left 2D DIC =============")
    result_graphL, POI_graph_listL = dic.bisection_search(readerL, start_index, end_index,
                                                          mesh, output_frame_listL,
                                                          subset_level=subset_radius,
                                                          subpixel_method=subpixel_method,
                                                          ZNSSD_limit=ZNSSD_limit)
    print("============= Right 2D DIC =============")
    result_graphR, POI_graph_listR = dic.bisection_search(readerR, start_index, end_index+1,
                                                          mesh, output_frame_listR,
                                                          subset_level=subset_radius,
                                                          subpixel_method=subpixel_method,
                                                          ZNSSD_limit=ZNSSD_limit)
    t2 = time.time()
    print('calculation time {:.2f}s'.format(t2 - t1))

    find_points = get_find_points(result_graphL, result_graphR)
    print('number of find points : {}'.format(len(find_points)))

    points3d = reconstruct_3d(result_graphL, result_graphR, find_points, P1, P2)
    print('Show reconstructed 3d points')
    show_3d_plot(points3d)

    print('Save displacemet figure')
    save_disp_fig(result_graphL, readerL, points3d, find_points, result_path, suffix=0) # suffix=0 --> left
    save_disp_fig(result_graphR, readerR, points3d, find_points, result_path, suffix=1) # suffix=1 --> right

    print('Save result.csv')
    df = {'meshL_x': result_graphL.nodes[0]['POIs'][find_points][:, 0],
          'meshL_y': result_graphL.nodes[0]['POIs'][find_points][:, 1],
          'meshR_x': result_graphR.nodes[1]['POIs'][find_points][:, 0],
          'meshR_y': result_graphR.nodes[1]['POIs'][find_points][:, 1],
          'ref_x': points3d[0][:, 0],
          'ref_y': points3d[0][:, 1],
          'ref_z': points3d[0][:, 2]}

    for i in range(1, len(points3d)):
        disp_x = points3d[i][:, 0] - df['ref_x']
        disp_y = points3d[i][:, 1] - df['ref_y']
        disp_z = points3d[i][:, 2] - df['ref_z']
        df['disp{:02d}_x'.format(i)] = disp_x
        df['disp{:02d}_y'.format(i)] = disp_y
        df['disp{:02d}_z'.format(i)] = disp_z
        df['disp{:02d}'.format(i)] = np.sqrt(disp_x ** 2 + disp_y ** 2 + disp_z ** 2)

    df = pd.DataFrame(df)
    df.to_csv(os.path.join(result_path, 'result.csv'))
    print('Done')






