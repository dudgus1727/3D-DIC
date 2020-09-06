import os
import sys
import json
from gooey import Gooey, GooeyParser
from cam import calibration, disparity_vis
from dic import dic3d

save_dir = './save_config'
camera_mtx_dir = 'camera_mtx'
mesh_dir = 'mesh'
suffix_list=['jpg','png','tif','tiff']
subpixel_method_list =['fagn','icgn']

def list_cam_mtx_savefiles():
    return list(sorted([save_file
                        for save_file in os.listdir(os.path.join(save_dir, camera_mtx_dir))
                        if '.sav' in save_file], reverse=True))

def list_mesh_savefiles():
    return list(sorted([save_file
                        for save_file in os.listdir(os.path.join(save_dir, mesh_dir))
                        if '.sav' in save_file], reverse=True))

def mk_savedir():
    cam_mtx_path = os.path.join(save_dir, camera_mtx_dir)
    mesh_path =os.path.join(save_dir, mesh_dir)
    if not os.path.exists(cam_mtx_path):
        os.makedirs(cam_mtx_path, exist_ok=True)
    if not os.path.exists(mesh_path):
        os.makedirs(mesh_path, exist_ok=True)

def chess_option(parser):
    parser.add_argument( "chess_rows",
                           type=int,
                           metavar='Number of Chessboard rows',
                           default=7,
                           gooey_options={
                               'validator': { 'test': 'int(user_input) > 0',
                                              'message': 'Must be positive integer.'}})
    parser.add_argument("chess_cols",
                          type=int,
                          metavar='Number of Chessboard cols',
                          default=7,
                          gooey_options={
                              'validator': { 'test': 'int(user_input) > 0',
                                             'message': 'Must be positive integer.'}})

def image_path_option(parser):
    parser.add_argument('left_img_path',
                             metavar='Left Images path',
                             widget = 'DirChooser',
                             type=str,
                             )

    parser.add_argument('right_img_path',
                             metavar='Right Images path',
                             widget = 'DirChooser',
                             type=str,
                             )
    parser.add_argument('suffix',
                             metavar='Image file suffix',
                             widget='Dropdown',
                             choices=suffix_list)


def result_path_option(parser):
    parser.add_argument('--result_img_path',
                           metavar='Path to save "undistorted images" (Optional)',
                           help="If you want to save 'undistorted images', set save directory.",
                           widget='DirChooser',
                           type=str)

def cam_option(parser):
    parser.add_argument("left_cam_num",
                        type=int,
                        metavar='Left Camera number',
                        default = 0,
                        gooey_options={
                            'validator': {'test': 'int(user_input) >= 0',
                                          'message': 'Must be unsigned integer.'}})
    parser.add_argument("right_cam_num",
                            type=int,
                            metavar='Right Camera number',
                            default = 1,
                            gooey_options={
                                'validator': { 'test': 'int(user_input) >= 0',
                                               'message': 'Must be unsigned integer.'}})

def add_args_calib_img(parser: GooeyParser = GooeyParser()):
    path_parser = parser.add_argument_group("Image paths", gooey_options={'show_border': True, 'columns': 1})
    image_path_option(path_parser)

    chess_parser = parser.add_argument_group("Chessboard option", gooey_options={'show_border': True, 'columns': 2})
    chess_option(chess_parser)

    result_parser = parser.add_argument_group("Result option", gooey_options={'show_border': True, 'columns': 1})
    result_path_option(result_parser)


def add_args_calib_cam(parser : GooeyParser = GooeyParser()):
    cam_parser = parser.add_argument_group("Camera option", gooey_options={'show_border': True, 'columns': 2})
    cam_option(cam_parser)

    cap_parser = parser.add_argument_group("Capture option", gooey_options={'show_border': True, 'columns': 1})
    cap_parser.add_argument("n_cap",
                            type=int,
                            metavar='Number of capture image',
                            default=10,
                            gooey_options={
                                'validator': { 'test': 'int(user_input) > 0',
                                               'message': 'Must be positive integer.'}})
    cap_parser.add_argument('--capture_img_path',
                            metavar='Path to save "capture image" (Optional)',
                            help="If you want to save 'captured images', set save directory.",
                            widget='DirChooser',
                            type=str)

    chess_parser = parser.add_argument_group("Chessboard option", gooey_options={'show_border': True, 'columns': 2})
    chess_option(chess_parser)

    result_parser = parser.add_argument_group("Result option", gooey_options={'show_border': True, 'columns': 1})
    result_path_option(result_parser)

def add_args_disp(parser : GooeyParser = GooeyParser()):
    cam_parser = parser.add_argument_group("Camera option", gooey_options={'show_border': True, 'columns': 2})
    cam_option(cam_parser)
    cam_parser.add_argument('load_cam_mtx',
                            metavar='Load Camera matrix',
                            widget='Dropdown',
                            choices=list_cam_mtx_savefiles(),
                            gooey_options={
                                'validator': { 'test': 'user_input != "Select Option"',
                                               'message': 'Choose a save file from the list'}})

def add_args_3d_dic(parser : GooeyParser = GooeyParser()) -> GooeyParser:
    path_parser = parser.add_argument_group("Image paths", gooey_options={'show_border': True, 'columns': 1})
    image_path_option(path_parser)

    mesh_parser = parser.add_argument_group("Mesh option (Must be selected)", gooey_options={'show_border': True, 'columns': 1})
    mesh_option = mesh_parser.add_mutually_exclusive_group()
    mesh_option.add_argument('--new_mesh',
                             action='store_true',
                             metavar="Save new mesh")
    mesh_option.add_argument('--load_mesh',
                             widget='Dropdown',
                             choices=list_mesh_savefiles(),
                             metavar="Load mesh")

    dic_parser = parser.add_argument_group("DIC option", gooey_options={'show_border': True, 'columns': 2})
    dic_parser.add_argument('subpixel_method',
                            widget='Dropdown',
                            choices=subpixel_method_list,
                            metavar="Subpixel method")
    dic_parser.add_argument('ZNSSD_limit',
                            type=float,
                            metavar="ZNSSD limit",
                            default=0.4,
                            gooey_options={
                                'validator': {
                                    'test': 'float(user_input) > 0',
                                    'message': 'Must be positive.'}})
    dic_parser.add_argument('start_index',
                            type=int,
                            metavar="Start index",
                            default=0,
                            gooey_options={
                                'validator': {
                                    'test': 'int(user_input) >= 0',
                                    'message': 'Must be unsigned integer.'}})
    dic_parser.add_argument('end_index',
                            type=int,
                            metavar="End index",
                            gooey_options={
                                'validator': {
                                    'test': 'int(user_input) >= 0',
                                    'message': 'Must be unsigned integer.'}})
    dic_parser.add_argument('load_cam_mtx',
                            metavar='Load Camera matrix',
                            widget='Dropdown',
                            choices=list_cam_mtx_savefiles(),
                            gooey_options={
                                'validator': {
                                    'test': 'user_input != "Select Option"',
                                    'message': 'Choose a save file from the list'}})

    result_parser = parser.add_argument_group("Result option", gooey_options={'show_border': True, 'columns': 1})
    result_parser.add_argument('result_path',
                               metavar='Path to save result',
                               help="set save directory",
                               widget='DirChooser',
                               type=str)



@Gooey(program_name="3D DIC",
       default_size=(600,900),
       poll_external_updates = True)
def main():
    parser = GooeyParser()
    subs = parser.add_subparsers(help='commands', dest='command')

    img_calib_parser = subs.add_parser(
        'Calib_image',
        help='Camera calibrattion by saved image'
    )
    add_args_calib_img(img_calib_parser)

    cam_calib_parser = subs.add_parser(
        'Calib_cam',
        help='Camera calibration by camera input directly'
    )
    add_args_calib_cam(cam_calib_parser)

    disp_parser = subs.add_parser(
        'Disparity_map',
        help='Validate Camera calibration by visualization disparity map'
    )
    add_args_disp(disp_parser)

    dic3d_parser = subs.add_parser(
        '3D_DIC', help='Configuration DIC option')
    add_args_3d_dic(dic3d_parser)

    args = parser.parse_args()
    print(args.command)
    if args.command == 'Calib_image':
        calibration.by_imgs(args, save_dir, camera_mtx_dir)
    elif args.command == 'Calib_cam':
        calibration.by_cam(args, save_dir, camera_mtx_dir)
    elif args.command == 'Disparity_map':
        disparity_vis.run(args, save_dir, camera_mtx_dir)
    elif args.command == '3D_DIC':
        dic3d.run(args, save_dir, camera_mtx_dir, mesh_dir)

if __name__ == '__main__':
    if 'gooey-seed-ui' in sys.argv:
        print(json.dumps({'load_cam_mtx': list_cam_mtx_savefiles(),'--load_mesh': list_mesh_savefiles()}))
    else:
        mk_savedir()
        main()
