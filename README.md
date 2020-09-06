# 3D-DIC
## Introduction

This is a 3D DIC(Digital Image Correlation) program based on python language.(GPU support)

![gui](https://github.com/dudgus1727/3D-DIC/blob/master/resources/gui.png)

## requirements
- python3
- pandas
- scikit-image
- scipy
- matplotlib
- natsort
- gooey
- networkx
- pytorch
- cupy

## 1. Stereo Camera Calibration
### - Take picture and calibraion

![picture](https://github.com/dudgus1727/3D-DIC/blob/master/resources/take_pictures.png)
### - Calibration directly by cam input 

![cam](https://github.com/dudgus1727/3D-DIC/blob/master/resources/cam.png)

## 2. Check calibration

You can check calibraion result by visualizing disparity map.

![disp](https://github.com/dudgus1727/3D-DIC/blob/master/resources/disparity.png)


## 3. 3D DIC

![3d_dic](https://github.com/dudgus1727/3D-DIC/blob/master/resources/3d_dic.png)

Test dataset : [https://sem.org/3ddic]

### (1) Set area, grid, subset

![set](https://github.com/dudgus1727/3D-DIC/blob/master/resources/grid.png)

### (2) Visualize 3d disparity

![3d](https://github.com/dudgus1727/3D-DIC/blob/master/resources/3d_visualize.png)

### (3) Save result

Disparity map image and csv file are saved.

![3d](https://github.com/dudgus1727/3D-DIC/blob/master/resources/result.png)

