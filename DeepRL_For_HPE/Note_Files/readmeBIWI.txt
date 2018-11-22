Biwi Kinect Head Pose Database

This database is made available for non-commercial use such as research and education. By using the data, you agree to cite our work:

@article{Fanelli_IJCV,
 author = {Fanelli, Gabriele and Dantone, Matthias and Gall, Juergen and Fossati, Andrea and Gool, Luc},
 title = {Random Forests for Real Time 3D Face Analysis},
 journal = {Int. J. Comput. Vision},
 volume = {101},
 number = {3},
 month = feb,
 year = {2013},
 pages = {437--458}
} 

The database contains 24 sequences acquired with a Kinect sensor. 20 people (some were recorded twice - 6 women and 14 men) were recorded while turning their heads, sitting in front of the sensor, at roughly one meter of distance. 

The data was meant for estimation (not tracking), that is why some frames are missing: That's where the automatic annotation failed. Keep this in mind if you do tracking, also when comparing your results to the ones reported in our papers, as they refer to frame-by-frame estimation experiments.

For each sequence, the corresponding .obj file represents a head template of the neutral face of that specific person.
In each folder, two .cal files contain calibration information for the depth and the color camera, e.g., the intrinsic camera matrix of the depth camera and the global rotation and translation to the rgb camera. This information can be used to align the RGB and depth data. the Please note that the calibration is not the same for each sequence.

For each frame, a _rgb.png and a _depth.bin files are provided, containing color and depth data. The depth is already segmented (the background is removed using a threshold on the distance) and the binary files compressed (an example c code is provided to show how to read and write the depth data into memory).
The _pose.txt and _pose.bin files contain the ground truth information, i.e., the location of the center of the head in 3D and the head rotation. The .txt files encode the rotation as a matrix, while the .bin file contain 6 floats representing the head center coordinates followed by pith, yaw, and roll angles.

The data is provided 'as is', without any warranty, use at your own risk.

The following list shows the correspondences between the subjects (M/FXX) and the sequence number.

01 - F01
02 - F02
03 - F03
04 - F04
05 - F05
06 - F06
07 - M01
08 - M02
09 - M03
10 - M04
11 - M05
12 - M06
13 - M07
14 - M08
15 - F03
16 - M09
17 - M10
18 - F05
19 - M11
20 - M12
21 - F02
22 - M01
23 - M13
24 - M14





