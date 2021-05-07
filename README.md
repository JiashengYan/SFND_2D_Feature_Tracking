# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load. 
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed. 
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson. 
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures. 

See the classroom instruction and code comments for more details on each of these parts. Once you are finished with this project, the keypoint matching part will be set up and you can proceed to the next lesson, where the focus is on integrating Lidar points and on object detection using deep-learning. 

## Dependencies for Running Locally
1. cmake >= 2.8
 * All OSes: [click here for installation instructions](https://cmake.org/install/)

2. make >= 4.1 (Linux, Mac), 3.81 (Windows)
 * Linux: make is installed by default on most Linux distros
 * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
 * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)

3. OpenCV >= 4.1
 * All OSes: refer to the [official instructions](https://docs.opencv.org/master/df/d65/tutorial_table_of_content_introduction.html)
 * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors. If using [homebrew](https://brew.sh/): `$> brew install --build-from-source opencv` will install required dependencies and compile opencv with the `opencv_contrib` module by default (no need to set `-DOPENCV_ENABLE_NONFREE=ON` manually). 
 * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)

4. gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using either [MinGW-w64](http://mingw-w64.org/doku.php/start) or [Microsoft's VCPKG, a C++ package manager](https://docs.microsoft.com/en-us/cpp/build/install-vcpkg?view=msvc-160&tabs=windows). VCPKG maintains its own binary distributions of OpenCV and many other packages. To see what packages are available, type `vcpkg search` at the command prompt. For example, once you've _VCPKG_ installed, you can install _OpenCV 4.1_ with the command:
```bash
c:\vcpkg> vcpkg install opencv4[nonfree,contrib]:x64-windows
```
Then, add *C:\vcpkg\installed\x64-windows\bin* and *C:\vcpkg\installed\x64-windows\debug\bin* to your user's _PATH_ variable. Also, set the _CMake Toolchain File_ to *c:\vcpkg\scripts\buildsystems\vcpkg.cmake*.


## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.

## Tasks
### MP.1 Data buffer optimization
        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        if(dataBuffer.size()>=dataBufferSize){
            dataBuffer.erase(dataBuffer.begin());
        }
        assert(dataBuffer.size() <= dataBufferSize);
        dataBuffer.push_back(frame);

### MP.2 Keypoint detection
_Lines 85105 in MidTermProject_Camera_Student.cpp_
```
        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, imgGray, false);
        }
        else if (detectorType.compare("FAST")  == 0 ||
                 detectorType.compare("BRISK") == 0 ||
                 detectorType.compare("ORB")   == 0 ||
                 detectorType.compare("AKAZE") == 0 ||
                 detectorType.compare("SIFT")  == 0)
        {
            detKeypointsModern(keypoints, imgGray, detectorType, false);
        }

        else
        {
            throw invalid_argument(detectorType + " is not a valid detectorType");
        }
```
### MP.3 Keypoint removal
_Lines 113-124 in MidTermProject_Camera_Student.cpp_
```
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            auto it = keypoints.begin();
            while(it != keypoints.end()){
                if(vehicleRect.contains(it->pt))
                    it++;
                else
                    it=keypoints.erase(it);
            }
        }
```

### MP.4 Keypoint descriptors
_Lines 157 in MidTermProject_Camera_Student.cpp_
```
descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
```

### MP.5 Descriptor matching
The function `matchDescriptors` in `matching2D_Student.cpp` contains a kind of decision tree structure, based on the settings of these string parameters:
- `descriptorCategory` either: `DES_BINARY` (binary), `DES_HOG` (histogram of gradients)
- `matcherType` either: `MAT_FLANN` (cv::FlannBasedMatcher), `MAT_BF` (brute force)
- `selectorType` either: `SEL_NN` (nearest neighbors), `SEL_KNN` (k nearest neighbors)

All descriptor type except SIFT use the descriptorCategory keyword `DES_BINARY`.

_Lines 174-178 in MidTermProject_Camera_Student.cpp_
```
            string descriptorArt; // DES_BINARY, DES_HOG
            if(descriptorType.compare("BRIEF")==0 or descriptorType.compare("BRISK")==0 or descriptorType.compare("AKAZE")==0 or descriptorType.compare("FREAK")==0 or descriptorType.compare("ORB")==0)
                descriptorArt="DES_BINARY";
            else
                descriptorArt="DES_HOG";
```
For the performance benchmarks (MP.7-9) below, `matcherType` was set to `MAT_BF` and `selectorType` was set to `SEL_KNN`, which implements match filtering based on the descriptor distance ratio.

### MP.6 Descriptor distance ratio
_Lines 36-49 in matching2D_Student.cpp_

This distance ratio filter compares the distance (SSD) between two candidate matched keypoint descriptors. A threshold of `0.8` is applied and the stronger candidate (minimum distance) is selected as the correct match. This method eliminates many false-positive keypoint matches.

    { // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2); // finds the 2 best matches
        // filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {

            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
    }

### MP.7 Performance evaluation 1
The number of keypoints within the bounding box of the preceding vehicle were counted for each detector type.

See the results in: result.txt

Harris had the fewest relevant keypoints, while the top three performers in this metric were:
1. BRISK (250-300 keypoints per image)
2. AKAZE (150-180 keypoints per image)
3. FAST  (140-160 keypoints per image)

### MP.8 Performance evaluation 2
The number of matched keypoints were then counted for each valid detector type and descriptor type combination. Note that ORB descriptors could not be extracted with SIFT detectors, and AKAZE descriptors worked only with AKAZE detectors.

The BRISK detectors  (broadly with most Keypoints) with BRIEF, SIFT, and BRISK descriptors consistently produced the largest number of matched keypoints (~280 per image).

### MP.9 Performance evaluation 3

The three fastest combinations were:
1. FAST + BRIEF
1. FAST + ORB
1. ORB + BRIEF
