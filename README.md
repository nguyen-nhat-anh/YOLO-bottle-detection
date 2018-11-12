YOLOv2 object detection using keras (YAD2K):
https://github.com/allanzelener/YAD2K
# Requirements
```conda env create -f environment.yml```
 * python 3
 * numpy
 * opencv-python
 * pillow
 * h5py
 * matplotlib
 * lxml
 * tensorflow/tensorflow-gpu
 * keras
 * font FiraMono-Medium.otf: https://github.com/mozilla/Fira/blob/master/otf/FiraMono-Medium.otf <br />
# Usage
* Object detection from images <br />
```python image_detection.py --input aquafina1.jpg```
* Object detection from videos <br />
```python video_detection.py --input bottle_test.mp4```
* Webcam object detection <br />
```python realtime_detection.py```
