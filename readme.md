# QuadScan
### Introduction
QuadScan application for measuring twiss parameters in the maxiv linac. It uses Tango connected 
devices for quad magnets and screen camera. It can also analyze saved scans of both single
quad (done by this program or Jason's matlab program) and multi quad scans (done by this program)

Computations are done with numpy and scipy. Image processing is done with openCV (cv2). 
PIL is used for some image file handling.

Both single quad and multi quad (4 quads) scans can be done. The available sections are MS1, MS2, MS3, 
and possibly SP02 at a later stage.

### Prerequisites
python3
numpy, scipy, pyqt5, PIL, cv2, pyqtgraph >=0.11
PyTango for taking scans


### Example usage

Start with: 

```python3 QuadScanGuiMulti.py```