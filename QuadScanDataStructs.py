from collections import namedtuple


QuadImage = namedtuple("QuadImage", "k_ind k_value image_ind image")
"""
QuadImage stores one full image from a quad scan.
:param k_ind: index for the k value list for this image
:param k_value: k value for this image
:param image_ind: index for the image list for this image
:param image: image data in the form of a 2d numpy array
"""


ProcessedImage = namedtuple("ProcessedImage",
                            "k_ind k_value image_ind pic_roi line_x line_y x_cent y_cent "
                            "sigma_x sigma_y q enabled threshold")
"""
ProcessedImage stores one processed image from a quad scan:
:param k_ind: index for the k value list for this image
:param k_value: k value for this image
:param image_ind: index for the image list for this image
:param pic_roi: cropped image data in the form of a 2d numpy array
:param line_x: Summed up pic_roi in y dir
:param line_y: Summed up pic_roi in x dir
:param x_cent: Centroid x coord
:param y_cent: Centroid y coord
:param sigma_x: RMS size x coord (second moment)
:param sigma_y: RMS size y coord (second moment)
:param q: pic_roi total charge (sum)
:param enabled: boolean if this image is to be used for fitting
:param threshold: Threshold used
 
"""


ScanParam = namedtuple("ScanParam", "scan_attr_name scan_device_name scan_start_pos scan_end_pos scan_step "
                                    "scan_pos_tol scan_pos_check_interval "
                                    "measure_attr_name_list measure_device_list measure_number measure_interval")
"""
ScanParam stores scan parameters needed to perform a scan. Scan attribute with start and end positions, step length, 
tolerance of position, list of attributes to measure.
Number of steps will be int((end_pos - start_pos)/step).

:param scan_attr_name: Tango name of attribute that will be scanned
:param scan_device_name: Tango device name of scan device
:param scan_start_pos: Starting position of scan attribute
:param scan_end_pos: End position of scan attribute
:param scan_step: Step length during scan.
:param scan_pos_tol: Tolerance of scan parameter when moving to the next position. Absolute tolerance used by default. 
:param scan_pos_check_interval: Time between scan pos reads. Should be fast enough that little time is wasted waiting,
                                but slow enough that the device is not swamped.
:param measure_attr_name_list: List of attribute tango names to measure at each scan position.
:param measure_device_list: List of tango devices, one for each of the measure attributes, even if they belong to the 
                            same device.
:param measure_number: Number of measurements to take for each attribute for the scan positions.
:param measure_interval: Time between consecutive measurements at a scan pos (should be the same as the rep rate of 
                         the machine). 
"""


ScanResult = namedtuple("ScanResult", "pos_list measure_list timestamp_list")
"""
ScanResult stores the result from a scan in three lists. The measurement list is a list of measurements of 
the measured attributes for each scan position. This in turn contains a list of the measurements taken at that point. 

:param pos_list: List of positions that were scanned.
:param measure_list: List of lists of measurements. 
:param timestamp_list: List of timestamps were the scan positions were reached.
"""

AcceleratorParameters = namedtuple("AcceleratorParameters", "electron_energy quad_length quad_screen_dist k_max k_min "
                                                            "num_k num_images cal quad_name screen_name "
                                                            "roi_center roi_dim")
"""
AcceleratorParameters stores accelerator parameters used during a scan for fitting purposes. 

:param electron_energy: Electron energy in MeV.
:param quad_length: Length of the quad in m.  
:param quad_screen_dist: Distance between the quad and the screen in m.
:param k_max: Max k value
:param k_min: Min k value
:param num_k: Number of k values recorded
:param num_images: Number of images recorded for each k value
:param cal: Pixel calibration, m/pixel
:param quad_name: Tango name of quad 
:param screen_name: Tango name of screen
:param roi_center: x and y coord of ROI center, pixels
:param roi_dim: x and y size of ROI, pixels 
"""


SectionDevices = namedtuple("SectionDevices", "sect_quad_dict sect_screen_dict")
"""
SectionDevices stores the quads and screens in each of the measurement sections in dictionaries. 
The sections are ms1, ms2, ms3, sp02.

:param sect_quad_dict: Dictionary with entries for each section, where the entries are lists of the quads in that
                       section. The quads are of the type SectionQuad.
:param sect_screen_dict: Dictionary with entries for each section, where the entries are lists of the screens in that
                         section. The screens are of the type SectionScreen.
"""


SectionQuad = namedtuple("SectionQuad", "name position length mag crq polarity")
"""
SectionQuad stores quad data for a quadrupole magnet in a section.

:param name: Base name of the quad, e.g. QB-01
:param position: Longitudinal position of the quad along the linac.
:param length: Quad length
:param mag: Tango MAG name of the quad, e.g. I-MS1/MAG/QB-01
:param crq: Tango name of the circuit connected to the magnet.
:param polarity: Magnet polarity
"""


SectionScreen = namedtuple("SectionScreen", "name position liveviewer beamviewer limaccd screen")
"""
SectionScreen stores screen data for a screen in a section.

:param name: Screen short name, e.g. SCRN-01
:param position: Longitudinal position of the screen along the linac.
:param liveviewer: Tango name of the camera liveviewer, e.g. lima/liveviewer/I-MS1-
:param beamviewer: Tango name of the camera beamviewer, e.g. lima/beamviewer/I-MS1-
:param limaccd: Tango name of the camera limaccd, e.g. lima/limaccd/I-MS1-
:param screen: Tango DIA name of the screen, e.g. I-MS1/DIA/SCRN-01
"""


QuadScanData = namedtuple("QuadScanData", "acc_params images proc_images")
"""
QuadScanData stores data from a quadscan. Scan parameters in acc_params, raw images, and processed images.

:param acc_params: Named tuple AcceleratorParameters
:param images: List of images for the scan.
:param proc_images: List of process images (roi crop, thresholding, median filtering).  
"""


FitResult = namedtuple("FitResult", "poly alpha beta eps eps_n gamma_e fit_data residual")
"""
FitResult stores fit parameters for a quad scan.
"""


class DataStore(object):
    def __init__(self):
        self.quad_scan_data = None      # type: QuadScanData
        self.section = None
        self.quad = None                # type: SectionQuad
        self.screen = None              # type: SectionScreen
        self.fit_result = None          # type: FitResult
        self.scan_param = None          # type: ScanParam
