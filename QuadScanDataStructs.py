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
                            "k_ind image_ind pic_roi line_x line_y x_cent y_cent sigma_x sigma_y q enabled")
"""
ProcessedImage stores one processed image from a quad scan:
:param k_ind: index for the k value list for this image
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
 
"""

ScanParam = namedtuple("ScanParam", "scan_attr_name scan_device_name scan_start_pos scan_end_pos scan_step "
                                    "scan_pos_tol scan_pos_check_interval "
                                    "measure_attr_name_list measure_device_list measure_number measure_interval")
""""""

ScanResult = namedtuple("ScanResult", "pos_list measure_list timestamp_list")
""""""

AcceleratorParameters = namedtuple("AcceleratorParameters", "electron_energy quad_length quad_screen_dist")
SectionDevices = namedtuple("SectionDevices", "sect_quad_dict sect_screen_dict")
ImageList = namedtuple("ImageList", "daq_data images proc_images")