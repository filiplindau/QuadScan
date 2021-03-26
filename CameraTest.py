"""
Test response of lima camera.

:created: 2021-03-19
:author: Filip Lindau <filip.lindau@maxiv.lu.se>
"""

import PyTango as pt
import time
import numpy as np
import logging
import json

logger = logging.getLogger("Task")
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

f = logging.Formatter("%(asctime)s - %(name)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


cam_name = "g-v-csdb-0:10000/lima/liveviewer/i-ms1-dia-scrn-01"
scrn_name = "g-v-csdb-0:10000/i-ms1/dia/scrn-01"
beam_name = "g-v-csdb-0:10000/lima/beamviewer/i-ms1-dia-scrn-01"
mag_name = "g-v-csdb-0:10000/i-ms1/mag/qb-01"
mag_name = "g-v-csdb-0:10000/i-ms1/mag/crq-01"

dev_cam = pt.DeviceProxy(cam_name)
dev_beam = pt.DeviceProxy(beam_name)
dev_scr = pt.DeviceProxy(scrn_name)
dev_mag = pt.DeviceProxy(mag_name)


def capture_data(dev_cam, t_cap, screen, mag):
    data = list()
    pic_list = list()
    t0 = time.time()
    t = time.time()
    while t - t0 < t_cap:
        a = dev_cam.read_attributes(["image", "imagecounter"])
        # pic_roi = a[0].value[roi[2]:roi[3], roi[0]:roi[1]]
        bkg = a[0].value[roi[2]-50:roi[2], roi[0]:roi[1]].mean() * 1.2
        pic_roi = np.maximum(a[0].value[roi[2]:roi[3], roi[0]:roi[1]] - bkg, 0)
        pic_list.append(pic_roi)
        d = [a[0].time.totime(), float(pic_roi.sum()), float(a[1].value), screen, mag]
        data.append(d)
        logger.debug("Read attributes {0:.1f}/{1}\n"
                     "Time {2}, Image {3:.0f}, Counter {4:.0f}".format(t - t0, t_cap, d[0], d[1], d[2]))
        t = time.time()
    return data, pic_list


framerate = dev_cam.framerate
logger.info("Camera framerate: {0:.1f} Hz".format(framerate))
roi = json.loads(dev_beam.roi)

data = list()
pic_list = list()
t_capture = 60.0

m = 0.8
dev_mag.mainfieldcomponent = m
quad = dev_mag.mainfieldcomponent
dev_scr.command_inout("moveout")
logger.info("\n=============================\n"
            "Screen out\n"
            "=============================")
time.sleep(5.0)
t0 = time.time()
d, pic_r = capture_data(dev_cam, t_capture, 0, quad)
data.extend(d)
pic_list.extend(pic_r)

dev_scr.command_inout("movein")
logger.info("\n=============================\n"
            "Screen in\n"
            "=============================")
sa = dev_scr.read_attribute("State")
t_scr_in = sa.time.totime()
d, pic_r = capture_data(dev_cam, t_capture, 1, quad)
data.extend(d)
pic_list.extend(pic_r)

m = -3.5
dev_mag.mainfieldcomponent = m
logger.info("\n=============================\n"
            "Mag {0}\n"
            "=============================".format(m))
sm = dev_mag.read_attribute("mainfieldcomponent")
t_mag = sm.time.totime()
d, pic_r = capture_data(dev_cam, t_capture, 1, sm.value)
data.extend(d)
pic_list.extend(pic_r)

m = 0.8
dev_mag.mainfieldcomponent = m
logger.info("\n=============================\n"
            "Mag {0}\n"
            "=============================".format(m))
sm = dev_mag.read_attribute("mainfieldcomponent")
t_mag2 = sm.time.totime()
d, pic_r = capture_data(dev_cam, t_capture, 1, sm.value)
data.extend(d)
pic_list.extend(pic_r)

da = np.array(data)
