import PyTango as pt
from PyTango.server import Device, DeviceMeta
from PyTango.server import attribute, command
from PyTango.server import device_property
import sys
import numpy as np


class DummyMagnet(Device):
    __metaclass__ = DeviceMeta

    # --- Operator attributes
    #
    mainfieldcomponent = attribute(label='mainfieldcomponent',
                                   dtype=float,
                                   access=pt.AttrWriteType.READ_WRITE,
                                   unit="k",
                                   format="%4.3f",
                                   min_value=-100.0,
                                   max_value=100.0,
                                   fget="get_mainfieldcomponent",
                                   fset="set_mainfieldcomponent",
                                   memorized=True,
                                   hw_memorized=True,
                                   doc="Magnetic field", )

    # --- Device properties
    #
    length = device_property(dtype=float,
                             doc="Quad length",
                             default_value=0.2)

    polarity = device_property(dtype=int,
                               doc="Polarity",
                               default_value=1)

    __si = device_property(dtype=float,
                           doc="Position",
                           default_value=5.0)

    def __init__(self, klass, name):
        self.mainfieldcomponent_data = 0.0
        Device.__init__(self, klass, name)

    def init_device(self):
        self.debug_stream("In init_device:")
        Device.init_device(self)
        self.set_state(pt.DevState.ON)

        self.debug_stream("init_device finished")

    def get_mainfieldcomponent(self):
        return self.mainfieldcomponent_data + np.random.rand() * 0.001

    def set_mainfieldcomponent(self, k):
        self.info_stream("In set_mainfieldcomponent: New k={0:.3f}".format(k))
        self.mainfieldcomponent_data = k
        return True

    # def device_name_factory(self, list):
    #     self.info_stream("Adding server MS1/MAG/QF01")
    #     list.append("MS1/MAG/QF01")
    #
    #     return list


if __name__ == "__main__":
    args = sys.argv
    pt.server.server_run((DummyMagnet,),  args=args)
