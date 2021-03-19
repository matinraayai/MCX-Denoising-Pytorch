import oct2py
import os
import pexpect

_oc = oct2py.Oct2Py()
_oc.addpath(os.path.dirname(__file__))


def mcx_filter(rima, v, f1, f2, rician=0, gpuid=0, bw=8):
    while True:
        try:
            return _oc.mcxfilter(rima, v, f1, f2, rician, gpuid, bw)
        except pexpect.exceptions.EOF as e:
            print(type(e))
            _oc.restart()
            _oc.addpath(os.path.dirname(__file__))
            continue


def mcx(cfg):
    while True:
        try:
            return _oc.mcx(cfg)
        except pexpect.exceptions.EOF:
            _oc.restart()
            _oc.addpath(os.path.dirname(__file__))
            continue

