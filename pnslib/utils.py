"""Utility functions.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""
from __future__ import print_function, absolute_import

import os
try:
    import subprocess32 as sp
except ImportError:
    import subprocess as sp


def speak(msg_str, options=None):
    """Text-to-Speech via espeak and mplayer.

    # Parameters
    msg_str : str
        The candidate string to speak
    options : list
        espeak program options, use default config if None
    """
    speak_base = ["/usr/bin/espeak"]
    if options is not None and isinstance(options, list):
        speak_config = options
    else:
        speak_config = ["-ven+f3", "-k5", "-s150"]
    mplayer_base = ["/usr/bin/mplayer", "/tmp/pns_speak.wav"]

    speak_cmd = speak_base+speak_config + \
        [msg_str, ">", "/tmp/pns_speak.wav", "|"]+mplayer_base

    return sp.check_output(speak_cmd, stderr=sp.PIPE).decode("UTF-8")
