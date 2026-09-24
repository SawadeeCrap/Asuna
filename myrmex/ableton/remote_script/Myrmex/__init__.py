# Myrmex Remote Script for Ableton Live 11 / 12.
#
# Install: copy this "Myrmex" folder to
#   macOS:   ~/Music/Ableton/User Library/Remote Scripts/Myrmex
#   Windows: \Users\<you>\Documents\Ableton\User Library\Remote Scripts\Myrmex
# restart Live, then Settings > Link, Tempo & MIDI > Control Surface: "Myrmex"
# (Input / Output: None).  It streams transport + upcoming clip notes as OSC to
# 127.0.0.1:9100, where `myrmex live` listens.
from .surface import MyrmexSurface


def create_instance(c_instance):
    return MyrmexSurface(c_instance)
