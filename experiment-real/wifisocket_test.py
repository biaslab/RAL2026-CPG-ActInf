"""Talk to the Bittle WiFi dongle.

The dongle is running Petoi's stock WebServer sketch: an HTTP server on port 80,
*not* a raw serial-over-TCP bridge.  Commands go out as

    GET /actionpage?name=<command>

and the reply is just the Actions page HTML again -- the sketch forwards the
token to the NyBoard over Serial but never sends the board's answer back.  So
this transport is write-only: fine for issuing gaits, useless for reading the
IMU.  For that, flash sketch_wifidongle.ino (raw TCP bridge on port 23).
"""

import urllib.parse
import urllib.request

BITTLE_URL = "http://192.168.0.100"  # DHCP; re-check with: nmap -n -Pn -p 80 192.168.0.0/24 --open


def send(cmd, timeout=5.0):
    """Send one OpenCat token (e.g. 'khi', 'kwkF', 'd') to the robot."""
    url = f"{BITTLE_URL}/actionpage?name={urllib.parse.quote(cmd, safe='')}"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status


if __name__ == "__main__":
    print(send("khi"))
