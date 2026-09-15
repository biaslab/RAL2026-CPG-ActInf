#!/usr/bin/env bash
#
# find_bittle.sh -- locate the Bittle WiFi dongle on the network.
#
# Prints the robot's IP on STDOUT (and nothing else there), so it drops straight
# into a command line:
#
#     python run_experiment.py --wifi "$(./find_bittle.sh)" --mode rate
#
# Everything else -- progress, diagnosis, the suggested command -- goes to stderr.
#
# The dongle turns up in one of two shapes and this script identifies which:
#
#   * AP mode (sketch_wifidongle.ino): the dongle IS the access point, so it is
#     always 192.168.4.1 and it is also your default gateway. Port 23 is a raw
#     TCP<->Serial bridge, which is the ONLY transport that can read the IMU back.
#   * Station mode (Petoi's stock WebServer sketch): the dongle joined your WLAN
#     and the router handed it a DHCP lease that moves. Port 80 serves the
#     OpenCat control page, which is write-only -- fine for firing gaits, useless
#     for the experiment.
#
# Exit status: 0 = found, 1 = nothing found, 2 = usage error.

set -uo pipefail

BRIDGE_PORT=23          # sketch_wifidongle.ino raw TCP bridge
HTTP_PORT=80            # Petoi stock WebServer sketch
AP_ADDR=192.168.4.1     # fixed address in AP mode
CONNECT_TIMEOUT=0.35    # per-host TCP connect budget during the sweep [s]
PROBE_TIMEOUT=3         # how long to wait for the board to answer '?' [s]

usage() {
    cat >&2 <<'USAGE'
usage: find_bittle.sh [-a] [-p PORT] [HOST ...]

  -a         probe every candidate instead of stopping at the first hit
  -p PORT    extra TCP port to treat as a bridge port (default: 23)
  HOST ...   check only these addresses, skipping the subnet sweep

With no arguments: tries the AP address, then sweeps the local /24.
USAGE
    exit 2
}

say() { printf '%s\n' "$*" >&2; }

ALL=0
EXTRA_PORTS=()
while getopts ":ap:h" opt; do
    case "$opt" in
        a) ALL=1 ;;
        p) EXTRA_PORTS+=("$OPTARG") ;;
        h) usage ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

# ── low-level probes ─────────────────────────────────────────────────────────

# Can we open a TCP connection at all? Uses bash's /dev/tcp so the script has no
# dependency beyond coreutils -- nmap is used when present, but only to shorten
# the sweep, never as a requirement.
port_open() {
    timeout "$CONNECT_TIMEOUT" bash -c "exec 3<>/dev/tcp/$1/$2" 2>/dev/null
}

# Ask the board who it is. '?' is OpenCat's identify token (the same one
# PetoiRobot.testPort uses over USB) and is inert -- it moves no servos.
probe_bridge() {
    local ip=$1 port=$2 out
    out=$(timeout "$PROBE_TIMEOUT" bash -c \
              "exec 3<>/dev/tcp/$ip/$port && printf '?\n' >&3 && cat <&3" \
          2>/dev/null | tr -d '\r')
    [[ -n $out ]] && grep -qiE 'bittle|nybble|dof16|petoi|opencat' <<<"$out"
}

# The stock sketch answers HTTP; its page title names the firmware.
probe_http() {
    local ip=$1 body
    if command -v curl >/dev/null 2>&1; then
        body=$(curl -s --max-time 2 "http://$ip/" 2>/dev/null)
    else
        body=$(timeout 3 bash -c \
                   "exec 3<>/dev/tcp/$ip/$HTTP_PORT && \
                    printf 'GET / HTTP/1.0\r\nHost: $ip\r\n\r\n' >&3 && cat <&3" \
               2>/dev/null)
    fi
    grep -qiE 'opencat|petoi|nybble|bittle' <<<"$body"
}

# ── identify one candidate ───────────────────────────────────────────────────
# Emits the IP on stdout and a verdict on stderr. Returns 0 only for the bridge
# or the web server; a host that merely has a port open is not the robot.
FOUND=0
check_host() {
    local ip=$1 port
    for port in "$BRIDGE_PORT" "${EXTRA_PORTS[@]+"${EXTRA_PORTS[@]}"}"; do
        port_open "$ip" "$port" || continue
        if probe_bridge "$ip" "$port"; then
            printf '%s\n' "$ip"
            say "  $ip:$port  TCP bridge (sketch_wifidongle.ino) -- full duplex, IMU readable"
            say "     python run_experiment.py --wifi $ip --wifi-port $port --mode rate"
            FOUND=1
            return 0
        fi
        say "  $ip:$port  port open but no OpenCat identity -- not the robot"
    done
    if port_open "$ip" "$HTTP_PORT" && probe_http "$ip"; then
        printf '%s\n' "$ip"
        say "  $ip:$HTTP_PORT  Petoi stock WebServer -- WRITE-ONLY (no IMU)."
        say "     Flash sketch_wifidongle.ino for the port-$BRIDGE_PORT bridge the experiment needs."
        FOUND=1
        return 0
    fi
    return 1
}

# ── explicit hosts ───────────────────────────────────────────────────────────
if (($# > 0)); then
    for ip in "$@"; do
        say "checking $ip ..."
        if check_host "$ip" && ((!ALL)); then
            exit 0
        fi
    done
    ((FOUND)) && exit 0
    say "nothing answered on the given hosts."
    exit 1
fi

# ── 1. AP mode: the dongle is its own access point at a fixed address ─────────
gw=$(ip -o -4 route show default 2>/dev/null | awk '{print $3; exit}')
iface=$(ip -o -4 route show default 2>/dev/null | awk '{print $5; exit}')
cidr=$(ip -o -4 addr show dev "${iface:-lo}" 2>/dev/null | awk '{print $4; exit}')

if [[ $gw == "$AP_ADDR" ]]; then
    say "default gateway is $AP_ADDR -- you are joined to the dongle's own AP."
else
    say "checking the AP-mode address $AP_ADDR ..."
fi
if check_host "$AP_ADDR"; then
    exit 0
fi
if [[ $gw == "$AP_ADDR" ]]; then
    say "!! you are on the dongle's AP but it did not answer -- is the NyBoard powered?"
    exit 1
fi

# ── 2. Station mode: sweep the local subnet ──────────────────────────────────
if [[ -z $cidr ]]; then
    say "no IPv4 route -- connect to WiFi first, or pass an address explicitly."
    exit 1
fi
say "not on the AP; sweeping $cidr for a robot (via $iface) ..."

base=${cidr%.*}                     # 192.168.0.107/24 -> 192.168.0
self=${cidr%%/*}
mask=${cidr#*/}
if [[ $mask != 24 ]]; then
    say "note: /$mask subnet, scanning its /24 slice only."
fi

hosts=()
if command -v nmap >/dev/null 2>&1; then
    # A ping sweep skips the ~250 dead addresses; hosts that block ping are
    # picked up by the fallback below.
    mapfile -t hosts < <(nmap -n -sn "$base.0/24" -oG - 2>/dev/null |
                             awk '/Status: Up/ {print $2}')
    say "nmap: ${#hosts[@]} host(s) up"
fi
if ((${#hosts[@]} <= 1)); then
    say "falling back to a full TCP sweep of $base.1-254 (a few seconds) ..."
    hosts=()
    for i in $(seq 1 254); do hosts+=("$base.$i"); done
fi

# Probe candidates in parallel -- a serial sweep at 0.35 s/host is a minute of
# waiting, which is long enough that people stop running the script. Each worker
# appends one short line, which is atomic on an O_APPEND fd, so no lock is needed.
scratch=$(mktemp) || { say "cannot create a temp file"; exit 1; }
trap 'rm -f "$scratch"' EXIT

probe_one() {
    { port_open "$1" "$BRIDGE_PORT" || port_open "$1" "$HTTP_PORT"; } &&
        printf '%s\n' "$1"
}

n=0
for ip in "${hosts[@]}"; do
    [[ $ip == "$self" || $ip == "$gw" ]] && continue
    probe_one "$ip" >>"$scratch" &
    if ((++n % 48 == 0)); then wait; fi
done
wait

live=()
mapfile -t live < <(sort -t. -k1,1n -k2,2n -k3,3n -k4,4n "$scratch")

if ((${#live[@]} == 0)); then
    say "no host on $base.0/24 has port $BRIDGE_PORT or $HTTP_PORT open."
    say "Is the dongle powered and joined to this WLAN? Check the router's DHCP"
    say "table for an Espressif MAC (c8:c9:a3, 24:0a:c4, 5c:cf:7f, ec:fa:bc, ...)."
    exit 1
fi

say "identifying ${#live[@]} candidate(s) ..."
for ip in "${live[@]}"; do
    if check_host "$ip" && ((!ALL)); then
        exit 0
    fi
done

((FOUND)) && exit 0
say "candidates answered TCP but none identified as a Petoi board."
exit 1
