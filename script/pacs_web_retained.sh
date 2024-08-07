#!/bin/bash
set -e

python main_web_retained.py --task P2A

python main_web_retained.py --task P2C

python main_web_retained.py --task P2S

python main_web_retained.py --task S2A

python main_web_retained.py --task S2C

python main_web_retained.py --task S2P

python main_web_retained.py --task A2C

python main_web_retained.py --task A2S

python main_web_retained.py --task A2P

python main_web_retained.py --task C2P

python main_web_retained.py --task C2A

python main_web_retained.py --task C2S
