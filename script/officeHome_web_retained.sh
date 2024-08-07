#!/bin/bash
set -e

python main_web_retained.py --task art2realWorld

python main_web_retained.py --task art2clipart

python main_web_retained.py --task art2product

python main_web_retained.py --task realWorld2art

python main_web_retained.py --task realWorld2clipart

python main_web_retained.py --task realWorld2product

python main_web_retained.py --task clipart2art

python main_web_retained.py --task clipart2realWorld

python main_web_retained.py --task clipart2product

python main_web_retained.py --task product2art

python main_web_retained.py --task product2realWorld

python main_web_retained.py --task product2clipart
