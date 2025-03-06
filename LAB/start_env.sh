#!/bin/sh
#File start_env.sh

if [ -d ".venv" ]; then
    rm -rf .venv
fi

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt