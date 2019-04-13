#!/usr/bin/env bash

# Default parameters
BP_PATH="$HOME/bp"
VIS_PORT=8097

while [[ $# -gt 0 ]]; do
    key="$1"
    case "$key" in
        -p|--path)
            BP_PATH="$2"
            shift ; shift ;;
        -v|--viz-port)
            VIS_PORT="$2"
            shift ; shift ;;
        *)
            shift ;;
    esac
done

# ENVIRONMENT VARIABLES

BACHPROPAGATION_ROOT_PATH="$BP_PATH"
PYTHONPATH="${PYTHONPATH}:$BP_PATH/src"
echo "> Exported ENV variables"

# VISDOM VISUALIZATION

if lsof -Pi :8097 -sTCP:LISTEN -t >/dev/null ; then
    while : ; do
        printf "Visdom server already running, do you want to restart it? [Y/N]: "; read -r restart
        [[ "$restart" == "Y" || "$restart" == "N" ]] && break
        echo "> Invalid option, try again"
    done

    if [[ "$restart" = "Y" ]]; then
        kill $(lsof -t -i:8097)
        visdom -logging_level ERROR -port 8097 > /dev/null &
        echo "> Restarted Visdom server"
    fi
else
    visdom -logging_level ERROR -port 8097 > /dev/null &
    echo "> Started Visdom server"
fi

# NGROK TUNNELING

ngrok http 8097 > /dev/null &
echo "> Tunneling URL with Ngrok"

WEB_HOOK_URL=$(curl --silent --connect-timeout 10 http://localhost:4040/api/tunnels | \
       pipenv run python -c "import json,sys;obj=json.load(sys.stdin);print(obj['tunnels'][0]['public_url'])")

echo "Visdom tunneled in $WEB_HOOK_URL"


# RUN MODEL
echo ${BACHPROPAGATION_ROOT_PATH}

pipenv run python src/main.py

