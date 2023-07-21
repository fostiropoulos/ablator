#!/bin/sh
service ssh start
bash /root/.bashrc
exec "$@"
