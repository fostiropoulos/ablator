#!/bin/sh
cd "$(dirname "$0")"
# pytest needs sshd service to be running
/usr/sbin/sshd -D &
exec pytest ./../tests/