#!/usr/bin/env bash
ps -eo cmd,nlwp | grep -w 'chen'|awk '{sum+=$2 ; print $0} END{print sum}'