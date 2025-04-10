#!/bin/sh
salloc -Jlab3_interactive --partition=ice-cpu,coc-cpu -N1 --ntasks-per-node=16 -C"intel&core24" --mem-per-cpu=1G --time=00:20:00
exit 0
