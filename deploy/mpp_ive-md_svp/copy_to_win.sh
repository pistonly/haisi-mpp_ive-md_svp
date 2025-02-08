#!/bin/sh
mkdir -p /root/win
mount -t nfs 192.168.0.111:/d/nfs_share /root/win -o nolock

ps aux | grep nnn_2chns | grep -v grep | awk '{print $1}' | xargs kill -9
tar cf /mnt/data/tmp.tar /mnt/data/tmp
cp /mnt/data/tmp.tar /root/win/
rm -r /mnt/data/tmp
rm /mnt/data/tmp.tar
mkdir /mnt/data/tmp
umount /root/win
