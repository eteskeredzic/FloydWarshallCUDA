#! /bin/bash

num=128

while [ $num -le $1 ]
do
./fwblocked $num $2
num=$(( num * 2 ))
done

