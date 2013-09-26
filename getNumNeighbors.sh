#!/bin/bash
#
# by: Mike Smith, 2009

set -e -u

dir=$1
code=$2
weka=$3

files=$( ls $dir | grep arff$)

for file in $files
do
   ds=$(echo $file | sed s/\.arff// )
   echo $ds
   java -classpath ${code}:${weka}:. numNeighbors ${dir}/${ds}.arff 17 > $ds.out
done
