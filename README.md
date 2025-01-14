# Phaseno_docker

## How to run docker image

use below command to run docker image
```
docker run --gpus all -v {Input File Directory}:/usr/src/app/example kasssap/phaseno
```
Input file directory must have two files
  1. .csv file of stations separeted with "|" (Column names must be the same with tr_station file)
  2. A file named "waveforms" consists of .mseed data (Name formats must be the same with files in the repository) 
>If any error accured please check the Predict.py file and build the docker image with changed python file again. 

## Model
Model I'm using is the same with original Phaseno repository:
https://github.com/sun-hongyu/PhaseNO/tree/master
