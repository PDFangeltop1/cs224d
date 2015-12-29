#!/bin/bash

# verbose
set -x
###################
# Update items below for each train/test
###################

# training params
epochs=200
step=1e-2
wvecDim=30

# for RNN2 only, otherwise doesnt matter
middleDim=25

model="RNN2Drop" #either RNN, RNN2, RNN3, RNTN, or DCNN


######################################################## 
# Probably a good idea to let items below here be
########################################################
if [ "$model" == "RNN2DropMaxout" ]; then 
    outfile="models/${model}_wvecDim_${wvecDim}_middleDim_${middleDim}_step_${step}_2.bin"
elif [ "$model" == "RNN2Drop" ]; then
    outfile="models/${model}_wvecDim_${wvecDim}_middleDim_${middleDim}_step_${step}_2.bin"    
else
    outfile="models/${model}_wvecDim_${wvecDim}_step_${step}_tanh2.bin"
fi


echo $outfile


python runNNet.py --step $step --epochs $epochs --outFile $outfile \
                --middleDim $middleDim --outputDim 5 --wvecDim $wvecDim --model $model 

