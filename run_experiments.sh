#!/bin/bash

# Script to reproduce results

envs=(
#    "FetchReach-v1"
    "FetchPush-v1"
    "FetchPickAndPlace-v1"
	)

for ((i=0;i<5;i+=1))
do 
	for env in ${envs[*]}
	do
		python Run_SAC+HER+IGL.py \
		--env $env \
		--seed $i
	done
done

for ((i=0;i<5;i+=1))
do
	for env in ${envs[*]}
	do
		python Run_SAC+HER.py \
		--env $env \
		--seed $i
	done
done