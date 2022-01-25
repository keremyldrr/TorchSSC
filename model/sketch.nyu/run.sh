#!/usr/bin/env bash
ngpus=${ngpus:-1}
gpus=${gpus:-0}
ew=${ew:-1}
nepoch=${nepoch:-100}
lr=${lr:-0.1}
ob=${ob:-}
port=${port:- 10097 }
dataset=${dataset:- NYUv2 }
cp=${cp:- }
while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
   fi

  shift
done
export NGPUS=$ngpus
export CUDA_VISIBLE_DEVICES=$gpus
echo python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py -p $port --lr $lr --ew $ew   $of $ob --dataset $dataset  --num_epochs $nepoch $cp #-c log/snapshot/last.pt

python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py -p $port --lr $lr --ew $ew   $of $ob --dataset $dataset  --num_epochs $nepoch $cp #-c log/snapshot/last.pt
