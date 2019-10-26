## Execution instructions

This file is meant as documentation on the setup of running this on exactly
mirceta's Azure VM. What has been changed and how to now use it.


### Open project

- First run the VM on Azure
- Navigate to Desktop/pycharm-community-2019.1.3/bin
- Run '''./pycharm.sh'''
- Open SPGraph repository at /home/km/superpoint_graph

### Transfer data to work on

- learning/learning.py and partition/partition.py contain info on how to do this.
- Transfer data to big data disk, as OS partition is very small

### Running SPGraph

- Need to be run in the exact following order:
- Run partition.py
  - The .cpp libraries are not globally installed. (Cannot run from pycharm)
  - Navigate to SPGraph folder
  - Run python partition/partition.py --dataset custom_dataset --ROOT_PATH /media/km/ad02048a-21c3-4454-b1b4-58c5a99df3c5/workspace2 --voxel_width 0.5 --reg_strength 0.8 --ver_batch 500
- Run dataset.py
  - Can run directly from pycharm: select the correct debug config and run.
- Run main.py
args = --CUSTOM_SET_PATH /media/km/ad02048a-21c3-4454-b1b4-58c5a99df3c5/workspace2
  - Can run direcly from pycharm -||-
  args = --dataset custom_dataset --CUSTOM_SET_PATH /media/km/ad02048a-21c3-4454-b1b4-58c5a99df3c5/workspace2 --epochs 20 --lr_steps '[275,320]' --test_nth_epoch 2 --model_config gru_10,f_2 --nworkers 2 --pc_attribs xyzelpsv --odir "results" --ptn_nfeat_stn 8 --batch_size 1 --cuda 1
