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
  - Run python partition/partition.py --dataset custom_dataset --ROOT_PATH /media/km/<GUID>/workspace --voxel_width 0.5 --reg_strength 0.8 --ver_batch 500
- Run dataset.py
  - Can run directly from pycharm: select the correct debug config and run.
- Run main.py
  - Can run direcly from pycharm -||-
