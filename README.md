# VideoLab2

Installation:

In order to install the BASE and MEGA approaches, the following line needs to be executed in the terminal:

```bash
conda create --name MEGA -y python=3.7 && source activate MEGA && conda install -y ipython pip && pip install ninja yacs cython matplotlib tqdm opencv-python scipy && conda install -y pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch && export INSTALL_DIR=$PWD && git clone https://github.com/cocodataset/cocoapi.git && cd cocoapi/PythonAPI && python setup.py build_ext install && cd $INSTALL_DIR && git clone https://github.com/mcordts/cityscapesScripts.git && cd cityscapesScripts && python setup.py build_ext install && cd $INSTALL_DIR && git clone https://github.com/NVIDIA/apex.git && cd apex && git checkout a1df804 && python setup.py build_ext install && cd $INSTALL_DIR && git clone https://github.com/Scalsol/mega.pytorch.git && cd mega.pytorch && python setup.py build develop && pip install 'pillow<7.0.0' && unset INSTALL_DIR
```

After this, download and move the files R_101.pth and MEGA_R_101.pth into the folder mega.pytorch.

Before running the script for the approaches, there are errors that need to be fixed beforehand due to the incompatibility of some of the module versions and their used attributes. Below you will find the locations of the files that needs to be replaced. Download these from the repository and replace their counterparts in their respective locations. The updated files replace some dependencies from the module torch that are incompatible with the version used.



FILE NUMBER ONE: grad_scaler.py

Error fixed:
AttributeError: module 'torch.cuda' has no attribute 'amp'

Location:
```bash
/.conda/envs/MEGA/lib/python3.7/site-packages/apex-0.1-py3.7.egg/apex/transformer/amp/grad_scaler.py
```


FILE NUMBER TWO: utils.py

Error fixed:
AttributeError: module 'torch.distributed' has no attribute '_all_gather_base

Location:
```bash
/.conda/envs/MEGA/lib/python3.7/site-packages/apex-0.1-py3.7.egg/apex/transformer/utils.py
```


FILE NUMBER THREE: mappings.py

Error fixed:
AttributeError: module 'torch.distributed' has no attribute '_reduce_scatter_base'

Location:
```bash
/.conda/envs/MEGA/lib/python3.7/site-packages/apex-0.1-py3.7.egg/apex/transformer/tensor_parallel/mappings.py
```


FILE NUMBER FOUR: common.py

Error fixed:
AttributeError: module 'torch.cuda' has no attribute 'amp'

Location:
```bash
/.conda/envs/MEGA/lib/python3.7/site-packages/apex-0.1-py3.7.egg/apex/transformer/pipeline_parallel/schedules/common.py
```

FILE NUMBER FIVE: fwd_bwd_no_pipelining.py

Error fixed:
AttributeError: module 'torch.cuda' has no attribute 'amp'

Location:
```bash
/.conda/envs/MEGA/lib/python3.7/site-packages/apex-0.1-py3.7.egg/apex/transformer/pipeline_parallel/schedules/fwd_bwd_no_pipelining.py
```


FILE NUMBER SIX: fwd_bwd_pipelining_with_interleaving.py

Error fixed:
AttributeError: module 'torch.cuda' has no attribute 'amp'

Location:
```bash
/.conda/envs/MEGA/lib/python3.7/site-packages/apex-0.1-py3.7.egg/apex/transformer/pipeline_parallel/schedules/fwd_bwd_pipelining_with_interleaving.py
```


FILE NUMBER SEVEN: fwd_bwd_pipelining_without_interleaving.py

Error fixed:
AttributeError: module 'torch.cuda' has no attribute 'amp'

Location:
```bash
/.conda/envs/MEGA/lib/python3.7/site-packages/apex-0.1-py3.7.egg/apex/transformer/pipeline_parallel/schedules/fwd_bwd_pipelining_without_interleaving.py
```


FILE NUMBER EIGHT: predictor.py

Error fixed:
cv2.error: OpenCV(4.10.0) :-1: error: (-5:Bad argument) in function 'putText'
> Overload resolution failed:
>  - Can't parse 'org'. Sequence item with index 0 has a wrong type
>  - Can't parse 'org'. Sequence item with index 0 has a wrong type

Location:
```bash
/mega.pytorch/demo/predictor.py
```



After all the steps have been taken, you can run the following example line in order to create the videos from the frames.

```bash
python demo/demo.py ${METHOD} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--visualize-path ${IMAGE-FOLDER}] [--suffix ${IMAGE_SUFFIX}][--output-folder ${FOLDER}] [--output-video]
```
Example for the BASE approach:

```bash
python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG"     --visualize-path /image_folder     --output-folder /output_videos --output-video
```

Example for the MEGA approach:

```bash
python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml R_101.pth --suffix ".JPEG"     --visualize-path /image_folder     --output-folder /output_videos --output-video
```

















