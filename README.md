# VideoLab2

Installation:

In order to install the BASE and MEGA approaches, the following line needs to be executed in the terminal:

```bash
conda create --name MEGA -y python=3.7 && source activate MEGA && conda install -y ipython pip && pip install ninja yacs cython matplotlib tqdm opencv-python scipy && conda install -y pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch && export INSTALL_DIR=$PWD && git clone https://github.com/cocodataset/cocoapi.git && cd cocoapi/PythonAPI && python setup.py build_ext install && cd $INSTALL_DIR && git clone https://github.com/mcordts/cityscapesScripts.git && cd cityscapesScripts && python setup.py build_ext install && cd $INSTALL_DIR && git clone https://github.com/NVIDIA/apex.git && cd apex && git checkout a1df804 && python setup.py build_ext install && cd $INSTALL_DIR && git clone https://github.com/Scalsol/mega.pytorch.git && cd mega.pytorch && python setup.py build develop && pip install 'pillow<7.0.0' && unset INSTALL_DIR
```

python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG" \
    --visualize-path /home/alumnos/e521295/Downloads/image_folder \
    --output-folder /home/alumnos/e521295/output_videos --output-video



FILE NUMBER ONE:

 class GradScaler(torch.cuda.amp.GradScaler):
AttributeError: module 'torch.cuda' has no attribute 'amp'

/home/alumnos/e521295/.conda/envs/MEGA/lib/python3.7/site-packages/apex-0.1-py3.7.egg/apex/transformer/amp/grad_scaler.py



FILE NUMBER TWO:

AttributeError: module 'torch.distributed' has no attribute '_all_gather_base

/home/alumnos/e521295/.conda/envs/MEGA/lib/python3.7/site-packages/apex-0.1-py3.7.egg/apex/transformer/utils.py



FILE NUMBER THREE:

AttributeError: module 'torch.distributed' has no attribute '_reduce_scatter_base'

/home/alumnos/e521295/.conda/envs/MEGA/lib/python3.7/site-packages/apex-0.1-py3.7.egg/apex/transformer/tensor_parallel/mappings.py



FILE NUMBER FOUR:

AttributeError: module 'torch.cuda' has no attribute 'amp'

/home/alumnos/e521295/.conda/envs/MEGA/lib/python3.7/site-packages/apex-0.1-py3.7.egg/apex/transformer/pipeline_parallel/schedules/common.py


FILE NUMBER FIVE:

AttributeError: module 'torch.cuda' has no attribute 'amp'

/home/alumnos/e521295/.conda/envs/MEGA/lib/python3.7/site-packages/apex-0.1-py3.7.egg/apex/transformer/pipeline_parallel/schedules/fwd_bwd_no_pipelining.py



FILE NUMBER SIX:

AttributeError: module 'torch.cuda' has no attribute 'amp'

/home/alumnos/e521295/.conda/envs/MEGA/lib/python3.7/site-packages/apex-0.1-py3.7.egg/apex/transformer/pipeline_parallel/schedules/fwd_bwd_pipelining_with_interleaving.py



FILE NUMBER SEVEN:

AttributeError: module 'torch.cuda' has no attribute 'amp'

/home/alumnos/e521295/.conda/envs/MEGA/lib/python3.7/site-packages/apex-0.1-py3.7.egg/apex/transformer/pipeline_parallel/schedules/fwd_bwd_pipelining_without_interleaving.py



FILE NUMBER EIGHT:

cv2.error: OpenCV(4.10.0) :-1: error: (-5:Bad argument) in function 'putText'
> Overload resolution failed:
>  - Can't parse 'org'. Sequence item with index 0 has a wrong type
>  - Can't parse 'org'. Sequence item with index 0 has a wrong type

/home/alumnos/e521295/mega.pytorch/demo/predictor.py




RUN:

python demo/demo.py base configs/vid_R_101_C4_1x.yaml R_101.pth --suffix ".JPEG"     --visualize-path /home/alumnos/e521295/Downloads/image_folder     --output-folder /home/alumnos/e521295/output_videos --output-video


python demo/demo.py mega configs/MEGA/vid_R_101_C4_MEGA_1x.yaml R_101.pth --suffix ".JPEG"     --visualize-path /home/alumnos/e521295/Downloads/image_folder     --output-folder /home/alumnos/e521295/output_videos --output-video


















