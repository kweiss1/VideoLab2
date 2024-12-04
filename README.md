# VideoLab2

Installation:

conda create --name MEGA -y python=3.7 && source activate MEGA && conda install ipython pip -y && pip install ninja yacs cython matplotlib tqdm opencv-python scipy && conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch -y && export INSTALL_DIR=$PWD && git clone https://github.com/cocodataset/cocoapi.git && cd cocoapi/PythonAPI && python setup.py build_ext install && cd $INSTALL_DIR && git clone https://github.com/mcordts/cityscapesScripts.git && cd cityscapesScripts && python setup.py build_ext install && cd $INSTALL_DIR && git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py build_ext install && cd $INSTALL_DIR && git clone https://github.com/Scalsol/mega.pytorch.git && cd mega.pytorch && python setup.py build develop && pip install 'pillow<7.0.0' && unset INSTALL_DIR
