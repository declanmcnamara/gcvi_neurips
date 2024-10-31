python run.py --multirun training.loss=favi training.device='cpu' training.lr=1e-4 training.favi_offset=0,30,60,90,120,150,180,210,240,270,300,330
python run.py --multirun training.loss=iwbo training.device='cpu' training.lr=1e-2 training.favi_offset=0,30,60,90,120,150,180,210,240,270,300,330 training.steps=5000
python plots.py
