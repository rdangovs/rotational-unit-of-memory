python qa_babi_task.py RUM -1 sentence False --gpu=0 -H=150 --single_pass=1 -U=False
python recall_task.py RUM -T=50 --gpu=0 -U=False -H=100 -L=1
python copying_task.py RUM --gpu=0 -U=False -H=150
