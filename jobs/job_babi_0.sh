python qa_babi_task.py RUM -1 sentence False --gpu=1 -H=100 --single_pass=1 -A=tanh
python qa_babi_task.py RUM -1 sentence True --gpu=1 -H=100 --single_pass=1 -A=tanh
python qa_babi_task.py RUM -1 sentence False --gpu=1 -H=100 --single_pass=1 -LN=True
python qa_babi_task.py RUM -1 sentence True --gpu=1 -H=100 --single_pass=1 -LN=True
