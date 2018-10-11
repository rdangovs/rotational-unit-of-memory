python qa_babi_task.py RUM -1 sentence False --gpu=1 -H=100 --single_pass=1
python qa_babi_task.py RUM -1 sentence True --gpu=1 -H=100 --single_pass=1
python qa_babi_task.py RUM -1 sentence False --gpu=1 -H=100 --single_pass=1 -L=1
python qa_babi_task.py RUM -1 sentence True --gpu=1 -H=100 --single_pass=1 -L=1


# python qa_babi_task.py RUM -1 sentence True --gpu=1 -H=100 --single_pass=1 -N=1.0
# python qa_babi_task.py RUM -1 sentence True --gpu=1 -H=100 --single_pass=1 -A=tanh
# python qa_babi_task.py RUM -1 sentence True --gpu=1 -H=100 --single_pass=1 -LN=True


# python qa_babi_task.py RUM -1 sentence True --gpu=0 -H=100 --single_pass=1 -LA=1
# python qa_babi_task.py RUM -1 sentence True --gpu=0 -H=100 --single_pass=1 -N=1.0 -LA=1
# python qa_babi_task.py RUM -1 sentence True --gpu=0 -H=100 --single_pass=1 -A=tanh -LA=1
# python qa_babi_task.py RUM -1 sentence True --gpu=0 -H=100 --single_pass=1 -LN=True -LA=1
