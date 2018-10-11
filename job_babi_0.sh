python qa_babi_task.py LSTM -1 sentence False --gpu=0 -H=90 --single_pass=1
python qa_babi_task.py LSTM -1 sentence True --gpu=0 -H=90 --single_pass=1
python qa_babi_task.py GRU -1 sentence False --gpu=0 -H=100 --single_pass=1
python qa_babi_task.py GRU -1 sentence True --gpu=0 -H=100 --single_pass=1

