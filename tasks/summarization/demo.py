import subprocess
import requests
import sys
import struct

import tensorflow as tf
from tensorflow.core.example import example_pb2
from make_datafiles import *

if __name__ == '__main__':
    # preamble
    print("Welcome to demo.")
    url = sys.argv[1]
    pdf_paper = requests.get(url)
    with open('./tmp/pdf_paper.pdf', 'wb') as f:
        f.write(pdf_paper.content)
    print("Wrote the PDF in `./tmp/pdf_paper.pdf`.")
    subprocess.call(['pdftotext', './tmp/pdf_paper.pdf'])
    print("Converted the PDF to TXT in `./tmp/pdf_paper.txt`.")

    # add hoc processing of the text
    f = open('./tmp/pdf_paper.txt', 'r')
    g = open('./tmp/pdf_paper_clean.txt', 'w')
    lines = f.readlines()
    for num, l in enumerate(lines):
        l_strip = l.replace(' ', '')
        length = len(l_strip)
        numbers = sum(map(str.isdigit, l_strip))
        lower_case_letters = sum(map(str.islower, l_strip))
        upper_case_letters = sum(map(str.isupper, l_strip))
        symbols = length - numbers - lower_case_letters - upper_case_letters
        if numbers > length / 10. or symbols > length / 10. \
                or lower_case_letters < length / 2.:
            continue
        g.write(l + '\n')
    print("Cleaned up the TXT file a bit in `./tmp/pdf_paper_clean.txt`.")

    # make a bin file
    with open('../../data/science_daily/advanced/inference/outbin', 'wb') as writer:
        article = get_lines('./tmp/pdf_paper_clean.txt')
        # write to tf.Example
        tf_example = example_pb2.Example()
        tf_example.features.feature[
            'article'].bytes_list.value.extend([article.encode('utf-8')])
        tf_example_str = tf_example.SerializeToString()
        str_len = len(tf_example_str)
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, tf_example_str))
    print("Created a tf.Example in `../../data/science_daily/advanced/inference/outbin`.")

    # run the summarization inference
    print("Now, running inference.")
    subprocess.call(['env',
                     'CUDA_VISIBLE_DEVICES=0',
                     'python3',
                     'run_summarization.py',
                     '--mode=decode',
                     '--data_path=science_daily/advanced/inference/outbin',
                     '--vocab_path=science_daily/advanced/paper2highlight/vocab',
                     '--log_root=advanced/test',
                     '--exp_name=rum',
                     '--grad_clip=1',
                     '--max_enc_steps=1000',
                     '--max_dec_steps=100',
                     '--sd=1',
                     '--rum=dec',
                     '--inference=1'])