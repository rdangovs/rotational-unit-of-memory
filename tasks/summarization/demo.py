import subprocess
import requests
import sys
import re
from itertools import imap
import hashlib
import struct
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2
from make_datafiles import *

reload(sys)
sys.setdefaultencoding('latin-1')

# remove nonprintable characters
control_chars = ''.join(map(unichr, range(0, 32) + range(127, 160)))
control_char_re = re.compile('[%s]' % re.escape(control_chars))


def remove_control_chars(s):
    return control_char_re.sub('', s)


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
    g = open('./tmp/pdf_paper_clean.txt', 'wb')
    lines = f.readlines()
    for num, l in enumerate(lines):
        # step 1
        l = remove_control_chars(l)
        # step 2
        l_strip = l.replace(' ', '')
        length = len(l_strip)
        numbers = sum(imap(str.isdigit, l_strip))
        lower_case_letters = sum(imap(str.islower, l_strip))
        upper_case_letters = sum(imap(str.isupper, l_strip))
        symbols = length - numbers - lower_case_letters - upper_case_letters
        if numbers > length / 10. or symbols > length / 10. \
                or lower_case_letters < length / 2.:
            continue
        # write line to file
        g.write(l + '\n')
    print("Cleaned up the TXT file a bit in `./tmp/pdf_paper_clean.txt`.")

    # make a bin file
    with open('../../data/science_daily/advanced/inference/outbin', 'wb') as writer:
        article = get_lines('./tmp/pdf_paper_clean.txt').encode('latin-1')
        # Write to tf.Example
        tf_example = example_pb2.Example()
        tf_example.features.feature[
            'article'].bytes_list.value.extend([article])
        tf_example_str = tf_example.SerializeToString()
        str_len = len(tf_example_str)
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, tf_example_str))
    print("Created a tf.Example in `../../data/science_daily/advanced/inference/outbin`.")

    # run the summarization inference
    print("Now, running inference.")
    subprocess.call(['env',
                     'CUDA_VISIBLE_DEVICES=2,3',
                     'python3',
                     'run_summarization.py',
                     '--mode=decode',
                     '--data_path=science_daily/advanced/inference/outbin',
                     '--vocab_path=science_daily/advanced/paper2highlight/vocab',
                     '--log_root=advanced/p2h',
                     '--exp_name=rum_best',
                     '--grad_clip=1',
                     '--max_enc_steps=1000',
                     '--max_dec_steps=100',
                     '--sd=1',
                     '--rum=dec',
                     '--inference=1'])
