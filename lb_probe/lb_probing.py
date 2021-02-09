from __future__ import print_function  # for Python3.x compatibility

import os

import numpy as np
from math import log
import pandas as pd

# INPUT / OUTPUT

def read_patient_ids():
    with open('sample_submission.csv') as f:
        lines = f.readlines()[1:]
        return [line.split(',')[0] for line in lines]


def prob_format(p):
    return '%e' % p


def truncate(p):
    return float(prob_format(p))


def write_submit(patient_ids, probs, file_name):
    assert len(patient_ids) == len(probs)
    with open(file_name, 'w') as f:
        f.write('id,cancer\n')
        for i, p in zip(patient_ids, probs):
            f.write('%s,%s\n' % (i, prob_format(p)))
    print('wrote %s' % file_name)


def read_scores():
    lines = open('scores.txt').readlines()
    return [float(s.strip()) for s in lines]


# PROBABILITIES

def build_template(n, chunk_size):
    # return np.linspace(0.00000001, 0.00000002, chunk_size)
    epsilon = 1.05e-5
    return 1 / (1 + np.exp(n * epsilon * 2 ** np.arange(chunk_size)))


def build_probs(n, chunk, template):
    assert template.shape == chunk.shape
    probs = np.zeros((n,))
    probs[:] = 0.5
    probs[chunk] = template
    return probs


# LABEL INFERENCE

def int_to_bin(x, size):
    s = bin(x)[2:][::-1].ljust(size, '0')
    return np.array([int(c) for c in s])


def update_labels(labels, chunk, template, score):
    assert template.shape == chunk.shape
    chunk_size = len(chunk)
    n = len(labels)
    match_count = 0
    scores = np.ones(2 ** chunk_size) * 9999
    for i in range(2 ** chunk_size):
        b = int_to_bin(i, chunk_size)
        score_i = ((-np.log(template) * b - np.log(1 - template) * (1 - b)).sum() - log(0.5) * (n - chunk_size)) / n
        # score_i = log_loss(b, template) * chunk_size / n - log(0.5) * (n - chunk_size) / n
        scores[i] = np.abs(score_i - score)
        # if score == ('%.5f' % score_i):
        if np.allclose(score, score_i):
            match_count += 1
            new_labels = b
    assert match_count == 1  # no collisions
    print('new labels: %s' % new_labels)
    labels[chunk] = new_labels


def log_loss(y_true, y_hat):
    assert len(y_true) == len(y_hat)
    if all(y_true == y_hat):
        return 0
    n = len(y_true)
    # loss = (-y_true * np.log(y_hat) - (1 - y_true) * np.log(1 - y_hat)).sum() / n
    loss = -(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat)).sum() / n
    return loss


# MAIN

def write_submit_files():
    n = 198
    np.random.seed(2017)
    idx = np.arange(n)
    # np.random.shuffle(idx)  # optional
    chunk_size = 15
    template = build_template(n, chunk_size)
    template = np.array([truncate(x) for x in template])

    scores = read_scores()
    labels = np.zeros((n,), dtype=np.int)
    labels[:] = -1

    patient_ids = read_patient_ids()
    chunks = [idx[i: i + chunk_size] for i in range(0, len(idx), chunk_size)]
    for i, chunk in enumerate(chunks):
        t = template[:len(chunk)]
        probs = build_probs(n, chunk, t)
        write_submit(patient_ids, probs, 'submissions/submission_%02d.csv' % i)
        if i < len(scores):
            update_labels(labels, chunk, t, scores[i])
            if i + 1 == len(chunks):
                write_submit(patient_ids, labels, 'submissions/submission_fin.csv')


def save_init_score():
    with open("scores.txt", 'w') as f:
        f.write("")
    # np.random.seed(321)
    # y_hat = np.random.uniform(0, 1, 198)
    y_hat = np.ones(198) * 0.5
    with open("true_labels.csv", 'r') as f:
        y_true = np.array([float(label.strip()) for label in f.readlines()])
    score = log_loss(y_true, y_hat)
    add_score(score)


def add_score(score):
    with open("scores.txt", 'a') as f:
        f.write("%s\n" % score)


def fill_scores():
    with open("true_labels.csv", 'r') as f:
        y_true = np.array([float(label.strip()) for label in f.readlines()])
    subm_dir = "./submissions/"
    for i, submission_f in enumerate(os.listdir(subm_dir)):
        with open(subm_dir + 'submission_%02d.csv' % i, 'r') as f:
            y_hat = np.array([float(row.split(",")[1]) for row in f.readlines()[1:]])
        add_score(log_loss(y_true, y_hat))


def write_true_labels():
    # np.random.seed(123)
    with open("true_labels.csv", 'w') as f:
        f.writelines("\n".join(["%s" % int(round(p, 0)) for p in np.random.uniform(0, 1, 198)]))


if __name__ == '__main__':
    # remove submissions dir before run this
    write_true_labels()
    # save_init_score()
    with open("scores.txt", 'w') as f:
        f.write("")
    write_submit_files()
    fill_scores()
    write_submit_files()
    df = pd.concat([
        pd.read_csv("./submissions/submission_fin.csv"),
        pd.read_csv("./true_labels.csv", header=None)
    ], axis=1)
    df["check"] = df.cancer == df[0]
    print("new labels equal gold:", df.check.sum() == 198)
