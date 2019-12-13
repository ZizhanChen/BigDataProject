import tensorflow as tf
import numpy as np
import os
import random
import csv
import sys
import mrjob


def encode_to_tfrecords(tfrecords_filename, data_num):
    ''' write into tfrecord file '''
    if os.path.exists(tfrecords_filename):
        os.remove(tfrecords_filename)

    writer = tf.python_io.TFRecordWriter('./' + tfrecords_filename)

    for i in range(data_num):
        img_raw = np.random.randint(0, 255, size=(56, 56))
        img_raw = img_raw.tostring()
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
        writer.write(example.SerializeToString())

    writer.close()
    return 0


def decode_from_tfrecords(filename_queue, is_batch):
    num_classes = 3862
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) 
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'id': tf.io.FixedLenFeature([], tf.string),
                                           "labels": tf.io.VarLenFeature(tf.int64),
                                           'mean_rgb': tf.io.FixedLenFeature([1024], tf.float32),
                                           'mean_audio': tf.io.FixedLenFeature([128], tf.float32),
                                       }) 
    id = features["id"]
    labels = features["labels"]
    mean_rgb = features["mean_rgb"]
    mean_audio = features["mean_audio"]

    if is_batch:
        batch_size = 1000
        min_after_dequeue = batch_size
        capacity = 5 * batch_size
        id, labels, mean_rgb, mean_audio = tf.train.shuffle_batch([id, labels, mean_rgb, mean_audio],
                                              batch_size=batch_size,
                                              num_threads=6,
                                              capacity=capacity,
                                              min_after_dequeue=min_after_dequeue)
    return id, labels, mean_rgb, mean_audio


if __name__ == '__main__':
    a = sys.argv[1]
    is_batch = False
    prepath = 'data/'
    outpath = './outdata/'
    inputfile = a + ".tfrecord"
    outputfile = a + ".csv"
    train_filename = os.path.join(prepath, inputfile)

    filename_queue = tf.train.string_input_producer([train_filename], num_epochs=None) 
    id, labels, mean_rgb, mean_audio = decode_from_tfrecords(filename_queue, is_batch=is_batch)

    csvfile = open(os.path.join(outpath, outputfile), "w")
    writer = csv.writer(csvfile, delimiter='\t')

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        cnt = 0
        try:
            for i in range(1000):
                i, l, mr, ma = sess.run([id, labels, mean_rgb, mean_audio])
                i = str(i, encoding = "utf-8")
                if is_batch:
                    cnt = cnt + 3
                else:
                    l = l[1]
                    l = l.tolist()
                    cnt = cnt + 1

                mr = mr.tolist()
                ma = ma.tolist()
                cl = random.randint(0,100)

                writer.writerow([i, l, mr, ma])


        except tf.errors.OutOfRangeError:
            print("cnt = ", cnt)
            print('Done reading')
        finally:
            coord.request_stop()

        coord.request_stop()
        coord.join(threads)
