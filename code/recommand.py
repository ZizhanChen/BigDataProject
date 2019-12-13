from mrjob.job import MRJob
import os
import numpy as np

root_dir = "/home/kenny/Course/BigData/project/BigDataProject/code/kmeans_data"
alpha = 0.5
beta = 0.9

def split_line(line):
    id, labels, mean_rgb, mean_audio, cluster = line.split("\t")
    id = id.strip()
    # print(str.encode(id))

    labels = labels.strip(']').strip('[').split(',')
    labels = [int(i.strip()) for i in labels]

    mean_rgb = mean_rgb.strip(']').strip('[').split(',')
    mean_rgb = [float(i.strip()) for i in mean_rgb]

    mean_audio = mean_audio.strip(']').strip('[').split(',')
    mean_audio = [float(i.strip()) for i in mean_audio]

    cluster = cluster.strip()

    return id, labels, mean_rgb, mean_audio, cluster


# the structure of target:
# [id:'abcd', labels:[1,2,3,4], mean_rgb:[1024 float], mean_audio:[128 float]]
def get_target(target_id):
    clusters = os.listdir(root_dir)
    for cluster in clusters:
        files = os.listdir(os.path.join(root_dir, cluster))
        for file in files:
            f = open(os.path.join(root_dir, cluster, file), "r")
            batch = f.readlines()
            for line in batch:
                id, labels, mean_rgb, mean_audio, cluster = split_line(line)
                if id == target_id:
                    return line


def jaccard_similarity(a, b):
    a = set(a)
    b = set(b)
    unions = len(a.union(b))
    intersections = len(a.intersection(b))
    return intersections / unions


def cosine_similarity(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))


def similarity(t_labels, t_mean_rgb, t_mean_audio, labels, mean_rgb, mean_audio):
    jaccard_sim = jaccard_similarity(t_labels, labels)
    cosine_sim_1 = cosine_similarity(t_mean_rgb, mean_rgb)
    cosine_sim_2 = cosine_similarity(t_mean_audio, mean_audio)
    return alpha * jaccard_sim + (1 - alpha) * (beta * cosine_sim_1 + (1-beta) * cosine_sim_2)


def get_k_nearest(target, k=10):
    dictionary = dict()
    smallest_key = ''
    t_id, t_labels, t_mean_rgb, t_mean_audio, t_cluster = split_line(target)

    files = os.listdir(os.path.join(root_dir, t_cluster))
    for file in files:
        f = open(os.path.join(root_dir, t_cluster, file), "r")
        batch = f.readlines()
        for line in batch:
            id, labels, mean_rgb, mean_audio, cluster = split_line(line)
            sim = similarity(t_labels, t_mean_rgb, t_mean_audio, labels, mean_rgb, mean_audio)
            if len(dictionary) <= k:
                dictionary[id] = sim
                smallest_key = sorted(dictionary.items(), key=lambda d: d[1])[0][0]
            elif sim > dictionary.get(smallest_key):
                dictionary.pop(smallest_key)
                dictionary[id] = sim
                smallest_key = sorted(dictionary.items(), key=lambda d: d[1])[0][0]
    reslut = sorted(dictionary.items(), key=lambda d: d[1], reverse=True)
    return [d[0] for d in reslut], [d[1] for d in reslut]


class MRRecommend(MRJob):
    def mapper(self, _, line):
        for target_id in line.split():
            target = get_target(target_id)
            yield (target, _)

    def reducer(self, target, _):
        nearests, sims = get_k_nearest(target)
        yield (nearests, sims)


if __name__ == '__main__':
    MRRecommend.run()
    #nearests, sims = get_k_nearest('qqab', k=10)
    #print(nearests)
    #print(sims)
