import re

import numpy as np
from mrjob.job import MRJob
from mrjob.job import MRStep
import os
import csv
import sys

class KmeansAlgorithm(MRJob):
    def configure_options(self):
        super(KmeansAlgorithm, self).configure_options()
        self.add_passthrough_option(
            "--k", type="int", help="Number of clusters.")
        self.add_file_option("--centroids")

    @staticmethod
    def retrieveCentroids(centroidsFile):
        with open(centroidsFile, "r") as inputFile:
            output_data = inputFile.readlines()

        rgb_list = []
        audio_list = []

        for point in output_data:
            contents = point.split('\t')
            mean_rgb = contents[0]
            mean_rgb = mean_rgb.strip(']').strip('[').split(',')
            mean_rgb = [float(i.strip()) for i in mean_rgb]
            rgb_vector = np.array(mean_rgb)
            rgb_list.append(rgb_vector)

            mean_audio = contents[1]
            mean_audio = mean_audio.strip('\n').strip(']').strip('[').split(',')
            mean_audio = [float(i.strip()) for i in mean_audio]
            audio_vector = np.array(mean_audio)
            audio_list.append(audio_vector)

        return rgb_list, audio_list

    @staticmethod
    def calculateNewCentroids(cluster, partial_sums):
        total_sum, total_counter = partial_sums.__next__()
        total_sum = np.array(total_sum)

        sum_rgb = np.array(total_sum[0])
        sum_audio = np.array(total_sum[1])
        for partial_sum, counter in partial_sums:
            sum_rgb += np.array(np.array(partial_sum)[0])
            sum_audio += np.array(np.array(partial_sum)[1])
            total_counter += counter
        rgb_result = [e / total_counter for e in sum_rgb.tolist()]
        audio_result = [e / total_counter for e in sum_audio.tolist()]
        yield cluster, (rgb_result, audio_result)



    def assignPointtoCluster(self, _, line):
        contents = line.split("\t")
        id = contents[0]
        id = id.strip()

        labels = contents[1]
        labels = labels.strip(']').strip('[').split(',')
        labels = [int(i.strip(" ")) for i in labels]

        mean_rgb = contents[2]
        mean_rgb = mean_rgb.strip(']').strip('[').split(',')
        mean_rgb = [float(i.strip()) for i in mean_rgb]
        rgb_vector = np.array(mean_rgb)

        mean_audio = contents[3]
        mean_audio = mean_audio.strip('\n').strip(']').strip('[').split(',')
        mean_audio = [float(i.strip()) for i in mean_audio]
        audio_vector = np.array(mean_audio)

        cen_rgb_list, cen_audio_list = self.retrieveCentroids(self.options.centroids)

        rgb_dis = [np.linalg.norm(rgb_vector - cen_rgb) for cen_rgb in cen_rgb_list]
        audio_dis = [np.linalg.norm(audio_vector - cen_audio) for cen_audio in cen_audio_list]
        a = 0.5
        b = 0.5
        dis = [a * rgb_dis[i] + b * audio_dis[i]  for i in range(len(audio_dis))]
        cluster = np.argmin(dis)
        data_point = np.array((mean_rgb, mean_audio))

        clusterfile = str(cluster) + ".csv"
        csvfilepath = os.path.join("/home/kenny/Course/BigData/project/youtube-8m-master/kmeans_data", clusterfile)
        if os.path.exists(csvfilepath):
            csvfile = open(csvfilepath, "a")
        else :
            csvfile = open(csvfilepath, "w")

        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow([id, labels, mean_rgb, mean_audio, cluster])
        csvfile.close()

        yield int(cluster), data_point.tolist()

    @staticmethod
    def fakereducer(cluster, data_points):
        yield cluster, data_points

    @staticmethod
    def calculatePartialSum(cluster, data_points):
        sum_points = np.array(data_points.__next__())
        sum_rgb = np.array(sum_points[0])
        sum_audio = np.array(sum_points[1])
        counter = 1
        for data_point in data_points:
            sum_rgb += np.array(np.array(data_point)[0])
            sum_audio += np.array(np.array(data_point)[1])
            counter += 1
        partial_sum = np.array((sum_rgb.tolist(), sum_audio.tolist()))
        yield cluster, (partial_sum.tolist(), counter)

    def steps(self):
        return [MRStep(mapper=self.assignPointtoCluster,
                       reducer=self.fakereducer)]

if __name__ == "__main__":
    KmeansAlgorithm.run()