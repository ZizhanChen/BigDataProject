import argparse
import os
import random
import re
import sys
import warnings

import numpy as np
import csv

sys.tracebacklimit = 0


class KmeansRunner(object):

    @staticmethod
    def retrieveData(inputFile):

        with open("test.csv", "r") as readcsvfile:
            points = []
            for line in readcsvfile.readlines():
                contents = line.split("\t")

                mean_rgb = contents[2]
                mean_rgb = mean_rgb.strip(']').strip('[').split(',')
                mean_rgb = [float(i.strip()) for i in mean_rgb]
                rgb_vector = np.array(mean_rgb)

                mean_audio = contents[3]
                mean_audio = mean_audio.strip('\n').strip(']').strip('[').split(',')
                mean_audio = [float(i.strip()) for i in mean_audio]
                audio_vector = np.array(mean_audio)

                data_point = np.array((mean_rgb, mean_audio))
                datalist = data_point.tolist()
                points.append(datalist)

            return np.array(points)

    def initialCentroids(self, inputFile):
        with open(inputFile, "r") as inputFile:
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
    def retrieveCentroids(centroidsfile):
        with open(centroidsfile, "r") as inputFile:
            output_data = inputFile.readlines()

        id_list = []
        rgb_list = []
        audio_list = []
        i = 1
        for point in output_data:
            contents = point.split(']')
            id = contents[0].split('[')[0].strip('\t')
            id_list.append(int(id))

            mean_rgb = contents[0].split('[')[2]
            mean_rgb = mean_rgb.strip('[').strip(',').strip(']').strip('[').split(',')
            mean_rgb = [float(i.strip()) for i in mean_rgb]
            rgb_vector = np.array(mean_rgb)
            rgb_list.append(rgb_vector)

            mean_audio = contents[1]
            mean_audio = mean_audio.strip('\n').strip(',]').strip(']').strip('[').split(',')
            mean_audio = [float(i.strip()) for i in mean_audio]
            audio_vector = np.array(mean_audio)
            audio_list.append(audio_vector)

        return id_list, rgb_list, audio_list

    def retrieveLabels(self, dataFile, centroidsFile):
        data_points = self.retrieveData(dataFile)
        centroids = self.retrieveCentroids(centroidsFile)
        labels = []
        for data_point in data_points:
            distances = [np.linalg.norm(data_point - centroid)
                         for centroid in centroids]
            cluster = np.argmin(distances)
            labels.append(int(cluster))
        return labels

    @staticmethod
    def writeCentroids(rgb_list, audio_list):
        f = open(CENTROIDS_FILE, "w+")
        writer = csv.writer(f, delimiter='\t')
        for i in range(len(rgb_list)):
            writer.writerow([rgb_list[i].tolist(), audio_list[i].tolist()])
        f.close()


CENTROIDS_FILE = "./kmeans_conf/center.csv"
OUTPUT_FILE = "./kmeans_conf/output.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="k-means algorithm"
                                                 " implementation on Hadoop",
                                     epilog="Go ahead and try it!")
    parser.add_argument("inputFile", type=str,
                        help="Input data points for the clustering algorithm.")
    parser.add_argument("centroids", type=int,
                        help="Number of clusters.")
    args = parser.parse_args()
    
    data = args.inputFile
    k = args.centroids
    instanceKmeans = KmeansRunner()
    rgb_list, audio_list = instanceKmeans.initialCentroids(CENTROIDS_FILE)

    i = 1
    while True:
        print("k-means iteration #%i" % i)
        command = "python kmeans_mr.py " \
                  + data + " --k=" \
                  + str(k) + " --centroids=" \
                  + CENTROIDS_FILE + " > " + OUTPUT_FILE
        os.system(command)
        
        cluster_id_list, new_rgb_list, new_audio_list = instanceKmeans.retrieveCentroids(OUTPUT_FILE)
    
        if len(rgb_list) != len(new_rgb_list):
            if len(rgb_list) != len(new_rgb_list):
                for i in range(len(cluster_id_list)):
                    rgb_list[cluster_id_list[i]] = new_rgb_list[i]
                    audio_list[cluster_id_list[i]] = new_audio_list[i]
            else:
                rgb_list = new_rgb_list
                audio_list = new_audio_list
            instanceKmeans.writeCentroids(rgb_list, audio_list)
        else:
            break
        i += 1
        