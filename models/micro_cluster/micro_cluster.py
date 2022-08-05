import math as math
import numpy as np


class MicroCluster:
    """
    Implementation of the MicroCluster data structure for the CluStream algorithm
    Parameters
    ----------
    nb_points: is the number of points in the cluster
    identifier: is the identifier of the cluster (take -1 if the cluster result from merging two clusters)
    merge: is used to indicate whether the cluster is resulting from the merge of two existing ones
    id_list: is the id list of merged clusters
    linear_sum: is the linear sum of the points in the cluster.
    squared_sum: is  the squared sum of all the points added to the cluster.
    linear_time_sum: is  the linear sum of all the timestamps of points added to the cluster.
    squared_time_sum: is  the squared sum of all the timestamps of points added to the cluster.
    m: is the number of points considered to determine the relevance stamp of a cluster
    update_timestamp: is used to indicate the last update time of the cluster
    """

    def __init__(self, nb_points=0, identifier=0, id_list=None, linear_sum=None, squared_sum=None, m=100):
        self.nb_points = nb_points
        self.identifier = identifier
        self.id_list = id_list
        self.linear_sum = linear_sum
        self.squared_sum = squared_sum
        self.m = m
        self.radius_factor = 1.8
        self.epsilon = 0.00005
        self.min_variance = math.pow(1, -5)

    def get_center(self):
        center = [self.linear_sum[i] / self.nb_points for i in range(len(self.linear_sum))]
        return center

    def get_weight(self):
        return self.nb_points

    def insert(self, new_point):
        self.nb_points += 1
        for i in range(len(new_point)):
            self.linear_sum[i] += new_point[i]
            self.squared_sum[i] += math.pow(new_point[i], 2)


    def merge(self, micro_cluster):
        ## micro_cluster must be removed
        self.nb_points += micro_cluster.nb_points
        self.linear_sum += micro_cluster.linear_sum
        self.squared_sum += micro_cluster.squared_sum

        if (self.identifier != -1):
            if (micro_cluster.identifier != -1):
                self.id_list = [self.identifier, micro_cluster.identifier]
            else:
                micro_cluster.id_list.append(self.identifier)
                self.id_list = micro_cluster.id_list.copy()
            self.identifier = -1
        else:
            if (micro_cluster.identifier != -1):
                self.id_list.append(micro_cluster.identifier)
            else:
                self.id_list.extend(micro_cluster.id_list)

    def get_radius(self):
        if self.nb_points == 1:
            return 0
        return self.get_deviation() * self.radius_factor


    def get_deviation(self):
        ls_mean = self.linear_sum / self.nb_points
        ss_mean = self.squared_sum / self.nb_points
        variance = ss_mean - ls_mean ** 2
        radius = np.sqrt(np.sum(variance))
        return radius

    def get_variance_vec(self):
        variance_vec = list()
        for i in range(len(self.linear_sum)):
            ls_mean = self.linear_sum[i] / self.nb_points
            ss_mean = self.squared_sum[i] / self.nb_points
            variance = ss_mean - math.pow(ls_mean, 2)
            if variance <= 0:
                if variance > - self.epsilon:
                    variance = self.min_variance

            variance_vec.append(variance)
        return variance_vec

    def inverse_error(self, x):
        z = (math.sqrt(math.pi) * x)
        inv_error = z / 2
        z_prod = math.pow(z, 3)
        inv_error += (1 / 24) * z_prod

        z_prod *= math.pow(z, 2)
        inv_error += (7 / 960) * z_prod

        z_prod = math.pow(z, 2)
        inv_error += (127 * z_prod) / 80640

        z_prod = math.pow(z, 2)
        inv_error += (4369 / z_prod) * 11612160

        z_prod = math.pow(z, 2)
        inv_error += (34807 / z_prod) * 364953600

        z_prod = math.pow(z, 2)
        inv_error += (20036983 / z_prod) * 0x797058662400d
        return z_prod

if __name__ == '__main__':
    micro_cluster=MicroCluster()
