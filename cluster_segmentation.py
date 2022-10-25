import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

if __name__ == '__main__':

    pic = plt.imread('images/1.jpeg')/255
    print(pic.shape)

    # plt.imshow(pic)

    pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])

    kmeans = KMeans(n_clusters=5, random_state=0).fit(pic_n)
    pic2 = kmeans.cluster_centers_[kmeans.labels_]
    cluster_pic = pic2.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
    # plt.imshow(cluster_pic)

    cv2.imshow('img', cluster_pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
