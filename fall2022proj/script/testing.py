import matlab.engine
import numpy as np
import pyLasaDataset as lasa


def get_gmm_from_matlab(position, velocity):
    est_options = {
        # "type": "diag",
        "type": 0,
        "maxK": 15.0,
        "fixed_K": matlab.double([]),
        "samplerIter": 20.0,
        "do_plots": 0,
        "sub_sample": 1,
        "estimate_l": 1.0,
        "l_sensitivity": 2.0,
        "length_scale": matlab.double([]),
    }

    pos_array = matlab.double(position.T)
    vel_array = matlab.double(velocity.T)

    priors, mu, sigma = matlab_eng.fit_gmm(
        pos_array, vel_array, est_options, nargout=0)

    priors = np.array(priors)
    mu = np.array(mu)
    sigma = np.array(sigma)

    return priors, mu, sigma


if (__name__) == "__main__":

    start_global_matlab_engine = True
    if start_global_matlab_engine and not "matlab_eng" in locals():
        matlab_eng = matlab.engine.start_matlab()
        print("Matlab engine started")
    a, b, c = matlab_eng.simple_script(nargout=3)
    print(a)
    print(b)
    print(c)

    data = lasa.DataSet.BendedLine.demos[0]
    pos = data.pos.transpose()
    vel = data.vel.transpose()
    print(pos.shape)
    print(vel.shape)

# import matplotlib as mpl
# import matplotlib.pyplot as plt

# import numpy as np

# from sklearn import datasets
# from sklearn.mixture import GaussianMixture
# from sklearn.model_selection import StratifiedKFold

# import pyLasaDataset as lasa

# # DataSet object has all the LASA handwriting data files
# # as attributes, eg:
# angle_data = lasa.DataSet.Angle
# sine_data = lasa.DataSet.Sine


# # Each Data object has attributes dt and demos (For documentation,
# # refer original dataset repo:
# # https://bitbucket.org/khansari/lasahandwritingdataset/src/master/Readme.txt)
# dt = angle_data.dt
# demos = angle_data.demos  # list of 7 Demo objects, each corresponding to a
# # repetition of the pattern


# # Each Demo object in demos list will have attributes pos, t, vel, acc
# # corresponding to the original .mat format described in
# # https://bitbucket.org/khansari/lasahandwritingdataset/src/master/Readme.txt
# demo_0 = demos[5]
# pos = demo_0.pos  # np.ndarray, shape: (2,2000)
# vel = demo_0.vel  # np.ndarray, shape: (2,2000)
# acc = demo_0.acc  # np.ndarray, shape: (2,2000)
# t = demo_0.t      # np.ndarray, shape: (1,2000)


# # To visualise the data (2D position and velocity) use the plot_model utility
# lasa.utilities.plot_model(lasa.DataSet.BendedLine)  # give any of the available
# # pattern data as argument


# colors = ["navy", "turquoise", "darkorange"]


# def make_ellipses(gmm, ax):
#     for n, color in enumerate(colors):
#         if gmm.covariance_type == "full":
#             covariances = gmm.covariances_[n][:2, :2]
#         elif gmm.covariance_type == "tied":
#             covariances = gmm.covariances_[:2, :2]
#         elif gmm.covariance_type == "diag":
#             covariances = np.diag(gmm.covariances_[n][:2])
#         elif gmm.covariance_type == "spherical":
#             covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
#         v, w = np.linalg.eigh(covariances)
#         u = w[0] / np.linalg.norm(w[0])
#         angle = np.arctan2(u[1], u[0])
#         angle = 180 * angle / np.pi  # convert to degrees
#         v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
#         ell = mpl.patches.Ellipse(
#             gmm.means_[n, :2], v[0], v[1], 180 + angle, color=color
#         )
#         ell.set_clip_box(ax.bbox)
#         ell.set_alpha(0.5)
#         ax.add_artist(ell)
#         ax.set_aspect("equal", "datalim")


# iris = datasets.load_iris()

# # Break up the dataset into non-overlapping training (75%) and testing
# # (25%) sets.
# skf = StratifiedKFold(n_splits=4)
# # Only take the first fold.
# train_index, test_index = next(iter(skf.split(iris.data, iris.target)))


# X_train = iris.data[train_index]
# y_train = iris.target[train_index]
# X_test = iris.data[test_index]
# y_test = iris.target[test_index]

# n_classes = len(np.unique(y_train))

# # Try GMMs using different types of covariances.
# estimators = {
#     cov_type: GaussianMixture(
#         n_components=n_classes, covariance_type=cov_type, max_iter=20, random_state=0
#     )
#     for cov_type in ["spherical", "diag", "tied", "full"]
# }

# n_estimators = len(estimators)

# plt.figure(figsize=(3 * n_estimators // 2, 6))
# plt.subplots_adjust(
#     bottom=0.01, top=0.95, hspace=0.15, wspace=0.05, left=0.01, right=0.99
# )


# for index, (name, estimator) in enumerate(estimators.items()):
#     # Since we have class labels for the training data, we can
#     # initialize the GMM parameters in a supervised manner.
#     estimator.means_init = np.array(
#         [X_train[y_train == i].mean(axis=0) for i in range(n_classes)]
#     )

#     # Train the other parameters using the EM algorithm.
#     estimator.fit(X_train)

#     h = plt.subplot(2, n_estimators // 2, index + 1)
#     make_ellipses(estimator, h)

#     for n, color in enumerate(colors):
#         data = iris.data[iris.target == n]
#         plt.scatter(
#             data[:, 0], data[:, 1], s=0.8, color=color, label=iris.target_names[n]
#         )
#     # Plot the test data with crosses
#     for n, color in enumerate(colors):
#         data = X_test[y_test == n]
#         plt.scatter(data[:, 0], data[:, 1], marker="x", color=color)

#     y_train_pred = estimator.predict(X_train)
#     train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
#     plt.text(0.05, 0.9, "Train accuracy: %.1f" %
#              train_accuracy, transform=h.transAxes)

#     y_test_pred = estimator.predict(X_test)
#     test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
#     plt.text(0.05, 0.8, "Test accuracy: %.1f" %
#              test_accuracy, transform=h.transAxes)

#     plt.xticks(())
#     plt.yticks(())
#     plt.title(name)

# plt.legend(scatterpoints=1, loc="lower right", prop=dict(size=12))


# plt.show()


# bendedLine = lasa.DataSet.BendedLine

# angle_data = lasa.DataSet.Angle
# demos = bendedLine.demos
# demo_0 = demos[0]
# pos = demo_0.pos  # np.ndarray, shape: (2,2000)
# vel = demo_0.vel  # np.ndarray, shape: (2,2000)
# acc = demo_0.acc  # np.ndarray, shape: (2,2000)
# t = demo_0.t  # np.ndarray, shape: (1,2000)

# X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

# X = pos.transpose()
# gm = GaussianMixture(n_components=5, random_state=0).fit(X)

# # print(gm.means_)

# colors = ["black", "darkgreen", "cyan",
#           "orangered", "olive"]  # "violet", "salmon"]

# plt.plot(pos[0], pos[1], markersize=0.2, marker=".", color="blue")

# for n, color in enumerate(colors):
#     plt.scatter(gm.means_[n][0], gm.means_[n][1], marker="x", color=color)

# plt.show()
