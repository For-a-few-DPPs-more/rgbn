from squarenet.sampler import list_methods
from squarenet.sampler import samplepoints as load_dataset

def generate_dataset(size = (1000, 2), dataset = "barbara", plot_data = True, list_method = True):
    if list_method:
        print("source: squarenet.sampler from module squarenet")
        print("available datasets:", list_methods())
    points = load_dataset(method = dataset, size = size, plot_points = plot_data)
    points -= points.min()
    points /= points.max()
    return points