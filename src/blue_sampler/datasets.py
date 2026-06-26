from squarenet.sampler import samplepoints, list_methods

def generate_dataset(size = (1000, 2), dataset = "barbara", plot_data = True, list_method = True):
    if list_method:
        print("source: squarenet.sampler from module squarenet")
        print("available datasets:", list_methods())
    points = samplepoints(method = dataset, size = size, plot_points = plot_data)
    points -= points.min()
    points /= points.max()
    return points