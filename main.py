from mlxtend.data import loadlocal_mnist
import torch

X, y = loadlocal_mnist(images_path = '/home/ahoque245/projects/AI/handwritten_digits/data/train/train-images-idx3-ubyte',
 labels_path='/home/ahoque245/projects/AI/handwritten_digits/data/train/train-labels-idx1-ubyte')

X = torch.from_numpy(X).cuda()
y = torch.from_numpy(y).cuda()



print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
print('\n1st row', X[0])
