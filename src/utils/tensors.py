from torch import cuda, device as dev

use_cuda = cuda.is_available()
device = dev('cuda' if use_cuda else 'cpu')
