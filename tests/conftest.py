import sys
import os
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), 'fixtures'))

from .fixtures.cross_validation_fixtures import *
from .fixtures.interactions_fixtures import *
from .fixtures.loss_fixtures import *
from .fixtures.metrics_fixtures import *
from .fixtures.model_fixtures import *
from .fixtures.movielens_fixtures import *
from .fixtures.utils_fixtures import *


@pytest.fixture(scope='session', params=['cpu', 'cuda:0'] if torch.cuda.is_available() else ['cpu'])
def device(request):
    return request.param


@pytest.fixture(scope='session')
def gpu_count(device):
    if device == 'cpu':
        return 0
    else:
        return 1
