from som import *
from utils import Checkpoint
from drax import Drax

def train(config, colors, verbose):
    """
    Trains the SOM.
    'input_vects' should be an iterable of 1-D NumPy arrays with
    dimensionality as provided during initialization of this SOM.
    Current weightage vectors for all neurons(initially random) are
    taken as starting conditions for training.
    """
    print("Train the vanilla SOM module ...")
    config['verbose'] = verbose

    som = SOM(**config)

    _, pre_state = nnx.split(som)
    outputs = som(colors, True, verbose)
    _, post_state = nnx.split(som)

    ckpt = Checkpoint("som-ckpt", True)
    ckpt.save(pre_state, post_state)

    print(outputs)


def test(config, colors, verbose):
    print("Test the trained SOM module ...")
    config['verbose'] = verbose

    som = SOM(**config)
    ckpt = Checkpoint("som-ckpt", False)
    ckpt.load(som)

    som.trained = True

    for c in colors:
        loc, color = som.bmu(c)
        print("color {} <-> center {}".format(c, loc))
        print("center value -> {}".format(color))


def image(config):
    print("Extract the weights and to save as image ...")

    som = SOM(**config)
    ckpt = Checkpoint("som-ckpt", False)
    ckpt.load(som)

    som.trained = True
    drax = Drax("som-cont.png")
    drax([{"data": som.image, "title": ""}])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="som.py 1.0")

    def str2bool(v):
        if isinstance(v, bool):
            return v
        return v.lower() in ('yes', 'true', 't', '1')

    parser.add_argument("--verbose", nargs='?', type=str2bool, choices=[True, False], default=False, const=True, help="output info during execution")
    parser.add_argument("--train", nargs='?', type=str2bool, choices=[True, False], default=False, const=True, help="train the SOM model")
    parser.add_argument("--test", nargs='?', type=str2bool, choices=[True, False], default=False, const=True, help="test the SOM model")
    parser.add_argument("--image", nargs='?', type=str2bool, choices=[True, False], default=False, const=True, help="output the trained weights as image")

    args = parser.parse_args()

    config = {
        "height": 256,
        "width": 256,
        "input_size": 3,
        "num_iters": 1000,
        "learning_rate": 0.1
    }

    colors = jnp.array([
        [0., 0., 0.5],
        [0.33, 0.4, 0.67],
        [1., 0., 0.],
        [0.9, 0.2, 1.],
        [.66, .56, .66]
    ])

    if args.train:
        train(config, colors, args.verbose)

    if args.test:
        test(config, colors, args.verbose)

    if args.image:
        image(config)

