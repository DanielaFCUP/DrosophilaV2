import argparse

import yaml


def load() -> dict[str, int | float | str]:
    parser = argparse.ArgumentParser(
        prog='Mosquinha',
        description='ML Test Bench',
        epilog='Can train and use different neural network models in the classification of Drosophilas.')

    # Define configuration parameters and default arguments
    parser.add_argument('-c', '--conf', type=str, default=None, nargs='?',
                        help='Path to a yaml configuration file. It will override command line arguments!')
    parser.add_argument('-r', '--run', type=str, choices=['full', 'preproc', 'prepare', 'classify'],
                        required=True, help='What should I do?')
    parser.add_argument('--optim', choices=['Adam'], type=str, default='Adam', nargs='?', help='Optimiser to use.')
    parser.add_argument('-m', '--model', choices=['resnet', 'efficientnet', 'densenet'], type=str, default='resnet',
                        nargs='?', help='Model to use.')
    parser.add_argument('--model_pth', type=str, default='out/model.pth', nargs='?', help='Use ready model.')
    parser.add_argument('--model_state_dict', type=str, default='out/model_state_dict.txt', nargs='?',
                        help='Use ready state dictionary.')
    parser.add_argument('--epochs', type=int, default=50, nargs='?', help='Number of epochs.')
    parser.add_argument('--batch', type=int, default=16, nargs='?', help='Batch size.')
    parser.add_argument('--lr', type=float, default=5e-4, nargs='?',
                        help='Learning Rate for the Optimiser.')
    parser.add_argument('--raw', type=str, default='in/', nargs='?', help='Path to the raw images directory.')
    parser.add_argument('-i', '--image', type=str, default=None, nargs='?', help='Path to the input image.')
    # Important: classes must be sorted
    parser.add_argument('--class-names', type=str, nargs='*', default=['class 1', 'class 2', 'class 3'],
                        help='Classes for Classification (sorted). Important: classes must be sorted.')
    parser.add_argument('--preproc', type=str,
                        choices=['skip', 'remove_background', 'bilateral', 'gaussian', 'median', 'unsharp'],
                        default='skip', nargs='?',
                        help='Type of the preprocessing for the input images.')

    # Load Command Line arguments
    args = vars(parser.parse_args())

    # IMPORTANT: YAML overrides any Command Line arguments and isn't validated.
    if args['conf'] is not None:
        args.update(yaml.load(open(args['conf']), Loader=yaml.FullLoader))

    # print("arguments: {}".format(str(args)))

    return args
