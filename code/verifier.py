import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import networks
from networks import FullyConnected, Conv
from transformers import DPLinear, DPReLU, DPNormalization, Validator, DPPrintLayer, DPConv2D
from deep_poly import DeepPoly
import warnings

warnings.filterwarnings("ignore")

DEVICE = 'cpu'
INPUT_SIZE = 28


def build_verifier_network(net, true_label, init_features, verbose=False):
    # obtain ordered list of layer names
    # have to filter for networks.FullyConnected, networks.Conv and nn.Sequential,
    # since .modules() works recursively, apparently
    ordered_layers = [module for module in net.modules()
                      if type(module) not in [networks.FullyConnected, networks.Conv, nn.Sequential]]
    # build a matching verifier model for this:
    verif_layers = []

    for layer in ordered_layers:
        # every layer now also keeps track of the features (needed for relu initalization mostly)
        in_features = verif_layers[-1].out_features if len(verif_layers) > 0 else init_features
        if verbose:
            verif_layers.append(DPPrintLayer(in_features))
        if type(layer) == networks.Normalization:
            # commented out for now, looks like we don't need a layer for that
            # verif_layers.append(DPNormalization(in_features))
            pass
        elif type(layer) == nn.Flatten:
            # alex: right now i don't think we need it, as we plan to treat Conv as flattened transformers anyway
            # but might be that we have to think about it some more
            # (also batch dimension, but that is another discussion)
            pass
        elif type(layer) == nn.Linear:
            verif_layers.append(DPLinear(layer))
        elif type(layer) == nn.ReLU:
            verif_layers.append(DPReLU(in_features))
        elif type(layer) == nn.Conv2d:
            verif_layers.append(DPConv2D(layer, in_features=in_features))
        else:
            print("Layer type not implemented: ", str(type(layer)))
            quit(-1)

    if verbose:
        verif_layers.append(DPPrintLayer(verif_layers[-1].out_features))

    verif_layers.append(Validator(verif_layers[-1].out_features, true_label=true_label, verbose=verbose))

    if verbose:
        print("List of layers in the loaded network: ", ordered_layers)
        print("List of layers in the DP verifier: ", verif_layers)

    return nn.Sequential(*verif_layers)


def loss_fn(net, verify_result, xlb, xub, true_label):
    # max makes more sense than sum, log is for stability
    return torch.log(-verify_result[verify_result < 0]).max()


def apply_lambda_heuristic(net, true_label):
    # TODO maybe find a static heuristic based on the net structure and weights? Can change the lambdas accordingly
    return net


def analyze(net, inputs, eps, true_label, VERBOSE=False):
    pixel_values = inputs.view(-1)
    mean = 0.1307
    sigma = 0.3081

    verif_net = build_verifier_network(net, true_label, len(pixel_values), verbose=False)
    verif_net = apply_lambda_heuristic(verif_net, true_label)

    lb = (pixel_values - eps).clamp(0, 1)
    ub = (pixel_values + eps).clamp(0, 1)
    lb = (lb - mean) / sigma
    ub = (ub - mean) / sigma

    opt = optim.Adam(verif_net.parameters(), lr=0.1)
    num_iter = 1000
    for i in range(num_iter):
        opt.zero_grad()
        # need to create new DeepPoly each iter
        x = DeepPoly(lb.shape[0], lb, ub)

        verify_result, xlb, xub = verif_net(x)

        if (verify_result > 0).all():
            if VERBOSE:
                print("Success after the ", i, "th iteration !")
                print("lambdas that worked for verification:")
                for p in verif_net.parameters():
                    print("\t", p)
            # if we ever verify, we know we are done (because we are sound)
            return True
        if i == num_iter - 1:
            return False

        loss = loss_fn(verif_net, verify_result, xlb, xub, true_label)

        if VERBOSE and i % (num_iter / 10) == 0:
            print("... iteration", i)
            print("loss:", loss)
            print("current result:", verify_result)
            print("lambdas:")
            for p in verif_net.parameters():
                print("\t", p)

        loss.backward()
        opt.step()
        with torch.no_grad():
            for param in verif_net.parameters():
                # maybe find a better way to do this
                # random restart, other things can go in here as well
                # if i == int(num_iter / 2):
                #     l = param.data.shape[0]
                #     param.data = torch.rand(l)
                param.data = param.data.clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepZ relaxation')
    parser.add_argument('--net',
                        type=str,
                        choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7', 'conv1', 'conv2', 'conv3'],
                        required=True,
                        help='Neural network architecture which is supposed to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    parser.add_argument('--verbose', '-v', help='Use verbose output for debugging.', action='store_true')
    args = parser.parse_args()
    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    if args.net == 'fc1':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif args.net == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif args.net == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif args.net == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc6':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc7':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    else:
        assert False

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label, args.verbose):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
