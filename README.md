# RIAI 2020 Course Project



## Folder structure
In the directory `code` you can find 2 files. 
File `networks.py` contains encoding of fully connected and convolutional neural network architectures as PyTorch classes.
The architectures extend `nn.Module` object and consist of standard PyTorch layers (`Linear`, `Flatten`, `ReLU`, `Conv2d`). Please note that first layer of each network performs normalization of the input image.
File `verifier.py` contains a template of verifier. Loading of the stored networks and test cases is already implemented in `main` function. If you decide to modify `main` function, please ensure that parsing of the test cases works correctly. Your task is to modify `analyze` function by building upon DeepPoly convex relaxation. Note that provided verifier template is guaranteed to achieve **0** points (by always outputting `not verified`).

In folder `mnist_nets` you can find 10 neural networks (7 fully connected and 3 convolutional). These networks are loaded using PyTorch in `verifier.py`.
In folder `test_cases` you can find 10 subfolders. Each subfolder is associated with one of the networks, using the same name. In a subfolder corresponding to a network, you can find 2 test cases for this network. 
As explained in the lecture, these test cases **are not** part of the set of test cases which we will use for the final evaluation. 

## Setup instructions

We recommend you to install Python virtual environment to ensure dependencies are same as the ones we will use for evaluation.
To evaluate your solution, we are going to use Python 3.7.
You can create virtual environment and install the dependencies using the following commands:

```bash
$ virtualenv venv --python=python3.7
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Running the verifier

We will run your verifier from `code` directory using the command:

```bash
$ python verifier.py --net {net} --spec ../test_cases/{net}/img{test_idx}_{eps}.txt
```

In this command, `{net}` is equal to one of the following values (each representing one of the networks we want to verify): `fc1, fc2, fc3, fc4, fc5, fc6, fc7, conv1, conv2, conv3`.
`test_idx` is an integer representing index of the test case, while `eps` is perturbation that verifier should certify in this test case.

To test your verifier, you can run for example:

```bash
$ python verifier.py --net fc1 --spec ../test_cases/fc1/img0_0.06200.txt
```

To evaluate the verifier on all networks and sample test cases, we provide the evaluation script.
You can run this script using the following commands:

```bash
chmod +x evaluate
./evaluate
```
