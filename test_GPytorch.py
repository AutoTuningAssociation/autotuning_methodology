from copy import deepcopy
import time
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import random

cuda_available = torch.cuda.is_available()
cuda_available = False
device = torch.device("cuda:0" if cuda_available else "cpu")
if cuda_available:
    print(f"CUDA is available, device: {torch.cuda.get_device_name(device)}")


observation_noise = 1e-3

# Objective function is sin(2*pi*x) with Gaussian noise
# objective = lambda x: math.sin(x * (2 * math.pi)) + random.normalvariate(0, 0.1) * math.sqrt(0.04)
# rmin, rmax = 0, 1

# Objective function is the multimodal sin(x) + sin(10 / 3 * x)
objective = lambda x: math.sin(x) + math.sin((10.0 / 3.0) * x)
rmin, rmax = -2.7, 7.5


# Training data is 5 points in [0,1] inclusive regularly spaced
train_x = torch.linspace(rmin, rmax, 10)
train_y = train_x.detach().clone().apply_(objective)
train_y_err = torch.ones_like(train_x) * observation_noise

# Test points are regularly spaced along [0,1]
test_x = torch.linspace(rmin, rmax, 500)

# # True function is sin(2*pi*x) with Gaussian noise
# train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn_like(train_x) * math.sqrt(0.04)
# objective = lambda x: math.sin(x * (2 * math.pi)) + random.random() * math.sqrt(0.04)

# mean normalization of inputs
mean_x, std_x = train_x.mean(), train_x.std()
train_x = (train_x - mean_x) / std_x
test_x = (test_x - mean_x) / std_x
scale_input = lambda x: (x - mean_x) / std_x
unscale_input = lambda x: x * std_x + mean_x

# mean normalization of outputs
mean_y, std_y = train_y.mean(),train_y.std()
train_y = (train_y - mean_y) / std_y
scale_output = lambda x: (x - mean_y) / std_y
unscale_output = lambda x: x * std_y + mean_y


# Put on the GPU if applicable
if cuda_available:
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    test_x = test_x.cuda()


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def instantiate_model(x, y):
    # initialize likelihood and model
    # likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=train_y_err, learn_additional_noise=False)
    # likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=gpytorch.priors.SmoothedBoxPrior(-0.001, 0))
    # likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.zeros_like(train_x)+2, learn_additional_noise=True)
    # print(likelihood)
    model = ExactGPModel(x, y, likelihood)
    # model.covar_module.base_kernel.initialize(lengthscale=1)

    # Put on the GPU if applicable
    if cuda_available:
        likelihood = likelihood.cuda()
        model = model.cuda()


    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)    # Includes GaussianLikelihood parameters
    # optimizer = torch.optim.LBFGS(model.parameters())     # Add closure function to step

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    return model, likelihood, optimizer, mll


model, likelihood, optimizer, mll = instantiate_model(train_x, train_y)


def train_model(training_iter = 50, retries = 0):
    global model, likelihood, optimizer, mll
    model.train()
    likelihood.train()

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        try:
            loss = -mll(output, train_y)
        except gpytorch.utils.errors.NotPSDError:
            if retries > 0:
                raise ValueError("It's not possible to create a non-positive definite model with this data")
            print("Matrix is not positive definite, instantiating a new model")
            model, likelihood, optimizer, mll = instantiate_model(train_x, train_y)
            # train_model(training_iter=1, retries=retries+1)
            break
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' %
        #     (i + 1, training_iter, loss.item(), model.covar_module.base_kernel.lengthscale.item(), model.likelihood.noise.item()))
        optimizer.step()

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

# Train the model
train_model()
# model.eval()
# likelihood.eval()

with gpytorch.settings.fast_pred_var():
    # Initialize plot
    fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    # ax.set_ylim([-3, 3])
    ax.set_ylabel('Value')
    ax.set_xlabel('Parameter')
    ax.legend(['Observed Data', 'Mean', 'Confidence'])

    def animate(i: int):
        """ Function that draws each frame of the animation """
        global model, likelihood, optimizer, mll, train_x, train_y, test_x

        # Do one evaluation
        candidate_param = random.uniform(rmin, rmax)
        train_x = torch.cat((train_x, torch.Tensor([scale_input(candidate_param)]).to(device)))
        train_y = torch.cat((train_y, torch.Tensor([scale_output(objective(candidate_param))]).to(device)))
        print(f"Frame {i}, candidate: {candidate_param}")

        # Remove the evaluation from test_x


        # Update the model
        # TODO get_fantasy_model is unstable, maybe use get_fantasy_model and then have set_train_data every 10 evaluations?
        # if i % 10 == 0:
        #     model.set_train_data(train_x, train_y, strict=False)
        # else:
        #     with torch.no_grad():
        #         model = model.get_fantasy_model(train_x, train_y)
        model.set_train_data(train_x, train_y, strict=False)
        # if i <= 100 and i % 20 == 0:
        #     train_model(training_iter=5)
        # if i % 20 == 0:
        # train_model(training_iter=5)

        # if i % 20 == 0:
        #     model, likelihood, optimizer, mll = instantiate_model(train_x, train_y)
        #     train_model()
        #     print(f"1 in 20 Frame {i}")
        # else:
        #     model.set_train_data(train_x, train_y, strict=False)


        # Make predictions by feeding model through likelihood
        # TODO look into "GPInputWarning: You have passed data through a FixedNoiseGaussianLikelihood that did not match the size of the fixed noise, *and* you did not specify noise. This is treated as a no-op."
        train_y_err = torch.ones_like(train_x) * observation_noise
        likelihood = likelihood.get_fantasy_likelihood(noise=train_y_err)
        observed_pred = likelihood(model(test_x))
        # Get mean, upper and lower confidence bounds
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()

        # Return to the CPU if applicable
        if cuda_available:
            mean = mean.cpu()
            lower = lower.cpu()
            upper = upper.cpu()
            train_x_read = train_x.cpu()
            train_y_read = train_y.cpu()
            test_x_read = test_x.cpu()
        else:
            train_x_read = train_x
            train_y_read = train_y
            test_x_read = test_x

        # Update the animation
        ax.clear()
        # Plot training data as black stars
        ax.plot(train_x_read.numpy(), train_y_read.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x_read.detach().numpy(), mean.detach().numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x_read.detach().numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)

    timer_start = time.perf_counter()
    ani = FuncAnimation(fig, animate, frames=1000, interval=100, repeat=False)
    timer_end = time.perf_counter()
    print(f"Time: { (timer_end - timer_start) / 1000 } seconds")
    plt.show()
