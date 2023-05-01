import time
import math
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import random


cuda_available = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda_available else "cpu")
if cuda_available:
    print(f"CUDA is available, device: {torch.cuda.get_device_name(device)}")

# Training data is 5 points in [0,1] inclusive regularly spaced
train_x = torch.linspace(0, 1, 5, device=device)

# Test points are regularly spaced along [0,1]
test_x = torch.linspace(0, 1, 51, device=device)

# True function is sin(2*pi*x) with Gaussian noise
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn_like(train_x) * math.sqrt(0.04)
objective = lambda x: math.sin(x * (2 * math.pi)) + random.random() * math.sqrt(0.04)


# initialize likelihood and model
gp = SingleTaskGP(train_x, train_y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

# Initialize plot
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.set_ylim([-3, 3])
ax.legend(['Observed Data', 'Mean', 'Confidence'])

def animate(i: int):
    """ Function that draws each frame of the animation """
    global train_x, train_y

    # Do one evaluation
    candidate_param = random.random()
    train_x = torch.cat((train_x, torch.Tensor([candidate_param]).cuda()))
    train_y = torch.cat((train_y, torch.Tensor([objective(candidate_param)]).cuda()))
    print(f"Frame {i}, candidate: {candidate_param}")

    # Update the model
    gp.set_train_data(train_x, train_y, strict=False)
    # gp.get_fantasy_model(train_x, train_y)

    # Make predictions by feeding model through likelihood
    observed_pred = gp.likelihood(gp(test_x))
    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()

    # Update the animation
    ax.clear()
    # Plot training data as black stars
    ax.plot(train_x.cpu().numpy(), train_y.cpu().numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x.cpu().numpy(), observed_pred.mean.cpu().numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5)

timer_start = time.perf_counter()
ani = FuncAnimation(fig, animate, frames=1000, interval=0, repeat=False)
timer_end = time.perf_counter()
print(f"Time: { (timer_end - timer_start) / 1000 } seconds")
plt.show()
