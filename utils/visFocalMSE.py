import torch
import matplotlib.pyplot as plt

def calculate_loss(inputs, targets, activate='tanh', beta=.2, gamma=1, exclude_zeros=False):
    loss = torch.nn.functional.mse_loss(inputs, targets)
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    
    maskweight = (targets == 0) * (1 - 0.95) + (targets != 0) * 0.95
    loss *= maskweight
    if exclude_zeros:
        n_pixels = torch.count_nonzero(targets)
        return torch.sum(loss) / n_pixels
    else:
        return torch.mean(loss)

# Generate input and target tensors
inputs = torch.linspace(-1, 1, 100)
targets = torch.linspace(-1, 1, 100)

# Calculate the loss for each input-target pair
loss_values_1 = []
loss_values_2 = []
loss_values_3 = []
loss_values_4 = []
loss_values_5 = []
loss_values_6 = []
loss_values_7 = []
loss_values_8 = []
for i in range(len(inputs)):

    loss_values_1.append(calculate_loss(inputs[i], targets[-i],beta=.1, gamma=1).item())
    loss_values_2.append(calculate_loss(inputs[i], targets[-i],beta=.2, gamma=1).item())
    loss_values_3.append(calculate_loss(inputs[i], targets[-i],beta=.3, gamma=1).item())
    loss_values_4.append(calculate_loss(inputs[i], targets[-i],beta=.4, gamma=1).item())
    loss_values_5.append(calculate_loss(inputs[i], targets[-i],beta=.5, gamma=1).item())
    loss_values_6.append(calculate_loss(inputs[i], targets[-i],beta=.6, gamma=1).item())
    loss_values_7.append(calculate_loss(inputs[i], targets[-i],beta=.7, gamma=1).item())
    loss_values_8.append(calculate_loss(inputs[i], targets[-i],beta=.8, gamma=1).item())

# Plot the loss function curve

plt.plot(inputs, loss_values_1, label='1')
plt.plot(inputs, loss_values_2, label='2')
plt.plot(inputs, loss_values_3, label='3')
plt.plot(inputs, loss_values_4, label='4')
plt.plot(inputs, loss_values_5, label='5')
plt.plot(inputs, loss_values_6, label='6')
plt.plot(inputs, loss_values_7, label='7')

plt.xlabel('Inputs')
plt.ylabel('Loss')
plt.title('Loss Function Curve')
plt.legend()
plt.savefig('./utils/Wfocalloss.png', dpi=300, pad_inches=0.05, bbox_inches='tight')
plt.close()
