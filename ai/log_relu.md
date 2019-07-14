# LogReLU: A New Activation Function Inspired By ReLU

The output of ReLU may be too large which makes deep learning networks unstable, in this blog, we introduce a new kind of activation function called LogReLU
<br>
LogReLU can be simply defined as log(max(x, 0) + 1)
<br>
Following is a PyTorch implementation of LogReLU
<pre>
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def log_relu(x):
    return torch.log(F.relu(x) + 1)


x = torch.linspace(-2, 4, 10000)

y1 = F.relu(x).numpy()
y2 = log_relu(x).numpy()
x = x.numpy()

fig1, = plt.plot(x, y1)
fig2, = plt.plot(x, y2)

plt.legend(handles=[fig1, fig2], labels=['relu', 'log_relu'], loc='upper left')
plt.show()
</pre>
