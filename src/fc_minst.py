import time
import numpy as np
import torch
import torchvision
from task1_operators import *
from optimizer import SGD
from mytensor import Tensor as hqhtensor
def parse_mnist():
    """
    读取MNIST数据集，并进行简单的处理，如归一化
    你可以可以引入任何的库来帮你进行数据处理和读取
    所以不会规定你的输入的格式
    但需要使得输出包括X_tr, y_tr和X_te, y_te
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),  # 将图像转换为张量
        torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 归一化处理
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # 将数据加载到一个批次中
    X_tr, y_tr = next(iter(train_loader))
    X_te, y_te = next(iter(test_loader))

    X_tr = X_tr.numpy()
    y_tr = y_tr.numpy()
    X_te = X_te.numpy()
    y_te = y_te.numpy()

    X_tr = X_tr.reshape(X_tr.shape[0], -1)
    X_te = X_te.reshape(X_te.shape[0], -1)

    return X_tr, y_tr, X_te, y_te
    

def set_structure(n, hidden_dim, k):
    """
    定义你的网络结构，并进行简单的初始化
    一个简单的网络结构为两个Linear层，中间加上ReLU
    Args:
        n: input dimension of the data.
        hidden_dim: hidden dimension of the network.
        k: output dimension of the network, which is the number of classes.
    Returns:
        List of Weights matrix.
    Example:
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)
    return list(W1, W2)
    """

    W1 = Tensor(np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim))
    
    W2 = Tensor(np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k))
    
    return [W1, W2]

def forward(X, weights):
    """
    使用你的网络结构，来计算给定输入X的输出
    Args:
        X : 2D input array of size (num_examples, input_dim).
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
    Returns:
        Logits calculated by your network structure.
    Example:
    W1 = weights[0]
    W2 = weights[1]
    return np.maximum(X@W1,0)@W2
    """
    W1 = weights[0]
    W2 = weights[1]
    
    # 第一层线性变换
    X = fullyconnected(X, W1)
    
    # 应用 ReLU 激活函数
    X = relu(X)
    
    # 第二层线性变换
    logits = fullyconnected(X, W2)

    return logits
    

def softmax_loss(Z:Tensor, y:Tensor):
    """ 
    一个写了很多遍的Softmax loss...

    Args:
        Z : 2D numpy array of shape (batch_size, num_classes), 
        containing the logit predictions for each class.
        y : 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    # 获取批次大小和类别数
    batch_size, num_classes = Z.shape
    
    # # 计算 Softmax 概率
    # exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # 防止数值不稳定
    # softmax_probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    # # 计算交叉熵损失
    # correct_class_probs = softmax_probs[np.arange(batch_size), y]
    # loss = -np.mean(np.log(correct_class_probs+1e-5))
    loss = softmax_and_cross_entropy(Z, y)
    
    return loss


def opti_epoch(X, y, weights, batch=100, optimizer=None):
    """
    优化一个epoch
    具体请参考SGD_epoch 和 Adam_epoch的代码
    """
    # if using_adam:
    #     Adam_epoch(X, y, weights, lr = lr, batch=batch, beta1=beta1, beta2=beta2)
    # else:
    #     SGD_epoch(X, y, weights, lr = lr, batch=batch)
    total_data = X.shape[0]
    for start_idx in range(0, total_data, batch):
        time1 = time.time()
        end_idx = min(start_idx + batch, total_data)
        X_batch = Tensor.make_const(X[start_idx:end_idx])
        y_batch = Tensor.make_const(y[start_idx:end_idx])
        time2 = time.time()
        optimizer.zero_grad()
        logits = forward(X_batch, weights)
        loss = softmax_loss(logits, y_batch)
        time3 = time.time()
        loss.backward()
        time4 = time.time()
        #optimizer.step()
       
        for parameter in weights:
            parameter.data = parameter.data - optimizer.lr * parameter.grad
        time5 = time.time()
        # print("time2-time1:", time2-time1)
        # print("time3-time2:", time3-time2)
        # print("time4-time3:", time4-time3)
        # print("time5-time4:", time5-time4)

        

            
    
            


def loss_err(h,y):
    """ 
    计算给定预测结果h和真实标签y的loss和error
    """
    h_numpy = h.realize_cached_data().data()
    y_numpy = y.realize_cached_data().data()
    return softmax_loss(h,y).realize_cached_data().data(), np.mean(h_numpy.argmax(axis=1) != y_numpy)

def accuracy(h, y):
    """ 
    计算给定预测结果h和真实标签y的准确率
    """
    h_numpy = h.realize_cached_data().data()
    y_numpy = y.realize_cached_data().data()
    return np.mean(h_numpy.argmax(axis=1) == y_numpy)


def train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100, beta1=0.9, beta2=0.999, using_adam=False):
    """ 
    训练过程
    """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(9)
    weights = set_structure(n, hidden_dim, k)
    if(using_adam):
        pass
    else:
        optimizer = SGD(weights, lr=lr)
    

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err | Time (s) |")
    for epoch in range(epochs):
        start_time = time.time()  # 记录开始时间
        opti_epoch(X_tr, y_tr, weights, batch=batch, optimizer=optimizer)
        train_loss, train_err = loss_err(forward(Tensor.make_const(X_tr), weights), Tensor.make_const(y_tr))
        test_loss, test_err = loss_err(forward(Tensor.make_const(X_te), weights), Tensor.make_const(y_te))

        train_loss = train_loss.item() if hasattr(train_loss, 'item') else train_loss
        train_err = train_err.item() if hasattr(train_err, 'item') else train_err
        test_loss = test_loss.item() if hasattr(test_loss, 'item') else test_loss
        test_err = test_err.item() if hasattr(test_err, 'item') else test_err

        epoch_time = time.time() - start_time  # 计算 epoch 耗时

        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |  {:.2f}  |"\
              .format(epoch, train_loss, train_err, test_loss, test_err, epoch_time))
    print("Final test accuracy: {:.4f}".format(accuracy(forward(Tensor.make_const(X_te), weights), Tensor.make_const(y_te))))



if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = parse_mnist()
    
    weights = set_structure(X_tr.shape[1], 100, y_tr.max() + 1)

    ## using SGD optimizer 
    train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim=100, epochs=10, lr = 0.05, batch=100)