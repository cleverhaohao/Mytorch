"""
此次作业借鉴和参考了Needle项目 https://github.com/dlsyscourse/lecture5
本文件我们给出一个基本完善的Tensor类，但是缺少梯度计算的功能
你需要把梯度计算所需要的运算的正反向计算补充完整
一共有12*2处
当你填写好之后，可以调用test_task1_*****.py中的函数进行测试
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from device import cpu, Device
from basic_operator import Op, Value
from task2_autodiff import compute_gradient_of_variables
import math
from device import cpu
from mytensor import Tensor as hqhtensor
from mytensor import fc_forward, fc_backward, Softmax, CrossEntropyLoss, CrossEntropyLoss_backward_with_Softmax, conv_forward, conv_backward, maxpooling_forward, maxpooling_backward

def constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False):
    """Generate constant Tensor"""
    device = cpu() if device is None else device
    array = device.ones(*shape, dtype=dtype) * c  # note: can change dtype
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def ones(*shape, device=None, dtype="float32", requires_grad=False):
    """Generate all-ones Tensor"""
    return constant(
        *shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad
    )

class Tensor(Value):
    grad: "Tensor"

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                cached_data = Tensor._hqhtensor_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else cpu()
            cached_data = Tensor._hqhtensor_from_numpy(array, device=device, dtype=dtype)

        self.grad = None
        
        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _hqhtensor_from_numpy(numpy_array, device, dtype):
        return hqhtensor(np.array(numpy_array))

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        return np.array(numpy_array, dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not tensor.requires_grad:
            return tensor.detach()
        tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        if isinstance(data, np.ndarray):
            data = Tensor._hqhtensor_from_numpy(data, device=cpu(), dtype=data.dtype)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        # assert value.dtype == self.dtype, "%s %s" % (
        #     value.dtype,
        #     self.dtype,
        # )
        self.cached_data = value.realize_cached_data()

    def detach(self):
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self):
        return cpu()


    # def backward(self, out_grad=None):
    #     out_grad = (
    #                 out_grad
    #                 if out_grad
    #                 else ones(*self.shape, dtype=self.dtype, device=self.device)
    #             )
    #     compute_gradient_of_variables(self, out_grad)
    
    def backward(self, out_grad=None):
        out_grad = (
                    out_grad
                    if out_grad
                    else ones(*self.shape)
                )
        compute_gradient_of_variables(self, out_grad)

    def __repr__(self):
        return "Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self):
        data = self.realize_cached_data()

        return data


    def __add__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, other)
        else:
            return AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return EWiseMul()(self, other)
        else:
            return MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return EWisePow()(self, other)
        else:
            return PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, Negate()(other))
        else:
            return AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return EWiseDiv()(self, other)
        else:
            return DivScalar(other)(self)

    def __matmul__(self, other):
        return MatMul()(self, other)

    def matmul(self, other):
        return MatMul()(self, other)

    def sum(self, axes=None):
        return Summation(axes)(self)

    def broadcast_to(self, shape):
        return BroadcastTo(shape)(self)

    def reshape(self, shape):
        return Reshape(shape)(self)

    def __neg__(self):
        return Negate()(self)

    def transpose(self, axes=None):
        return Transpose(axes)(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__

class TensorOp(Op):
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)


class ReLU(TensorOp):
    def compute(self, a):
        a_data = a.data()
        return hqhtensor(np.maximum(0, a_data))
        

    def gradient(self, out_grad, node):
        out = node.realize_cached_data()
        out_data = out.data().copy()
        out_data[out_data > 0] = 1
        return out_grad * Tensor(out_data)
        
def relu(a):
    return ReLU()(a)


class FullyConnected(TensorOp):
    def compute(self, a, weight):
        return hqhtensor(fc_forward(a, weight))
    
    def gradient(self, out_grad, node):
        x_grad, weight_grad= fc_backward( node.inputs[0].realize_cached_data(), 
                                          node.inputs[1].realize_cached_data(), 
                                          out_grad.realize_cached_data())
        return Tensor(x_grad), Tensor(weight_grad)

def fullyconnected(a, weight):
    return FullyConnected()(a, weight)


class Convolution(TensorOp):
    def compute(self, a, weight):
        return hqhtensor(conv_forward(a, weight))
    def gradient(self, out_grad, node):
        x_grad, weight_grad= conv_backward( node.inputs[0].realize_cached_data(), 
                                          node.inputs[1].realize_cached_data(), 
                                          out_grad.realize_cached_data())
        return Tensor(x_grad), Tensor(weight_grad)
    
def convolution(a, weight):
    return Convolution()(a, weight)

class MaxPooling(TensorOp):
    def compute(self, a):
        result, mask = maxpooling_forward(a)
        self.mask = hqhtensor(mask)
        return hqhtensor(result)
    def gradient(self, out_grad, node):
        x_grad = maxpooling_backward(node.inputs[0].realize_cached_data(), out_grad.realize_cached_data(), self.mask)
        return Tensor(x_grad)

def maxpooling(a):
    return MaxPooling()(a)

class Softmax_and_CrossEntropy(TensorOp):
    def __init__(self, label):
        self.label = label.realize_cached_data()
    def compute(self, a):
        numpy_after_softmax = Softmax(a)
        hqhtensor_after_softmax = hqhtensor(numpy_after_softmax)
        return hqhtensor(CrossEntropyLoss(hqhtensor_after_softmax, self.label))
        #return hqhtensor(np.array(1.5))
    def gradient(self, out_grad, node):
        x_grad = CrossEntropyLoss_backward_with_Softmax(node.inputs[0].realize_cached_data(), self.label)
        return Tensor(x_grad)

def softmax_and_cross_entropy(a, label):
    return Softmax_and_CrossEntropy(label)(a)




class EWiseAdd(TensorOp):
    def compute(self, a: hqhtensor, b: hqhtensor):
        return hqhtensor(a.data() + b.data())   

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: hqhtensor):
        return hqhtensor(a.data() + self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: hqhtensor, b: hqhtensor):
        return hqhtensor(a.data() * b.data())

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: hqhtensor):
        return hqhtensor(a.data() * self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """逐点乘方，用标量做指数"""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: hqhtensor) -> hqhtensor:
        
        return hqhtensor(a.data() ** self.scalar)
        
        

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], Tensor):
            raise ValueError("The input must be a tensor.")
        return out_grad * self.scalar * (node.inputs[0] ** (self.scalar - 1))


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """逐点乘方"""

    def compute(self, a: hqhtensor, b: hqhtensor) -> hqhtensor:
        return hqhtensor(a.data() ** b.data())

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], Tensor) or not isinstance(
            node.inputs[1], Tensor
        ):
            raise ValueError("Both inputs must be tensors.")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """逐点相除"""

    def compute(self, a, b):
        return hqhtensor(a.data() / b.data())
        

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], Tensor) or not isinstance(
            node.inputs[1], Tensor
        ):
            raise ValueError("Both inputs must be tensors.")
        a, b = node.inputs
        return out_grad / b, -out_grad * a / (b ** 2)
        


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return hqhtensor(a.data() / self.scalar)
        

    def gradient(self, out_grad, node):
        return out_grad / self.scalar
        


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        a_data = a.data()
        if self.axes is None:
            return hqhtensor(np.swapaxes(a_data, -1, -2))
        else:
            dim1 = self.axes[0]
            dim2 = self.axes[1]

            return hqhtensor(np.swapaxes(a_data, dim1, dim2))
        

    def gradient(self, out_grad, node):
        
        return transpose(out_grad, axes=self.axes)
        


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        a_data = a.data()
        return hqhtensor(np.reshape(a_data, self.shape))
        

    def gradient(self, out_grad, node):
        return reshape(out_grad, node.inputs[0].shape)
        


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        a_data = a.data()
        return hqhtensor(np.broadcast_to(a_data, self.shape))
        

    def gradient(self, out_grad, node):
        original_shape = node.inputs[0].shape
        changed_dims = [i for i in range(len(self.shape))]
        for i, (original, current) in enumerate(zip(reversed(original_shape), reversed(self.shape))):
            if original == current:
                changed_dims[-i-1] = -1
        changed_dims = tuple(filter(lambda x: x >= 0, changed_dims))
        return summation(out_grad,changed_dims).reshape(original_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        a_data = a.data()
        return hqhtensor(np.sum(a_data, axis=self.axes))
        

    def gradient(self, out_grad, node):
        original_shape = list(node.inputs[0].shape)
        axes = range(len(original_shape)) if self.axes is None else self.axes
        for ax in axes:
            original_shape[ax] = 1
        return broadcast_to(out_grad.reshape(original_shape), node.inputs[0].shape)

        
        


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        a_data = a.data()
        b_data = b.data()
        return hqhtensor(np.matmul(a_data, b_data))
        

    def gradient(self, out_grad, node):
        
    
        a, b = node.inputs

        aGrad, bGrad = matmul(out_grad, transpose(b)), matmul(transpose(a), out_grad)
        if len(a.shape) < len(aGrad.shape):
            aGrad = aGrad.sum(tuple([i for i in range(len(aGrad.shape) - len(a.shape))]))


        if len(b.shape) < len(bGrad.shape):
            bGrad = bGrad.sum(tuple([i for i in range(len(bGrad.shape) - len(b.shape))]))

        return aGrad, bGrad
        


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        a_data = a.data()
        return hqhtensor(-a_data)
        

    def gradient(self, out_grad, node):
        return -out_grad
        


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        a_data = a.data()
        return hqhtensor(np.log(a_data))
        

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]
        


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        a_data = a.data()
        return hqhtensor(np.exp(a_data))
        

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])
        


def exp(a):
    return Exp()(a)




