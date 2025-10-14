import torch

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None  # function to propagate gradients
        self._prev = set(_children)  # set of parent Value nodes
        self._op = _op  

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op='+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, _children=(self,), _op='**')

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(max(0, self.data), _children=(self,), _op='ReLU')

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)


        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  
        return self + other

    def __sub__(self, other):  
        return self + (-other)

    def __rsub__(self, other):  
        return other + (-self)

    def __rmul__(self, other):  
        return self * other

    def __truediv__(self, other):  
        return self * (other if isinstance(other, Value) else Value(other)) ** -1

    def __rtruediv__(self, other):  
        return (other if isinstance(other, Value) else Value(other)) * self ** -1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"




def test_sanity_check():
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    assert abs(ymg.data - ypt.data.item()) < 1e-6
    print(xmg, xpt, xpt.grad)
    assert abs(xmg.grad - xpt.grad.item()) < 1e-6



if __name__ == "__main__":
    a = Value(-4.0)
    b = Value(2.0)
    d = Value(3.0)

    c = a + b
    e = c * d
    e.backward()

    test_sanity_check()
