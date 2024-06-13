import torch
import torch.nn as nn
import torch.utils
from numpy import ceil
import torch.nn.functional as F

class AFIR1(nn.Module):
    def __init__(self, M, D):
        super(AFIR, self).__init__()
        self.real = nn.Conv1d(1, 1, M, padding=int((M-1)/2), bias=False).double()
        self.imag = nn.Conv1d(1, 1, M, padding=int((M-1)/2), bias=False).double()
        self.real.weight.data.fill_(0.0)
        self.imag.weight.data.fill_(0.0)
        self.real.weight.data[0, 0, int((M-1)/2)+D] = 1.0
    def forward(self, x):
        r1 = self.real(x[:,0].view(1,1,-1))
        r2 = self.imag(x[:,1].view(1,1,-1))
        i1 = self.real(x[:,1].view(1,1,-1))
        i2 = self.imag(x[:,0].view(1,1,-1))
        return torch.cat((r1-r2, i1+i2), dim=1)



class Delay(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = int(M)
        if self.M < 0:
            self.delay = nn.ConstantPad1d(padding=(0, -self.M), value=0)
        else:
            self.delay = nn.ConstantPad1d(padding=(self.M, 0), value=0)

    def forward(self, x):
        if self.M < 0:
            return self.delay(x)[:, :, -self.M:]
        else:
            return self.delay(x)[:, :, :x.shape[-1]]
class AFIR(nn.Module):
    def __init__(self, M, D, complex_coef=True, trainable=True, weights=None):
        super().__init__()
        coef_dtype = torch.complex128 if complex_coef else torch.float64
        self.afir = nn.Conv1d(in_channels=1,
                              out_channels=1,
                              kernel_size=M,
                              padding=int((M - 1) / 2 + D),
                              bias=False,
                              dtype=coef_dtype)
        if weights is None:
            self.afir.weight.data.fill_(0.0)
            self.afir.weight.data[0, 0, int (ceil((M - 1) / 2))] = 1.0
            self.f = 1 if (M % 2 == 0) else 0
            self.f = int(self.f)
        else:
            self.afir.weight.data = torch.tensor(weights, dtype=torch.complex128)[None, None, :]
            self.f = 0

        if not trainable:
            self.afir.weight.requires_grad = False

    def forward(self, x):
        return self.afir(F.pad(x, (self.f, 0)))


class Gain(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.gain = g

    def forward(self, x):
        return x * self.gain

class Prod_cmp(nn.Module):
    def __init__(self):
        super(Prod_cmp, self).__init__()
    def forward(self, inp1, inp2):
        r1 = inp1[:,0].view(1,1,-1)*inp2[:,0].view(1,1,-1)
        r2 = inp1[:,1].view(1,1,-1)*inp2[:,1].view(1,1,-1)
        i1 = inp1[:,1].view(1,1,-1)*inp2[:,0].view(1,1,-1)
        i2 = inp1[:,0].view(1,1,-1)*inp2[:,1].view(1,1,-1)
        return torch.cat((r1-r2, i1+i2), dim=1)



class ABS(nn.Module):  # |x|
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x)  # abs is applied along all tensor dims

class Term(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.d_2 = int(t.so)
        self.si_e = 0 if t.si[:2] == 'xE' else 1
        self.si_o = 0 if t.si[2:4] == 'xE' else 1
        self.phi = 0 if t.si[4:] == '' else torch.tensor(np.complex128(t.si[4:]), dtype=torch.complex64)
        self.gain = Gain(torch.tensor(t.gain, dtype=torch.float64))
        self.o_branch = nn.Sequential()
        self.abs = ABS()

        if self.d_2 and self.phi:
            if abs(self.phi) == 4:
                self.phi = self.phi / 4
            self.o_branch = nn.Sequential(
                Delay(self.d_2),
                Gain(self.phi)
            )
        elif self.d_2:
            self.o_branch = nn.Sequential(
                Delay(self.d_2)
            )
        elif self.phi:
            if abs(self.phi) == 4:
                self.phi = self.phi / 4
            self.o_branch = nn.Sequential(
                Gain(self.phi)
            )

    def forward(self, x):
        return self.abs(self.gain(x[:, self.si_e, :].unsqueeze(1) + self.o_branch(x[:, self.si_o, :].unsqueeze(1))))

class Polynomial(nn.Module):
    def __init__(self, Poly_order,passthrough=False):
        super(Polynomial, self).__init__()
        self.order = Poly_order
#         self.powers=[range(Poly_orders)]
        self.fc=nn.Linear(self.order,1)
        self.fc.weight.data=torch.zeros((1,Poly_order), dtype=torch.complex128)
        if passthrough:
            self.fc.weight.data[1,0] = 1
#         else:
#             torch.linspace(0,1,Poly_order,out=self.weights[0,:],device=device,requires_grad=True)
    def forward(self, x):
        x=x.unsqueeze(-1)

        
        return self.fc(torch.cat([x**i for i in range(self.order)],dim=-1)).squeeze(-1)