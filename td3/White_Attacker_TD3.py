import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable



class FGSM(nn.Module):
    def __init__(self,model,obs,eps):
        super(FGSM, self).__init__()
        self.model=model
        self.device=torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        
        self.obs = torch.Tensor(obs)
        self.eps=eps
        
        
    def fgsm_attack(self, data, data_grad):

        sign_data_grad = data_grad.sign()
        perturbed_obs = data + self.eps*sign_data_grad
        perturbed_obs = torch.clamp(perturbed_obs, -1, 1)

        return perturbed_obs    
    


    def fgsm_perturbation(self):

        obs = Variable(self.obs,requires_grad=True)
        onpolicy_actions = self.model.policy(obs).rsample()
        q = self.model.q_func1((obs, onpolicy_actions))
        loss = -q.mean()


        loss.backward()

        obs_grad = obs.grad

        perturbed_data = self.fgsm_attack(self.obs, obs_grad)

        return perturbed_data



class PGD(FGSM):
    def __init__(self,model,obs,num_steps=40, step_size=0.01, eps=0.3):
        super(PGD, self).__init__(model,obs,eps)
        self.num_steps=num_steps        
        self.step_size = step_size
 

    def projected_gradient_descent(self,x, clamp):

        np.random.seed(10**5)
        x_adv = (2*np.random.random() -1) * x
        x_adv = x_adv.requires_grad_(True).to(x.device)

        for i in range(self.num_steps):
            _x_adv = x_adv.clone().detach().requires_grad_(True)

            onpolicy_actions = self.model.policy(_x_adv).rsample()
            q = self.model.q_func1((_x_adv, onpolicy_actions))
            loss = -q.mean()

            loss.backward()

            with torch.no_grad():

                gradients = _x_adv.grad.sign() * self.step_size


                x_adv += gradients

            eta = torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
            x_adv = torch.clamp(x + eta, *clamp).detach_()


        return x_adv.detach()


    def PGD_perturbation(self,clip):

        if clip:
            clamp = (-1,1)

        obs = Variable(self.obs,requires_grad=True)

        perturbed_data = self.projected_gradient_descent(obs,clamp)

        return perturbed_data