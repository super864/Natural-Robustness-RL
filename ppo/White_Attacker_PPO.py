import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable



class FGSM(nn.Module):
    def __init__(self,model,obs,eps, gamma=0.995):
        super(FGSM, self).__init__()
        self.model=model
        self.device=torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        
        self.obs = torch.Tensor(obs)
        self.eps=eps

        self.gamma = gamma
        
        
    def fgsm_attack(self, data, data_grad):

        sign_data_grad = data_grad.sign()
        perturbed_obs = data + self.eps*sign_data_grad
        perturbed_obs = torch.clamp(perturbed_obs, -1, 1)

        return perturbed_obs    


    def fgsm_perturbation(self, env_copy):

        obs = Variable(self.obs,requires_grad=True)

        action_distrib, vs_pred = self.model.model(obs)


        action = action_distrib.sample().cpu().numpy()


        obs_next, rewards, done, _ = env_copy.step(action)

        obs_next = torch.Tensor(obs_next)
        _, next_vs = self.model.model(obs_next)


        loss = (
            rewards
            + (self.gamma * done * next_vs)
            - vs_pred
        )


        loss.backward()

        obs_grad = obs.grad

        perturbed_data = self.fgsm_attack(self.obs, obs_grad)

        return perturbed_data


class PGD(FGSM):
    def __init__(self,model,obs,gamma, num_steps=40, step_size=0.01, eps=0.3):
        super(PGD, self).__init__(model,obs,eps,gamma )
        self.num_steps=num_steps        
        self.step_size = step_size
 



    def PPO_gradient_Advantage(self,_x_adv,env_copy ):
        action_distrib, vs_pred = self.model.model(_x_adv)


        action_adv = action_distrib.sample().cpu().numpy()


        obs_next, rewards, done, _ = env_copy.step(action_adv)

        obs_next = torch.Tensor(obs_next)
        _, next_vs = self.model.model(obs_next)


        loss = (
            rewards
            + (self.gamma * done * next_vs)
            - vs_pred
        )

        return loss 
    def projected_gradient_descent(self,x, env_copy, clamp):

        np.random.seed(10**5)
        x_adv = (2*np.random.random() -1) * x
        x_adv = x_adv.requires_grad_(True).to(x.device)



        for i in range(self.num_steps):
            _x_adv = x_adv.clone().detach().requires_grad_(True)



            loss = self.PPO_gradient_Advantage(_x_adv,env_copy)
            loss.backward()


            with torch.no_grad():

                gradients = _x_adv.grad.sign() * self.step_size

                x_adv += gradients

            eta = torch.clamp(x_adv - x, min=-self.eps, max=self.eps)
            x_adv = torch.clamp(x + eta, *clamp).detach_()


        return x_adv.detach()


    def PGD_perturbation(self,clip,env_copy):

        if clip:
            clamp = (-1,1)

        obs = Variable(self.obs,requires_grad=True)

        perturbed_data = self.projected_gradient_descent(obs,env_copy,clamp)

        return perturbed_data