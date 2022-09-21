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
        
        self.obs = obs
        self.eps=eps
        
        
    def fgsm_attack(self, data, data_grad):

        sign_data_grad = data_grad.sign()
        perturbed_obs = data + self.eps*sign_data_grad
        perturbed_obs = torch.clamp(perturbed_obs, -1, 1)

        return perturbed_obs    

    def fgsm_perturbation(self):
        obs = np.expand_dims(self.obs.astype(np.float32), axis=0)
        obs = torch.from_numpy(obs)
        obs = Variable(obs,requires_grad=True)
        action_distrib = self.model.policy(obs)
        actions = action_distrib.rsample()
        log_prob = action_distrib.log_prob(actions)

        q1 = self.model.q_func1((obs, actions))
        q2 = self.model.q_func2((obs, actions))


        q = torch.min(q1, q2)


        entropy_term = self.model.temperature * log_prob[..., None]
        assert q.shape == entropy_term.shape
        loss = torch.mean(entropy_term - q)

        self.model.policy_optimizer.zero_grad()
        loss.backward()

        obs_grad = obs.grad

        perturbed_data = self.fgsm_attack(obs, obs_grad)

        perturbed_data = torch.squeeze(perturbed_data.detach())

        return perturbed_data


class PGD(FGSM):
    def __init__(self,model,obs, num_steps=40, step_size=0.01, eps=0.3):
        super(PGD, self).__init__(model,obs,eps )
        self.num_steps=num_steps        
        self.step_size = step_size
 



    def PPO_gradient_Advantage(self,_x_adv ):
        action_distrib = self.model.policy(_x_adv) 


        action_adv = action_distrib.rsample()


        log_prob = action_distrib.log_prob(action_adv)

        q1 = self.model.q_func1((_x_adv, action_adv))
        q2 = self.model.q_func2((_x_adv, action_adv))


        q = torch.min(q1, q2)


        entropy_term = self.model.temperature * log_prob[..., None]
        assert q.shape == entropy_term.shape
        loss = torch.mean(entropy_term - q)

        self.model.policy_optimizer.zero_grad()


        return loss 
    def projected_gradient_descent(self,x, clamp):

        np.random.seed(10**5)
        x_adv = (2*np.random.random() -1) * x
        x_adv = x_adv.requires_grad_(True).to(x.device)



        for i in range(self.num_steps):
            _x_adv = x_adv.clone().detach().requires_grad_(True)



            loss = self.PPO_gradient_Advantage(_x_adv)
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

        obs = np.expand_dims(self.obs.astype(np.float32), axis=0)
        obs = torch.from_numpy(obs)
        obs = Variable(obs,requires_grad=True)

        perturbed_data = self.projected_gradient_descent(obs,clamp)
        perturbed_data = torch.squeeze(perturbed_data.detach())
        
        return perturbed_data