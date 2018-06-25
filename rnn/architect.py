import torch
import numpy as np
import torch.nn as nn
from model_search import RNNModel
from torch.autograd import Variable


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


def _clip(grads, max_norm):
    total_norm = 0
    for g in grads:
        param_norm = g.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grads:
            g.data.mul_(clip_coef)
    return total_norm


class Architect(object):

  def __init__(self, model, args):
    self.network_weight_decay = args.wdecay
    self.network_clip = args.clip
    self.model = model
    self.optimizer = torch.optim.Adam(
        self.model.arch_parameters(), lr=args.arch_lr, weight_decay=args.arch_wdecay)

  def _compute_unrolled_model(self, hidden, input, target, eta):
    loss, hidden_next = self.model._loss(hidden, input, target)
    theta = _concat(self.model.parameters()).data
    grads = torch.autograd.grad(loss, self.model.parameters())
    _clip(grads, self.network_clip)
    dtheta = _concat(grads).data + self.network_weight_decay*theta
    model_unrolled = self._construct_model_from_theta(theta.sub(eta, dtheta))
    return model_unrolled

  def step(self, hidden_train, input_train, target_train,
          hidden_valid, input_valid, target_valid, network_optimizer, unrolled):
    eta = network_optimizer.param_groups[0]['lr']
    self.optimizer.zero_grad()
    if unrolled:
        hidden = self._backward_step_unrolled(
            hidden_train, input_train, target_train,
            hidden_valid, input_valid, target_valid, eta)
    else:
        hidden = self._backward_step(hidden_valid, input_valid, target_valid)

    self.optimizer.step()
    return hidden, None

  def _backward_step(self, hidden, input, target):
    loss, hidden_next = self.model._loss(hidden, input, target)
    for v in self.model.arch_parameters():
      if v.grad is not None:
        v.grad.data.zero_()
    loss.backward()
    return hidden_next

  def _backward_step_unrolled(self, hidden_train, input_train, target_train,
          hidden_valid, input_valid, target_valid, eta):
    model_unrolled = self._compute_unrolled_model(hidden_train, input_train, target_train, eta)
    loss, hidden_next = model_unrolled._loss(hidden_valid, input_valid, target_valid)
    grads = torch.autograd.grad(loss, model_unrolled.arch_parameters(), retain_graph=True)

    theta = model_unrolled.parameters()
    dtheta = torch.autograd.grad(loss, model_unrolled.parameters())
    _clip(dtheta, self.network_clip)
    vector = [dt.data.add(self.network_weight_decay, t.data) for dt, t in zip(dtheta, theta)]
    grads_implicit = self._hessian_vector_product(
            model_unrolled, vector, hidden_train, input_train, target_train)

    for g, ig in zip(grads, grads_implicit):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), grads):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)
    return hidden_next

  def _construct_model_from_theta(self, theta):
    model_clone = self.model.clone()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_clone.load_state_dict(model_dict)
    return model_clone.cuda()

  def _hessian_vector_product(self, model, vector, hidden, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(model.parameters(), vector):
      p.data.add_(R, v)
    loss, _ = model._loss(hidden, input, target)
    grads_p = torch.autograd.grad(loss, model.arch_parameters())

    for p, v in zip(model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss, _ = model._loss(hidden, input, target)
    grads_n = torch.autograd.grad(loss, model.arch_parameters())

    for p, v in zip(model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R + 1e-10) for x, y in zip(grads_p, grads_n)]

