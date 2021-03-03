import os
import argparse
import numpy as np
import torch
import sys
from BVH import load, save
from ForwardKinematics import *
from Quaternions import *

def get_coord(l, q, o):
    ret = o[l[0]]
    for i in l[1:]:
        ret = Quaternions(q[i]) * ret
        ret = ret + o[i]
    return ret


def fk(quats, offsets, parents):
    ret = np.zeros(offsets.shape)
    for i in range(len(ret)):
        curr_list = get_joints_list(parents, i)
        ret[i] = get_coord(curr_list, quats, offsets)
    return ret

def add_frame(anim, qs, order):
    ret = np.concatenate((anim.rotations.qs, qs), 0)
    anim.rotations.qs = ret
    anim.positions = np.concatenate((anim.positions, np.array([anim.positions[-1]])), 0)
    return anim

def get_joints_list(parents, joint_id):
    ret = [joint_id]
    curr = joint_id
    while parents[curr] != -1:
        ret.append(parents[curr])
        curr = parents[curr]
    return ret

def partial_deriv(fk, idx, eulers, order, target_idx, initial_pos, anim):
    ret = np.zeros((3,3))
    for i in range(3):
        eulers[idx][i] += 0.0001
        #print("eulers", eulers)
        rots = Quaternions.from_euler(eulers, order = order, world = False).qs
        #pos = fk.run_local(torch.Tensor(np.array([rots])).double())[0][0][target_idx]
        #print("rots", rots)
        pos = fk(rots, anim.offsets, anim.parents)[target_idx]
        #print("pos", pos)
        diff = pos - initial_pos
        ret[i] = diff/0.0001
        eulers[idx][i] -= 0.0001
    return ret

def analytical_deriv(pos, joint_num, initial_pos):
    ret = np.zeros((3,3))
    for i in range(3):
        axis = np.zeros(3)
        axis[2-i] = 1
        ret[i] = np.cross(axis, initial_pos - pos[joint_num]) 
    return ret


def build_jacobian(anim, joint_names, target_joint, target_pos, eulers, order, fk, mode = "numerical"):
    joint_id = joint_names.index(target_joint)
    joint_list = get_joints_list(anim.parents, joint_id)
    jacobian = np.zeros((len(joint_list)*3, 3))
    #initial_pos = fk.run_local(torch.Tensor(np.array([rots])).double())[0][0][joint_id]
    #print(Quaternions.from_euler(eulers, order = order).qs)
    pos = fk(Quaternions.from_euler(eulers, order = order).qs, anim.offsets, anim.parents)
    initial_pos = pos[joint_id]
    
    for i in range(len(joint_list)):
        if mode == "numerical":
            curr_pd = partial_deriv(fk, joint_list[i], eulers, order, joint_id, initial_pos, anim)
        else:
            curr_pd = analytical_deriv(pos, joint_list[i], initial_pos)
        jacobian[3*i]     = curr_pd[0]
        jacobian[3*i + 1] = curr_pd[1]
        jacobian[3*i + 2] = curr_pd[2]
    return jacobian.T, joint_list, initial_pos

def modify_eulers(eulers, diff, l, no_change = [], rate = 0.00001):
    for i in range(len(l)):
        if l[i] not in no_change:
            eulers[l[i]] += rate*diff[i]
    return eulers



def ik_t(anim, joint_names, target_joint, target_pos, order, fk, N = 300, no_change = [], rate = 0.0001):
    curr_eulers = Quaternions(anim.rotations.qs[0]).euler(order = order)
    ret_eulers = []
    for i in range(N):
        print(i)
        for k in range(len(target_joint)):
            ret_eulers.append(np.copy(curr_eulers))
            j, l , ip = build_jacobian(anim, joint_names, target_joint[k], target_pos[k], curr_eulers, order, fk, mode = "analytical")
            de = target_pos[k] - np.array(ip)
            print(np.linalg.norm(de))
            shifts = j.T@de.T
            shifts = np.array([shifts]).reshape((len(shifts)//3, 3))
            curr_eulers = modify_eulers(curr_eulers, shifts, l, no_change = no_change, rate = rate)
        anim = add_frame(anim, np.array([Quaternions.from_euler(curr_eulers, order = order[::-1]).qs]), order)
    ret_eulers.append(curr_eulers)
    return anim, np.array(ret_eulers)

def ik_pi(anim, joint_names, target_joint, target_pos, order, fk, N = 1000, lam = 10, no_change = [], rate = 0.01):
    curr_eulers = anim.rotations.euler(order = order)[0]
    ret_eulers = []
    for i in range(N):
        print(i)
        for k in range(len(target_joint)):
            ret_eulers.append(np.copy(curr_eulers))
            #print(curr_eulers[-4])
            j, l , ip = build_jacobian(anim, joint_names, target_joint[k], target_pos[k], curr_eulers, order, fk, mode = "analytical")
            #print(l)
            de = target_pos[k] - np.array(ip)
            pseudo_inv = j.T@(np.linalg.inv(j@j.T + lam*lam*np.diag([1.0,1.0,1.0])))
            #print(j@j.T)
            shifts = pseudo_inv@de.T
            #print(pseudo_inv)
            #print(shifts)
            print(np.linalg.norm(de))
            shifts = np.array([shifts]).reshape((len(shifts)//3, 3))
            curr_eulers = modify_eulers(curr_eulers, shifts, l, no_change = no_change, rate = rate)
        anim = add_frame(anim, np.array([Quaternions.from_euler(curr_eulers, order = order[::-1]).qs]), order)
    ret_eulers.append(curr_eulers)
    return anim, np.array(ret_eulers)