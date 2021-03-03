import os
import argparse
import numpy as np
import torch
import sys
from BVH import load, save
from ForwardKinematics import *
from Quaternions import *

# getting the coordinates of a particular joint
def get_coord(l, q, o):
    ret = o[l[0]]
    for i in l[1:]:
        ret = Quaternions(q[i]) * ret
        ret = ret + o[i]
    return ret

# forward kinematics loop
def fk(quats, offsets, parents):
    ret = np.zeros(offsets.shape)
    for i in range(len(ret)):
        curr_list = get_joints_list(parents, i)
        ret[i] = get_coord(curr_list, quats, offsets)
    return ret

# helper function for adding frame to bvh file
def add_frame(anim, qs, order):
    ret = np.concatenate((anim.rotations.qs, qs), 0)
    anim.rotations.qs = ret
    anim.positions = np.concatenate((anim.positions, np.array([anim.positions[-1]])), 0)
    return anim

# helper function to get list of parents
def get_joints_list(parents, joint_id):
    ret = [joint_id]
    curr = joint_id
    while parents[curr] != -1:
        ret.append(parents[curr])
        curr = parents[curr]
    return ret

# calculate numerical partial derivatives in respect to pose of a single joint
def partial_deriv(fk, idx, eulers, order, target_idx, initial_pos, anim):
    ret = np.zeros((3,3))
    for i in range(3):
        eulers[idx][i] += 0.0001
        rots = Quaternions.from_euler(eulers, order = order, world = False).qs
        pos = fk(rots, anim.offsets, anim.parents)[target_idx]
        diff = pos - initial_pos
        ret[i] = diff/0.0001
        eulers[idx][i] -= 0.0001
    return ret

# calculate analytical derivative in respect to pose of a single joint
def analytical_deriv(pos, joint_num, initial_pos):
    ret = np.zeros((3,3))
    for i in range(3):
        axis = np.zeros(3)
        axis[2-i] = 1
        ret[i] = np.cross(axis, initial_pos - pos[joint_num]) 
    return ret

# build jacobian
def build_jacobian(anim, joint_names, target_joint, target_pos, eulers, order, fk, mode = "numerical"):
    joint_id = joint_names.index(target_joint)
    joint_list = get_joints_list(anim.parents, joint_id)
    jacobian = np.zeros((len(joint_list)*3, 3))
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

# change euler angles
def modify_eulers(eulers, diff, l, no_change = [], rate = 0.00001):
    for i in range(len(l)):
        if l[i] not in no_change:
            eulers[l[i]] += rate*diff[i]
    return eulers

# IK loop for jacobian transpose
def ik_t(anim, joint_names, target_joint, target_pos, order, fk, N = 300, no_change = [], rate = 0.0001):
    curr_eulers = Quaternions(anim.rotations.qs[0]).euler(order = order)
    ret_eulers = []
    for i in range(N):
        for k in range(len(target_joint)):
            ret_eulers.append(np.copy(curr_eulers))
            j, l , ip = build_jacobian(anim, joint_names, target_joint[k], target_pos[k], curr_eulers, order, fk, mode = "analytical")
            de = target_pos[k] - np.array(ip)
            shifts = j.T@de.T
            shifts = np.array([shifts]).reshape((len(shifts)//3, 3))
            curr_eulers = modify_eulers(curr_eulers, shifts, l, no_change = no_change, rate = rate)
        anim = add_frame(anim, np.array([Quaternions.from_euler(curr_eulers, order = order[::-1]).qs]), order)
    ret_eulers.append(curr_eulers)
    return anim, np.array(ret_eulers)

# IK loop for jacobian pseudo inverse
def ik_pi(anim, joint_names, target_joint, target_pos, order, fk, N = 1000, lam = 10, no_change = [], rate = 0.01):
    curr_eulers = anim.rotations.euler(order = order)[0]
    ret_eulers = []
    for i in range(N):
        for k in range(len(target_joint)):
            ret_eulers.append(np.copy(curr_eulers))
            j, l , ip = build_jacobian(anim, joint_names, target_joint[k], target_pos[k], curr_eulers, order, fk, mode = "analytical")
            de = target_pos[k] - np.array(ip)
            pseudo_inv = j.T@(np.linalg.inv(j@j.T + lam*lam*np.diag([1.0,1.0,1.0])))
            shifts = pseudo_inv@de.T
            shifts = np.array([shifts]).reshape((len(shifts)//3, 3))
            curr_eulers = modify_eulers(curr_eulers, shifts, l, no_change = no_change, rate = rate)
        anim = add_frame(anim, np.array([Quaternions.from_euler(curr_eulers, order = order[::-1]).qs]), order)
    ret_eulers.append(curr_eulers)
    return anim, np.array(ret_eulers)
