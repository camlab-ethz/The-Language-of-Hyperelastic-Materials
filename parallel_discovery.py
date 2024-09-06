from concurrent import futures

import numpy as np

import matplotlib.pyplot as plt
import scienceplots
import time
import edist.ted as ted
import joblib

# try to load data
import os

import hyperelastic_laws as hyperelastic
import torch
import sympy 
from sympy import Pow, MatrixSymbol, Trace, log, MatMul
import numpy as np
import symengine as se
from sympy.parsing.sympy_parser import parse_expr

import tree

import cma

import recursive_tree_grammar_auto_encoder as rtg_ae

T = 3125*6
N = 32
N_test = 100

learning_rate = 1E-3
weight_decay = 1e-4

dim = 128
dim_vae  = 8
num_params = 0
model = rtg_ae.TreeGrammarAutoEncoder(hyperelastic.grammar, dim = dim, dim_vae = dim_vae)
for key in model.state_dict():
    num_params += model.state_dict()[key].numel()
print('The model contains %d parameters' % (num_params))

def to_algebraic_string(nodes, adj, i = 0):
    if list(',') in list(map(str.split, nodes[i])):
        return list(map(str.split, str(nodes[i])))[0][0] + '.'+ list(map(str.split, str(nodes[i])))[2][0]
    if nodes[i] == '+' or nodes[i] == '*' or nodes[i] == '-'  or nodes[i] == '/':
        return to_algebraic_string(nodes, adj, adj[i][0]) + ' ' + nodes[i] + ' ' + to_algebraic_string(nodes, adj, adj[i][1])
    if nodes[i] == 'log' or nodes[i] == 'exp':
        return nodes[i] + '(' + to_algebraic_string(nodes, adj, adj[i][0]) + ')'
    if nodes[i] == 'pow':
        return '(' + to_algebraic_string(nodes, adj, adj[i][0]) + ')' + '**' + to_algebraic_string(nodes, adj, adj[i][1])
    if nodes[i] == '(J-1)':
        return '(' + nodes[i] + ')' 
    if nodes[i] == '(I1-3)':
        return '(' + 'I1-3' + ')' 
    if nodes[i] == '(I2-3)':
        return '(' + 'I2-3' + ')'
    else:
        return nodes[i]

def evaluate_se_W(nodes, adj, I1_tilde, I2_tilde, J_tilde):
    J, I1, I2 = se.symbols("J I1 I2")
    sympy_expression = se.sympify(to_algebraic_string(nodes,adj))
    y_all = np.zeros_like(I1_tilde)
    for i in range(0, I1_tilde.shape[0]):
        y_all[i] = np.array(sympy_expression.subs({J:float(J_tilde[i]), I1:float(I1_tilde[i]), I2:float(I2_tilde[i])}).evalf(16)).astype(np.float128)
    return y_all

def compute_invariants():
    F11, F12, F21, F22 = se.symbols("F11 F12 F21 F22")
    F = sympy.MutableDenseMatrix([[F11, F12,0],[F21, F22, 0],[0,0,1]])
    J  = sympy.Pow(sympy.Determinant(F.T*F),0.5)
    I1 = sympy.Pow(J,-2/3)*sympy.Trace(F.T*F)
    I2 = sympy.Pow(J,-4/3)*(0.5*(sympy.Pow(sympy.Trace(F.T*F),2) - sympy.Trace(sympy.Pow(F.T*F,2))))
    return sympy.simplify(J), sympy.simplify(I1), sympy.simplify(I2), F11, F12, F21, F22

# def compute_invariants():
#     F11, F12, F21, F22 = se.symbols("F11 F12 F21 F22")
#     C11 = F11**2 + F21**2
#     C12 = F11*F12 + F21*F22
#     C21 = F11*F12 + F21*F22
#     C22 = F12**2 + F22**2

#     # Compute computeStrainInvariants
#     I1 = C11 + C22 + 1.0
#     I2 = C11 + C22 - C12*C21 + C11*C22
#     I3 = C11*C22 - C12*C21
#     I1 = I1 * sympy.Pow(I3,-1/3) 
#     I2 = (I1 + I3 - 1.) * sympy.Pow(I3,-2/3)
#     J = sympy.Pow(I3,0.5)
#     return sympy.simplify(J), sympy.simplify(I1), sympy.simplify(I2), F11, F12, F21, F22

Ji, I1i, I2i, F11i, F12i, F21i, F22i = compute_invariants()

def evaluate_se_mb_test(nodes, adj, F11_v, F12_v, F21_v, F22_v):
    sympy_expression = se.sympify(to_algebraic_string(nodes,adj))
    J, I1, I2 = se.symbols("J I1 I2")
    sympy_expression = sympy_expression.subs({J:Ji, I1:I1i, I2:I2i})
    y_all = np.zeros_like(F11_v)
    for i in range(0, F11_v.shape[0]):
        y_all[i] = np.array(sympy_expression.subs({F11i:F11_v[i], F12i:F12_v[i], F21i:F21_v[i], F22i:F22_v[i]}).evalf(16)).astype(np.float128)
    return y_all

def objective_function_W(nodes, adj, I1_tilde, I2_tilde, J_tilde, y):
    y_pred = evaluate_se_W(nodes, adj, I1_tilde, I2_tilde, J_tilde)
    return np.sqrt(np.mean((y - y_pred) ** 2 ))

def evaluate_se(F11_v, F12_v, F21_v, F22_v):
    J, I1, I2 = se.symbols("J I1 I2")
    W = W.subs({J:Ji, I1:I1i, I2:I2i})
    y_all = np.zeros_like(F11_v)
    for i in range(0, F11_v.shape[0]):
        y_all[i] = np.array(W.subs({F11i:F11_v[i], F12i:F12_v[i], F21i:F21_v[i], F22i:F22_v[i]}).evalf(16)).astype(np.float128)
    return y_all

def objective_function_inv(F11_v, F12_v, F21_v, F22_v, y):
    y_pred = evaluate_se(F11_v, F12_v, F21_v, F22_v)
    return np.sqrt(np.mean((y - y_pred) ** 2 ))

def evaluate_se_mb(nodes, adj, F11, F12, F21, F22, num_nodes_per_element, numNodes,\
                 voigt_map, gradNa, qpWeights, connectivity, dirichlet_nodes, \
                 reactions, dim=2):

    W = se.sympify(to_algebraic_string(nodes,adj))
    # W = se.sympify("0.5*(I1-3) + 1.5*(J-1)**2")
    # print(W)
    J, I1, I2 = se.symbols("J I1 I2")
    # dWdJ = se.diff(W,J) 
    # Wc = dWdJ*(J-1)
    W = W.subs({J:Ji, I1:I1i, I2:I2i})
    F11_0 = 1.
    F12_0 = 0
    F21_0 = 0
    F22_0 = 1

    # Get gradients of W w.r.t F (compute)
    dW_NN_dF11 = se.diff(W,F11i)
    dW_NN_dF12 = se.diff(W,F12i)
    dW_NN_dF21 = se.diff(W,F21i)
    dW_NN_dF22 = se.diff(W,F22i)

    # print(dW_NN_dF11)
    # print(dW_NN_dF12)
    # print(dW_NN_dF21)
    # print(dW_NN_dF22)

    P11 = np.zeros((F11.shape[0],))
    P21 = np.zeros((F11.shape[0],))
    P12 = np.zeros((F11.shape[0],))
    P22 = np.zeros((F11.shape[0],))

    P11_0 = np.zeros((F11.shape[0],))
    P21_0 = np.zeros((F11.shape[0],))
    P12_0 = np.zeros((F11.shape[0],))
    P22_0 = np.zeros((F11.shape[0],))

    # Sc  = np.array(Wc.subs({F11i:1, F12i:0, F21i:0, F22i:1}).evalf(16)).astype(np.float128)
    # W0  =  np.array(W.subs({F11i:1, F12i:0, F21i:0, F22i:1}).evalf(16)).astype(np.float128)
    # print(W0, "W0")
    # print(Sc, "Sc")
    # print( np.nan_to_num(W0, neginf=10), "W0 not -inf")
    # print( np.nan_to_num(Sc, neginf=10), "Sc not -inf")

    for i in range(0, F11.shape[0]):
        P11[i] = np.array(dW_NN_dF11.subs({F11i:F11[i], F12i:F12[i], F21i:F21[i], F22i:F22[i]}).evalf(16)).astype(np.float128)
        P12[i] = np.array(dW_NN_dF12.subs({F11i:F11[i], F12i:F12[i], F21i:F21[i], F22i:F22[i]}).evalf(16)).astype(np.float128)
        P21[i] = np.array(dW_NN_dF21.subs({F11i:F11[i], F12i:F12[i], F21i:F21[i], F22i:F22[i]}).evalf(16)).astype(np.float128)
        P22[i] = np.array(dW_NN_dF22.subs({F11i:F11[i], F12i:F12[i], F21i:F21[i], F22i:F22[i]}).evalf(16)).astype(np.float128)

        P11_0[i] = np.array(dW_NN_dF11.subs({F11i:F11_0, F12i:F12_0, F21i:F21_0, F22i:F22_0}).evalf(16)).astype(np.float128)
        P12_0[i] = np.array(dW_NN_dF12.subs({F11i:F11_0, F12i:F12_0, F21i:F21_0, F22i:F22_0}).evalf(16)).astype(np.float128)
        P21_0[i] = np.array(dW_NN_dF21.subs({F11i:F11_0, F12i:F12_0, F21i:F21_0, F22i:F22_0}).evalf(16)).astype(np.float128)
        P22_0[i] = np.array(dW_NN_dF22.subs({F11i:F11_0, F12i:F12_0, F21i:F21_0, F22i:F22_0}).evalf(16)).astype(np.float128)

    # Assemble First Piola-Kirchhoff stress components
    P_N = torch.from_numpy(np.concatenate((P11[:,None],P12[:,None],P21[:,None], P22[:,None]),axis=1)).double()

    # # Get gradients of W_NN_0 w.r.t F
    P_0 = torch.from_numpy(np.concatenate((P11_0[:,None],P12_0[:,None],P21_0[:,None], P22_0[:,None]),axis=1)).double()

    P_cor = torch.zeros_like(P_0)

    # # Compute stress correction components according to Ansatz
    P_cor[:,0:1] = torch.from_numpy(F11[:,None])*-P_0[:,0:1] + torch.from_numpy(F12[:,None])*-P_0[:,2:3]
    P_cor[:,1:2] = torch.from_numpy(F11[:,None])*-P_0[:,1:2] + torch.from_numpy(F12[:,None])*-P_0[:,3:4]
    P_cor[:,2:3] = torch.from_numpy(F21[:,None])*-P_0[:,0:1] + torch.from_numpy(F22[:,None])*-P_0[:,2:3]
    P_cor[:,3:4] = torch.from_numpy(F21[:,None])*-P_0[:,1:2] + torch.from_numpy(F22[:,None])*-P_0[:,3:4]

    # # Compute final stress (NN + correction)
    P = P_N + 100*P_cor

    # compute internal forces on nodes
    f_int_nodes = torch.zeros(numNodes,dim).double()
    for a in range(num_nodes_per_element):
        for i in range(dim):
            for j in range(dim):
                force = P[:,voigt_map[i][j]].double() * gradNa[a][:,j].double() * qpWeights.double()
                f_int_nodes[:,i].index_add_(0, connectivity[a],force.double())

    # clone f_int_nodes
    f_int_nodes_clone = f_int_nodes.clone()
    # set force on Dirichlet BC nodes to zero
    f_int_nodes_clone[dirichlet_nodes] = 0.
    # loss for force equillibrium
    eqb_loss = torch.sum(f_int_nodes_clone**2).double()

    reaction_loss = torch.tensor([0.])
    for reaction in reactions:
        reaction_loss += (torch.sum(f_int_nodes[reaction.dofs]).double() - reaction.force.astype(np.float128))**2
    loss = eqb_loss.detach().numpy().astype(np.float128) + reaction_loss.detach().numpy().astype(np.float128) 

    return loss[0].astype(np.float128)

def objective_function(nodes, adj, F11, F12, F21, F22, y,  num_nodes_per_element, numNodes,\
                 voigt_map, gradNa, qpWeights, connectivity, dirichlet_nodes, \
                 reactions):
    # y_pred = evaluate_se_mb(nodes, adj, F11, F12, F21, F22)
    loss = evaluate_se_mb(nodes, adj, F11, F12, F21, F22, num_nodes_per_element, numNodes,\
                 voigt_map, gradNa, qpWeights, connectivity, dirichlet_nodes, \
                 reactions)
    # print(loss)
    return loss

def Neo_Hookean(I1, I2, J):
    return 0.5*(I1 - 3) + 1.5*(J - 1)**2

def Isihara(I1, I2, J):
    return 0.5*(I1 - 3) + (I2 - 3) + (I1 - 3)**2 + 1.5*(J-1)**2

def Haines_Wilson(I1, I2, J):
    return 0.5*(I1 - 3) + (I2 - 3) + 0.7*(I1 - 3)*(I2 - 3) + 0.2*(I1 - 3)**3 + 1.5*(J-1)**2

def Gent_Thomas(I1, I2, J):
    return 0.5*(I1 - 3) + np.log(I2/3) + 1.5*(J-1)**2

def Ogden(I1, I2, J):
        kappa_ogden = 1.5
        mu_ogden = 0.65
        alpha_ogden = 0.65
        I1_tilde = I1 + 0.0000001
        I1t_0 = np.array([3]) + 0.0000001
        J_0 = np.array([1]) + 0.0000001
        W_offset = kappa_ogden*(J_0-1)**2 + 1/alpha_ogden * 2. * (0.5**alpha_ogden*(I1t_0  +  np.sqrt(  (I1t_0-1/(J_0**(2./3.)))**2 - 4*J_0**(2./3.)) - 1/(J_0**(2./3.)) )**alpha_ogden+( 0.5*I1t_0 - 0.5*np.sqrt(  (I1t_0-1/(J_0**(2./3.)))**2 - 4*J_0**(2./3.))  - 0.5/(J_0**(2./3.)) )**alpha_ogden + J_0**(-alpha_ogden*2./3.) ) * mu_ogden
        W_truth = kappa_ogden*(J-1)**2 + 1/alpha_ogden * 2. * (0.5**alpha_ogden*(I1_tilde  +  np.sqrt(  (I1_tilde-1/(J**(2./3.)))**2 - 4*J**(2./3.)) - 1/(J**(2./3.)) )**alpha_ogden+( 0.5*I1_tilde - 0.5*np.sqrt(  (I1_tilde-1/(J**(2./3.)))**2 - 4*J**(2./3.))  - 0.5/(J**(2./3.)) )**alpha_ogden + J**(-alpha_ogden*2./3.) ) * mu_ogden - W_offset
        return W_truth

def decoding_functions(h):
    nodes, adj, _ = model.decode(torch.tensor(h, dtype=torch.float), max_size = 2*11)
    return nodes, adj


# model_name = 'Neo-Hookean'
model_name = 'Isihara'
# model_name = 'Haines-Wilson'
# model_name = 'Gent-Thomas'
# model_name = 'Ogden'

invalid_const = 1000.
noiseLevel = 'noise1e4'

if model_name == "Neo-Hookean":
    save_file  = '/home/gkissas/GrammarVAEs/rtgae-master 2/' + 'NeoHookean30_' + noiseLevel
    d30 = np.load(save_file +'.npz',allow_pickle=True)
    f = Neo_Hookean

elif model_name == "Isihara":
    save_file      = '/home/gkissas/GrammarVAEs/rtgae-master 2/' + 'Isihara30_' + noiseLevel
    d30 = np.load(save_file +'.npz',allow_pickle=True)
    f = Isihara

elif model_name == "Haines-Wilson":
    save_file = '/home/gkissas/GrammarVAEs/rtgae-master 2/' + 'HainesWilson30_' + noiseLevel
    d30 = np.load(save_file +'.npz',allow_pickle=True)
    f = Haines_Wilson

elif model_name == "Gent-Thomas":
    save_file  = '/home/gkissas/GrammarVAEs/rtgae-master 2/' + 'GentThomas30_' + noiseLevel
    d30 = np.load(save_file +'.npz',allow_pickle=True)
    f = Gent_Thomas

elif model_name == "Ogden":
    save_file  = '/home/gkissas/GrammarVAEs/rtgae-master 2/' + 'Ogden30_' + noiseLevel
    d30 = np.load(save_file +'.npz',allow_pickle=True)
    f = Ogden
print(model_name)
print(save_file)
J   = d30['J'][:,0]
I1  = d30['I1'][:,0]
I2  = d30['I2'][:,0]

# I1 = I1 * np.power(I3,-1/3) 
# I2 = (I1 + I3 - 1.) * np.power(I3,-2/3)
# J = np.power(I3,0.5)

I1 = J**(-2/3)*I1
I2 = J**(-4/3)*I2
y = f(I1.astype(np.float128), I2.astype(np.float128), J.astype(np.float128))

F11  = d30['F'][:,0]
F12  = d30['F'][:,1]
F21  = d30['F'][:,2]
F22  = d30['F'][:,3]
num_nodes_per_element = d30['numNodesPerElement']
numNodes  = d30['numNodes'] 
voigt_map = torch.from_numpy(d30['voigtMap'])
gradNa0= d30['gradNa0']
gradNa1= d30['gradNa1']
gradNa2= d30['gradNa2']
gradNa = torch.from_numpy(np.concatenate((gradNa0[None,...],gradNa1[None,...],gradNa2[None,...]), axis=0))
qpWeights= torch.from_numpy(d30['qpWeights'])
connectivity0= d30['connectivity0']
connectivity1= d30['connectivity1']
connectivity2= d30['connectivity2']
connectivity = torch.from_numpy(np.concatenate((connectivity0[None,...],connectivity1[None,...],connectivity2[None,...]), axis=0))
dirichlet_nodes= torch.from_numpy(d30['dirichlet_nodes'])
reactions=d30['reactions']

model.load_state_dict(torch.load('results/saved_model_hyperelastic.torch'))

def objective_fun(h):
    try:
        nodes, adj = decoding_functions(h)
    # return objective_function(nodes, adj, I1, I2,J, y)
        return objective_function(nodes, adj, F11, F12, F21, F22, y, num_nodes_per_element, numNodes,\
                    voigt_map, gradNa, qpWeights, connectivity, dirichlet_nodes, \
                    reactions)
    except Exception as ex:
        return invalid_const


def create_tree_function(counter):
    print(counter)
    text_file = open(model_name +"_" + noiseLevel +"_" + "Flaschel" + "_%d"%counter + ".txt", "w")
    mu    = np.zeros(dim_vae)
    es = cma.CMAEvolutionStrategy(mu, 0.1, {'popsize' : 100, 'ftarget': 1e-4,'verbose':1, 'maxiter' : 15})
    es.optimize(objective_fun)
    h_opt = es.best.__dict__['x']
    nodes, adj = decoding_functions(h_opt)
    # f_opt =objective_function(nodes, adj, I1, I2, J, y)
    f_opt =objective_function(nodes, adj, F11, F12, F21, F22, y,  num_nodes_per_element, numNodes,\
                 voigt_map, gradNa, qpWeights, connectivity, dirichlet_nodes, \
                 reactions)
    W = se.sympify(to_algebraic_string(nodes,adj))
    W_init = W
    J, I1, I2 = se.symbols("J I1 I2")
    potential_correction = sympy.simplify(W.subs({J:1, I1:3, I2:3})).evalf(16) 
    corrected_expression = W_init - potential_correction
    print('CMA-ES found the following optimal tree: %s with evaluation %g \n' % (corrected_expression, f_opt))
    text_file.write('CMA-ES found the following optimal tree: %s with evaluation %g \n' % (corrected_expression, f_opt))
    text_file.close()


number_of_trees = np.arange(100, 160)
if __name__ == '__main__':
    with futures.ProcessPoolExecutor(max_workers=20) as executor:
        executor.map(create_tree_function, number_of_trees) 

# create_tree_function(0)