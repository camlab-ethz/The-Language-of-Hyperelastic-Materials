"""
Follows the implementation of Kusner et al. (2017).
"""

import random
import numpy as np
import tree
import tree_grammar
import sympy
from sympy.parsing.sympy_parser import parse_expr

# set up grammar of the expressions domain. Note that this grammar does
# _not_ contain information about the specific shape of the training data
# imposed by Kusner et al., but permits any valid expression over the
# alphabet.

alphabet = {'+' : 2, '*' : 2, '-':2, '/' : 2, 'log' : 1, 'exp' : 1, '(J-1)' : 0, '(I1-3)' : 0, '(I2-3)' : 0, '0,5' : 0, '3' : 0, '1' : 0, '1,5' : 0,
           '0,7' : 0, '0,2' : 0, '2':0, 'pow':2}

nonterminals = ['S']
start    = 'S'

rules    = { 'S' : [
    ('+', ['S', 'S']),
    ('-', ['S', 'S']),
    ('*', ['S', 'S']),
    ('/', ['S', 'S']),
    ('pow', ['S', 'S']),
    ('log', ['S']),
    ('exp', ['S']),
    ('(J-1)', []),
    ('(I1-3)', []),
    ('(I2-3)', []),
    ('0,2',[]),
    ('0,5',[]), 
    ('0,7',[]), 
    ('1,5',[]), 
    ('2',[]), 
    ('3',[])
]}

grammar = tree_grammar.TreeGrammar(alphabet, nonterminals, start, rules)

def tensor_dot_product_3D(S, F):
    return S[0,0]*F[0,0] + S[0,1]*F[0,1] + S[0,2]*F[0,2] +\
           S[1,0]*F[1,0] + S[1,1]*F[1,1] + S[1,2]*F[1,2] +\
           S[2,0]*F[2,0] + S[2,1]*F[2,1] + S[2,2]*F[2,2] 

def differentiate_scalar_with_matrix(model, F):
    P = sympy.Matrix([[0,0,0],[0,0,0],[0,0,0]])
    for ii in range(3):
        for jj in range(3):
            P[ii,jj] = sympy.diff(model,F[ii,jj])
    return P
 
def sample_tree():
    r = random.randrange(2)
    combination1 = _sample_combination(flag='vol')
    combination2 = _sample_combination(flag='iso')
    combination3 = _sample_combination(flag='iso')
    psi_vol = combination1
    psi_iso = tree.Tree('+', [combination2, combination3])
    expr = tree.Tree('+', [psi_vol, psi_iso])
    return expr.to_list_format()

def _sample_combination(flag=None):
    r = random.randrange(4)
    if r == 0:
        return _sample_binary(flag=flag)
    if r == 1:
        return _sample_unary(flag=flag)
    if r == 2:
        r = random.randrange(2)
        left = _sample_binary(flag=flag)
        right = tree.Tree('23'[r])
        children = [left, right]
        return tree.Tree('pow', children)
    if r == 3:
        left = _sample_binary(flag=flag)
        right = _sample_binary(flag=flag)
        children = [left, right]
        return tree.Tree('*', children)
 

def _sample_binary(flag=None):
    r = random.randrange(4)
    if r ==0:
        return _sample_literal(flag=flag)
    left = _sample_literal(flag=flag)
    right = _sample_literal(flag=flag)
    children = [left, right]
    if r == 1:
        return tree.Tree('+', children)
    if r == 1:
        return tree.Tree('*', children)
    if r == 3:
        return tree.Tree('/', children)

def _sample_unary(flag=None):
    r = random.randrange(2)
    if r ==0:
        return _sample_literal(flag=flag)
    children = [_sample_literal(flag=flag)]
    if r == 1:
        return tree.Tree('log', children)

def _sample_literal(flag=None):
    r = random.randrange(2)
    if r==0:
        r = random.randrange(4)
        if r == 0:
            return tree.Tree('0,2')
        if r == 1:
            return tree.Tree('0,5')
        if r == 2:
            return tree.Tree('0,7')
        if r == 3:
            return tree.Tree('1,5')
    if r == 1:
        r1 = random.randrange(3)
        if r1 == 0:
            return tree.Tree('(I1-3)')
        if r1 == 1:
            return tree.Tree('(I2-3)')
        if r1 == 2:
            return tree.Tree('(J-1)')

 
def to_algebraic_string(nodes, adj, i = 0):
    """ Transforms a given tree representation of an algebraic expression into
    a more readable string form.

    Note that this method only works for inputs conforming to the structure of
    sample_tree(). Otherwise, the bracketing may be wrong.

    Parameters
    ----------
    nodes: list
        The node list of the input tree.
    adj: list
        The adjacency list of the input tree.
    i: int (optional, default = 0)
        The root index of the input tree.

    Returns
    -------
    str: string
        The string representation of the input tree.
    """
    if nodes[i] == '+' or nodes[i] == '*' or nodes[i] == '-'  or nodes[i] == '/':
        return to_algebraic_string(nodes, adj, adj[i][0]) + ' ' + nodes[i] + ' ' + to_algebraic_string(nodes, adj, adj[i][1])
    if nodes[i] == 'log' or nodes[i] == 'exp':
        return nodes[i] + '(' + to_algebraic_string(nodes, adj, adj[i][0]) + ')'
    if nodes[i] == 'pow':
        return '(' + to_algebraic_string(nodes, adj, adj[i][0]) + ')' + '**' + to_algebraic_string(nodes, adj, adj[i][1])
    else:
        return nodes[i]

def objective_function(nodes, adj):
    """ Returns the prediction error of the given expression compared to the
    ground truth expression 1/3+x+sin(x*x) as described by Kusner et al.
    (2017). This is the objective functiion for expression optimization.

    In more detail, the returned error is log(1 + MSE), where the MSE is
    computed on 1000 linearly spaced points in the range -10 to 10.

    Parameters
    ----------
    nodes: list
        The node list of the input expression tree.
    adj: list
        The adjacency list of the input expression tree.

    Returns
    -------
    loss: float
        The prediction error as descibed above.
    """
    # create inputs
    I1 = np.linspace(0, 2., 1000)
    I2 = np.linspace(0, 2, 1000)
    J  = np.linspace(0, 2, 1000)
    # compute ground-truth values
    y = 0.5*(I1 - 3) + 1.5*(J - 1)**2
    # compute predicted values
    y_pred = evaluate(nodes, adj, I1, I2, J)
    return np.log(1. + np.mean((y - y_pred) ** 2))