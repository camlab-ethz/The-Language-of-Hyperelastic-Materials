"""
This code is based on the work of Benjamin Paaßen from The University of Sydney 
"""
import random
import numpy as np
import tree
from sympy.parsing.sympy_parser import parse_expr
import numpy as np
import sympy
import networkx as nx
import matplotlib.pyplot as plt


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

def scalar_derivation_wrt_matrix(sympy_expression, matrix, dim=3):
    P = sympy.zeros(dim)
    for ii in range(dim):
        for jj in range(dim):
            P[ii,jj] = sympy.diff(sympy_expression,matrix[ii,jj])
    P_matrix = sympy.Matrix(P)
    return P_matrix

def stress_correction(sympy_expression, F, dim=3):
    I = sympy.Identity(dim).as_explicit() 
    I1 = get_I1()
    I2 = get_I2()
    J = get_J()
    dpsidJ = sympy.diff(sympy_expression, J)
    dpsidJ = dpsidJ.subs({J:get_J_wrt_C()})
    n = dpsidJ
    n = sympy.simplify(n.subs({F:I}))
    stress_correction_FequalI = -n*(J-1)
    return stress_correction_FequalI

def get_J(dim=3):
    return sympy.Symbol("J")
    
def get_J_wrt_F(dim=3):
    return sympy.Determinant(sympy.MatrixSymbol("F",dim,dim).as_explicit())

def get_J_wrt_C(dim=3):
    F  = sympy.MatrixSymbol("F",dim,dim).as_explicit()
    return sympy.Pow(sympy.Determinant(F.T*F),0.5)

def get_I1(dim=3):
    return sympy.Symbol("I1")

def get_I1_wrt_F(dim=3):
    return sympy.Pow(sympy.Determinant(sympy.MatrixSymbol("F",dim,dim).as_explicit()),-2/3)*sympy.Trace(sympy.MatrixSymbol("F",dim,dim).T.as_explicit()*sympy.MatrixSymbol("F",dim,dim).as_explicit())

def get_I2(dim=3):
    return sympy.Symbol("I2")

def get_I2_wrt_F(dim=3):
    return sympy.Pow(sympy.Determinant(sympy.MatrixSymbol("F",dim,dim).as_explicit()),-4/3)*(0.5*(sympy.Pow(sympy.Trace(sympy.MatMul(sympy.MatrixSymbol("F",dim,dim).T.as_explicit(),sympy.MatrixSymbol("F",dim,dim).as_explicit())),2) - sympy.Trace(sympy.MatPow(sympy.MatMul(sympy.MatrixSymbol("F",dim,dim).T.as_explicit(),sympy.MatrixSymbol("F",dim,dim).as_explicit()),2))))

def sample_tree_with_corrections(dim=3, growthSwitch=0, positivitySwitch=0):
    F = sympy.MatrixSymbol('F', dim, dim)
    I = sympy.Identity(dim).as_explicit() 
    nodes,adj = sample_tree()

    sympy_expression = to_sympy_expression(to_algebraic_string(nodes,adj))

    simplified_sympy_expression = to_sympy_expression(to_invariants_algebraic_string(nodes,adj)).evalf(4)

    potential_correction = sympy.simplify(sympy_expression.subs({F:I})).evalf(4) 

    sympy_expression_w_invariants = to_sympy_expression(to_algebraic_string_w_invariants(nodes,adj))
    stress_correction_FequalI = stress_correction(sympy_expression_w_invariants, F)
    potential_correction_expression =  - potential_correction

    # F = sympy.symbols("F") 
    final_expression = simplified_sympy_expression  + potential_correction_expression.evalf(4) + stress_correction_FequalI.evalf(4)

    condition_count,  growthSwitch, positivitySwitch = perform_checks(sympy_expression, growthSwitch, positivitySwitch)
    # print(condition_count)
    if condition_count == 11:
        # print(stress_correction(final_expression, F))
        return final_expression, growthSwitch, positivitySwitch
    else:
        return sample_tree_with_corrections(dim, growthSwitch, positivitySwitch)

def sample_tree():
    production_rules = [_sample_literal_invariant_combo, \
                        _sample_unary, _sample_binary,\
                        _sample_combination]

    r = random.randrange(4)
    combination1 = production_rules[r]()
    r = random.randrange(4)
    combination2 = production_rules[r]()
    r = random.randrange(4)
    combination3 = production_rules[r]()
    r = random.randrange(4)
    combination4 = production_rules[r]()
    
    psi_vol = tree.Tree('+', [combination1, combination2])
    psi_iso = tree.Tree('+', [combination3, combination4])
    expr   = tree.Tree('+', [psi_vol, psi_iso])
    return expr.to_list_format()

def _sample_combination():
    r = random.randrange(3)
    if r == 0:
        children = [_sample_binary(), _sample_power()]
        return tree.Tree('pow', children)
    if r == 1:
        children = [_sample_binary()]
        return tree.Tree('exp', children)
    if r == 2:
        children = [_sample_binary()]
        # return tree.Tree('log', children)
        rlog =  tree.Tree('log', children)
        return tree.Tree('*',[tree.Tree('-1'), rlog])

def _sample_binary():
    r = random.randrange(6)
    left = _sample_invariants()
    right = _sample_literal()
    children = [left, right]
    if r == 0:
        return tree.Tree('+', children)
    if r == 1:
        return tree.Tree('-', children)
    if r == 2:
        return tree.Tree('*', children)
    if r == 3:
        return tree.Tree('/', children)
    if r == 4:
        children = [_sample_invariants(), _sample_power()]
        return tree.Tree('pow', children)
    if r == 5:
        children = [_sample_literal(), _sample_binary()]
        return tree.Tree('*', children)

def _sample_unary():
    r = random.randrange(2)
    if r == 0:
        children = [_sample_literal_invariant_combo()]
        return tree.Tree('exp', children)
    if r == 1:
        children = [_sample_literal_invariant_combo()]
        # return tree.Tree('log', children)
        rlog =  tree.Tree('log', children)
        return tree.Tree('*',[tree.Tree('-1'), rlog])

def _sample_literal():
    s = f'{np.random.uniform(0.,10.):.2f}'
    return tree.Tree(s)

def _sample_power():
    s = f'{np.random.randint(2,3):.2f}'
    return tree.Tree(s)

    # r = random.randrange(10)
    # s1 = '0123456789'[r]
    # r = random.randrange(10)
    # s2 = '0123456789'[r]
    # r = random.randrange(10)
    # s3 = '0123456789'[r]
    # return tree.Tree('0123456789'[r])

def _sample_invariants():
    r = random.randrange(3)
    if r == 0:
        s = 'I1'
    elif r == 1:
        # s = 'I2'
        children = [tree.Tree('I2'), tree.Tree('1.5')]
        return tree.Tree('pow', children)
    elif r == 2:
        s = 'J'
    return tree.Tree(s)

def _sample_literal_invariant_combo():
    r = random.randrange(4)
    if r == 0:
        s = 'I1'
    if r == 1:
        # s = 'I2'
        children = [tree.Tree('I2'), tree.Tree('1.5')]
        return tree.Tree('pow', children)
    elif r == 2:
        s = 'J'
    elif r == 3:
        s = f'{np.random.uniform(0,2):.2f}'
    return tree.Tree(s)

def to_algebraic_string(nodes, adj, i = 0):
    if nodes[i] == 'log' or nodes[i] == 'exp':
        return nodes[i] + '(' + to_algebraic_string(nodes, adj, adj[i][0]) + ')'
    if nodes[i] == '+' or nodes[i] == '*' or nodes[i] == '-' or nodes[i] == '/':
        return to_algebraic_string(nodes, adj, adj[i][0]) + ' ' + nodes[i] + ' ' + to_algebraic_string(nodes, adj, adj[i][1])
    if nodes[i] == 'pow':
        return 'Pow' + '(' + to_algebraic_string(nodes, adj, adj[i][0]) + ',' + to_algebraic_string(nodes, adj, adj[i][1]) + ')'
    if nodes[i] == 'J':
        return '(Determinant(MatrixSymbol("F",3,3).as_explicit()))'
    if nodes[i] == 'I1':
        return '(Pow(Determinant(MatrixSymbol("F",3,3).as_explicit()),-2/3)*Trace(MatrixSymbol("F",3,3).T.as_explicit()*MatrixSymbol("F",3,3).as_explicit()))'
    if nodes[i] == 'I2':
        return '(Pow(Determinant(MatrixSymbol("F",3,3).as_explicit()),-4/3)*(0.5*(Pow(Trace(MatMul(MatrixSymbol("F",3,3).T.as_explicit(),MatrixSymbol("F",3,3).as_explicit())),2) - Trace(MatPow(MatMul(MatrixSymbol("F",3,3).T.as_explicit(),MatrixSymbol("F",3,3).as_explicit()),2)))))'
    else:
        return nodes[i]

def to_algebraic_string_w_invariants(nodes, adj, i = 0):
    if nodes[i] == 'log' or nodes[i] == 'exp':
        return nodes[i] + '(' + to_algebraic_string_w_invariants(nodes, adj, adj[i][0]) + ')'
    if nodes[i] == '+' or nodes[i] == '*' or nodes[i] == '-' or nodes[i] == '/':
        return to_algebraic_string_w_invariants(nodes, adj, adj[i][0]) + ' ' + nodes[i] + ' ' + to_algebraic_string_w_invariants(nodes, adj, adj[i][1])
    if nodes[i] == 'pow':
        return 'Pow' + '(' + to_algebraic_string_w_invariants(nodes, adj, adj[i][0]) + ',' + to_algebraic_string_w_invariants(nodes, adj, adj[i][1]) + ')'
    if nodes[i] == 'J':
        return 'Symbol("J")'
    if nodes[i] == 'I1':
        return 'Symbol("I1")'
    if nodes[i] == 'I2':
        return 'Symbol("I2")'
    else:
        return nodes[i]
    
def to_invariants_algebraic_string(nodes, adj, i = 0):
    if nodes[i] == '+' or nodes[i] == '*' or nodes[i] == '-' or nodes[i] == '/':
        return to_invariants_algebraic_string(nodes, adj, adj[i][0]) + ' ' + nodes[i] + ' ' + to_invariants_algebraic_string(nodes, adj, adj[i][1])
    if nodes[i] == 'pow':
        return 'Pow' + '(' + to_invariants_algebraic_string(nodes, adj, adj[i][0]) + ',' + to_algebraic_string(nodes, adj, adj[i][1]) + ')'
    if nodes[i] == 'log' or nodes[i] == 'exp':
        return nodes[i] + '(' + to_invariants_algebraic_string(nodes, adj, adj[i][0]) + ')'
    if nodes[i] == 'J':
        return 'J'
    if nodes[i] == 'I1':
        return 'I1'
    if nodes[i] == 'I2':
        return 'I2'
    else:
        return nodes[i]


def to_numpy_algebraic_string(nodes, adj, i = 0):
    if nodes[i] == '+' or nodes[i] == '*' or nodes[i] == '-' or nodes[i] == '/':
        return to_numpy_algebraic_string(nodes, adj, adj[i][0]) + ' ' + nodes[i] + ' ' + to_numpy_algebraic_string(nodes, adj, adj[i][1])
    if nodes[i] == 'pow':
        return 'np.pow' + '(' + to_numpy_algebraic_string(nodes, adj, adj[i][0]) + ',' + to_algebraic_string(nodes, adj, adj[i][1]) + ')'
    if nodes[i] == 'log' or nodes[i] == 'exp':
        return 'np.' + nodes[i] + '(' + to_numpy_algebraic_string(nodes, adj, adj[i][0]) + ')'
    if nodes[i] == 'J':
        return 'np.linalg.det(F)'
    if nodes[i] == 'I1':
        return 'np.pow(np.linalg.det(F),-2/3)*np.trace(np.matmul(F.T,F))'
    if nodes[i] == 'I2':
        return 'np.pow(np.linalg.det(F),-4/3)*(0.5*(np.pow(np.trace(np.matmul(F.T,F)),2) - np.trace(np.pow(np.matmul(F.T,F),2))))'
    else:
        return nodes[i]

def to_sympy_expression(expression):
    return parse_expr(expression) 

def to_algebraic_string_polish_notation(nodes, adj, i = 0):
    if nodes[i] == 'I1' or nodes[i] == 'J' or nodes[i]== 'I2':
        return nodes[i]
    elif nodes[i] == 'pow':
        return  'Pow' + '(' + to_algebraic_string_infix_notation(nodes, adj, adj[i][0]) + ',' + to_algebraic_string_infix_notation(nodes, adj, adj[i][1]) + ')'
    elif nodes[i] == 'log' or nodes[i] == 'exp':
        return  nodes[i] + ' (' + to_algebraic_string_infix_notation(nodes, adj, adj[i][0]) + ') ' 
    elif nodes[i] == '+' or nodes[i] == '-' or nodes[i] == '/' or nodes[i] == '*':
        try:
            return  to_algebraic_string_infix_notation(nodes, adj, adj[i][0]) + ' ' + nodes[i] + ' (' +  to_algebraic_string_infix_notation(nodes, adj, adj[i][1]) + ') '
        except:
            return  nodes[i] + ' (' + to_algebraic_string_infix_notation(nodes, adj, adj[i][0]) + ') ' 
    else:
        return nodes[i]

def to_algebraic_string_infix_notation(nodes, adj, i = 0):
    if nodes[i] == 'I1' or nodes[i] == 'J'  or nodes[i] == 'I2':
        return nodes[i]
    elif nodes[i] == 'pow':
        return  'Pow' + '(' + to_algebraic_string_infix_notation(nodes, adj, adj[i][0]) + ',' + to_algebraic_string_infix_notation(nodes, adj, adj[i][1]) + ')'
    elif nodes[i] == 'log' or nodes[i] == 'exp':
        return  nodes[i] + ' (' + to_algebraic_string_infix_notation(nodes, adj, adj[i][0]) + ') ' 
    elif nodes[i] == '+' or nodes[i] == '-' or nodes[i] == '/' or nodes[i] == '*':
        try:
            return  to_algebraic_string_infix_notation(nodes, adj, adj[i][0]) + ' ' + nodes[i] + ' (' +  to_algebraic_string_infix_notation(nodes, adj, adj[i][1]) + ') '
        except:
            return  nodes[i] + ' (' + to_algebraic_string_infix_notation(nodes, adj, adj[i][0]) + ') ' 
    else:
        return nodes[i]

def get_stress_tensors(gamma1=0.55, gamma2=0.60):
    a1 = 1 + gamma1
    a2 = 1 + gamma2
    b = 1.
    c = 1.
    F_UT1 = sympy.diag(a1, b, c)
    F_UT2 = sympy.diag(a2, b, c)
    a1 = 1./(1 + gamma1)
    a2 = 1./(1 + gamma2)
    F_UC1 = sympy.diag(a1, b, c)
    F_UC2 = sympy.diag(a2, b, c)
    a1 = 1
    a2 = 1.
    off_Diag1 = sympy.zeros(3)
    off_Diag1[0,1] = gamma1
    off_Diag2 = sympy.zeros(3)
    off_Diag2[0,1] = gamma2
    F_SS1 = sympy.diag(a1, b, c) + off_Diag1
    F_SS2 = sympy.diag(a2, b, c) + off_Diag2
    a1 = 1. + gamma1
    a2 = 1. + gamma2
    b1 = 1. + gamma1
    b2 = 1. + gamma2
    F_BT1 = sympy.diag(a1, b1, c)
    F_BT2 = sympy.diag(a2, b2, c)
    gamma1 = 0.55
    gamma2 = 0.60
    a1 = 1./(1. + gamma1)
    a2 = 1./(1. + gamma2)
    b1 = 1./(1. + gamma1)
    b2 = 1./(1. + gamma2)
    F_BC1 = sympy.diag(a1, b1, c)
    F_BC2 = sympy.diag(a2, b2, c)
    gamma1 = 0.55
    gamma2 = 0.60
    a1 = 1. + gamma1
    a2 = 1. + gamma2
    b1 = 1./(1. + gamma1)
    b2 = 1./(1. + gamma2)
    F_PS1 = sympy.diag(a1, b1, c)
    F_PS2 = sympy.diag(a2, b2, c)

    a1 = sympy.sqrt(float(gamma1))
    b1 = sympy.sqrt((4. - float(gamma1))/(1. + 2.*float(gamma1)))
    F_new11 = sympy.diag(a1, b1, c)

    a2 = sympy.sqrt(float(gamma2))
    b2 = sympy.sqrt((4. - float(gamma2))/(1. + 2.*float(gamma2)))
    F_new12 = sympy.diag(a2, b2, c)

    a1 = sympy.sqrt(float(gamma1)/(2*float(gamma1)-1))
    b1 = sympy.sqrt(float(gamma1))
    F_new21 = sympy.diag(a1, b1, c)

    a2 = sympy.sqrt(float(gamma2)/(2*float(gamma2)-1))
    b2 = sympy.sqrt(float(gamma2))
    F_new22 = sympy.diag(a2, b2, c)

    a1 = sympy.sqrt((5- float(gamma1))/(1 + 3*float(gamma1)))
    b1 = sympy.sqrt(float(gamma1))
    F_new31 = sympy.diag(a1, b1, c)

    a2 = sympy.sqrt((5- float(gamma2))/(1 + 3*float(gamma2)))
    b2 = sympy.sqrt(float(gamma2))
    F_new32 = sympy.diag(a2, b2, c)

    return F_UT1, F_UT2, F_UC1, F_UC2, F_SS1, F_SS2, F_BT1, \
           F_BT2, F_BC1, F_BC2, F_PS1, F_PS2, F_new11, F_new12, \
           F_new21, F_new22, F_new31, F_new32

def check_non_negativity(condition_count, expression, tensors, F, positivitySwitch):
    F_UT1, F_UT2, F_UC1, F_UC2, F_SS1, F_SS2, F_BT1, \
           F_BT2, F_BC1, F_BC2, F_PS1, F_PS2, F_new11, F_new12, \
           F_new21, F_new22, F_new31, F_new32 = tensors

    init_count = condition_count

    try:
        if sympy.simplify(expression.subs({F:F_UT2})) > sympy.simplify(expression.subs({F:F_UT1})):
            condition_count +=1 
            # print("UT")
    except:
        return condition_count, positivitySwitch

    try:
        if sympy.simplify(expression.subs({F:F_UC2})) > sympy.simplify(expression.subs({F:F_UC1})):
            condition_count +=1 
            # print("UC")
    except:
        return condition_count, positivitySwitch

    try:
        if sympy.simplify(expression.subs({F:F_SS2})) > sympy.simplify(expression.subs({F:F_SS1})):
            condition_count +=1 
            # print("SS")
    except:
        return condition_count, positivitySwitch

    try:
        if sympy.simplify(expression.subs({F:F_BT2})) > sympy.simplify(expression.subs({F:F_BT1})):
            condition_count +=1 
            # print("BT")
    except:
        return condition_count, positivitySwitch

    try:
        if sympy.simplify(expression.subs({F:F_BC2})) > sympy.simplify(expression.subs({F:F_BC1})):
            condition_count +=1 
            # print("BC")
    except:
        return condition_count, positivitySwitch

    try:
        if sympy.simplify(expression.subs({F:F_PS2})) > sympy.simplify(expression.subs({F:F_PS1})):
            condition_count +=1 
            # print("PS")
    except:
        return condition_count, positivitySwitch

    # print(sympy.simplify(expression.subs({F:F_new12})), sympy.simplify(expression.subs({F:F_new11})), "New1")
    # print(sympy.simplify(expression.subs({F:F_new22})), sympy.simplify(expression.subs({F:F_new21})), "New2")
    # print(sympy.simplify(expression.subs({F:F_new32})), sympy.simplify(expression.subs({F:F_new31})), "New3")
    try:
        if sympy.simplify(expression.subs({F:F_new12})) < sympy.simplify(expression.subs({F:F_new11})):
            condition_count +=1 
            # print("New1")
    except:
        return condition_count, positivitySwitch

    try:
        if sympy.simplify(expression.subs({F:F_new22})) < sympy.simplify(expression.subs({F:F_new21})):
            condition_count +=1 
            # print("New2")
    except:
        return condition_count, positivitySwitch

    try:
        if sympy.simplify(expression.subs({F:F_new32})) < sympy.simplify(expression.subs({F:F_new31})):
            condition_count +=1 
            # print("New3")
    except:
        return condition_count, positivitySwitch
       
    if condition_count - init_count < 9:
        positivitySwitch += 1
    else:
        positivitySwitch += 0

    return condition_count, positivitySwitch


def check_growth(condition_count, expression, F, growthSwitch):
    init_count = condition_count
    if sympy.simplify(expression.subs({F:sympy.Matrix([[1000,0,0],[0,1,0],[0,0,1]])})).evalf(4) > 1000:
        condition_count +=1
    if sympy.simplify(expression.subs({F:sympy.Matrix([[0.0001,0,0],[0,1,0],[0,0,1]])})).evalf(4) > 1000:
        condition_count +=1

    if condition_count - init_count < 9:
        growthSwitch += 1
    else:
        growthSwitch += 0

    return condition_count, growthSwitch

def perform_checks(expression, growthSwitch, positivitySwitch):
    F = sympy.MatrixSymbol('F', 3, 3)
    condition_count = 0
    if expression is None:
        return condition_count
    tensors = get_stress_tensors()
    condition_count, growthSwitch     = check_growth(condition_count, expression, F, growthSwitch)
    condition_count, positivitySwitch = check_non_negativity(condition_count, expression, tensors, F, positivitySwitch)
    return condition_count,  growthSwitch, positivitySwitch

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def plot_graph(expr):
    """
    Make a graph plot of the internal representation of SymPy expression.
    """

    node_list = []
    link_list = []

    class Id:
        """A helper class for autoincrementing node numbers."""
        counter = 0

        @classmethod
        def get(cls):
            cls.counter += 1
            return cls.counter

    class Node:
        """Represents a single operation or atomic argument."""

        def __init__(self, label, expr_id):
            self.id = expr_id
            self.name = label

        def __repr__(self):
            return self.name
        
        
    def _walk(parent, expr):
        """Walk over the expression tree recursively creating nodes and links."""
        if expr.is_Atom:
            node = Node(str(expr), Id.get())
            node_list.append({"id": node.id, "name": node.name})
            link_list.append({"source": parent.id, "target": node.id})
        else:
            node = Node(str(type(expr).__name__), Id.get())
            node_list.append({"id": node.id, "name": node.name})
            link_list.append({"source": parent.id, "target": node.id})
            for arg in expr.args:
                _walk(node, arg)

    _walk(Node("Root", 0), expr)

    # Create the graph from the lists of nodes and links:    
    graph_json = {"nodes": node_list, "links": link_list}
    node_labels = {node['id']: node['name'] for node in graph_json['nodes']}
    for n in graph_json['nodes']:
        del n['name']
    graph = json_graph.node_link_graph(graph_json, directed=True, multigraph=False)
    pos = hierarchy_pos(graph,0)    
    # Layout and plot the graph
    plt.figure(1,figsize=(12,12)) 
    nx.draw_networkx(graph.to_directed(), pos, labels=node_labels, node_shape="s",  node_color="none", bbox=dict(facecolor="skyblue", edgecolor='black', boxstyle='round,pad=0.2'))
    plt.show()