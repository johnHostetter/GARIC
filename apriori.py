# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 19:52:17 2020

@author: jhost
"""

from neurofuzzynetwork import NFN_bellShapedMembership, Rule, Term, Variable

def init():
    # --- antecedent terms ---
    po1 = Term(2, NFN_bellShapedMembership, {'center':0.3, 'sigma':0.3}, label='PO1')
    ze1 = Term(2, NFN_bellShapedMembership, {'center':0.0, 'sigma':0.3}, label='ZE1')
    ne1 = Term(2, NFN_bellShapedMembership, {'center':-0.3, 'sigma':0.3}, label='NE1')
    vs1 = Term(2, NFN_bellShapedMembership, {'center':0.0, 'sigma':0.05}, label='VS1')
    po2 = Term(3, NFN_bellShapedMembership, {'center':1.0, 'sigma':1.0}, label='PO2')
    ze2 = Term(3, NFN_bellShapedMembership, {'center':0.0, 'sigma':1.0}, label='ZE2')
    ne2 = Term(3, NFN_bellShapedMembership, {'center':-1.0, 'sigma':1.0}, label='NE2')
    vs2 = Term(3, NFN_bellShapedMembership, {'center':0.0, 'sigma':0.1}, label='VS2')
    po3 = Term(0, NFN_bellShapedMembership, {'center':0.3, 'sigma':0.3}, label='PO3')
    ne3 = Term(0, NFN_bellShapedMembership, {'center':-0.5, 'sigma':0.5}, label='NE3')
    po4 = Term(1, NFN_bellShapedMembership, {'center':1.0, 'sigma':1.0}, label='PO4')
    ne4 = Term(1, NFN_bellShapedMembership, {'center':-1.0, 'sigma':1.0}, label='NE4')
    ps4 = Term(1, NFN_bellShapedMembership, {'center':0.0, 'sigma':1.0}, label='PS4')
    ns4 = Term(1, NFN_bellShapedMembership, {'center':0.0, 'sigma':1.0}, label='NS4')
    
    # --- consequent terms ---
    pl = Term(4, NFN_bellShapedMembership, {'center':20.0, 'sigma':2.5}, label='PL')
    pm = Term(4, NFN_bellShapedMembership, {'center':10.0, 'sigma':5.0}, label='PM')
    ps = Term(4, NFN_bellShapedMembership, {'center':5.0, 'sigma':4.0}, label='PS')
    pvs = Term(4, NFN_bellShapedMembership, {'center':1.0, 'sigma':1.0}, label='PVS')
    nl = Term(4, NFN_bellShapedMembership, {'center':-20.0, 'sigma':2.5}, label='NL')
    nm = Term(4, NFN_bellShapedMembership, {'center':-10.0, 'sigma':5.0}, label='NM')
    ns = Term(4, NFN_bellShapedMembership, {'center':-5.0, 'sigma':4.0}, label='NS')
    nvs = Term(4, NFN_bellShapedMembership, {'center':-1.0, 'sigma':1.0}, label='NVS')
    ze = Term(4, NFN_bellShapedMembership, {'center':0.0, 'sigma':1.0}, label='ZE')
    
    # --- antecedent variables ---
    var0 = Variable(0, [po3, ne3], 'Cart Position')
    var1 = Variable(1, [po4, ps4, ne4, ns4], 'Cart Velocity')
    var2 = Variable(2, [po1, ze1, ne1, vs1], 'Pole Angle')
    var3 = Variable(3, [po2, ze2, ne2, vs2], 'Pole Velocity At Tip')
    
    # --- consequent variables ---
    var4 = Variable(0, [pl, pm, ps, pvs, nl, nm, ns, nvs, ze], 'Force')
    
    inputVariables = [var0, var1, var2, var3]
    outputVariables = [var4]
    
    rules = [
            Rule([po1, po2, None, None], [pl]),
            Rule([po1, ze2, None, None], [pm]),
            Rule([po1, ne2, None, None], [ze]),
            Rule([ze1, po2, None, None], [ps]),
            Rule([ze1, ze2, None, None], [ze]),
            Rule([ze1, ne2, None, None], [ns]),
            Rule([ne1, po2, None, None], [ze]),
            Rule([ne1, ze2, None, None], [nm]),
            Rule([ne1, ne2, None, None], [nl]),
            Rule([vs1, vs2, po3, po4], [ps]),
            Rule([vs1, vs2, po3, ps4], [pvs]),
            Rule([vs1, vs2, ne3, ne4], [ns]),
            Rule([vs1, vs2, ne3, ns4], [nvs])
            ]
    return (inputVariables, outputVariables, rules)