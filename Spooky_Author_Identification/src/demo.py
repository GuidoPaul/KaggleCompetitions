#!/usr/bin/env python
# -*- coding: utf-8 -*-


def fun_a(arg1, arg2, arg3):
    print(arg1, arg2, arg3)


def fun_b(arg1, arg2, arg3):
    print(arg1, arg2)


def fun_c(arg1, arg2, arg3):
    print(arg1)


def run(fun, arg1=None, arg2=None, arg3=None):
    fun(arg1, arg2, arg3)


run(fun_c, 1)
