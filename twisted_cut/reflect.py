# -*- test-case-name: twisted.test.test_reflect -*-
# Copyright (c) Twisted Matrix Laboratories.
# See LICENSE for details.

"""
Standardized versions of various cool and/or strange things that you can do
with Python's reflection capabilities.
"""

from __future__ import division, absolute_import, print_function

import types
import pickle
import re
import traceback

from io import BytesIO as NativeStringIO

RegexType = type(re.compile(""))


def prefixedMethodNames(classObj, prefix):
    """
    Given a class object C{classObj}, returns a list of method names that match
    the string C{prefix}.

    @param classObj: A class object from which to collect method names.

    @param prefix: A native string giving a prefix.  Each method with a name
        which begins with this prefix will be returned.
    @type prefix: L{str}

    @return: A list of the names of matching methods of C{classObj} (and base
        classes of C{classObj}).
    @rtype: L{list} of L{str}
    """
    dct = {}
    addMethodNamesToDict(classObj, dct, prefix)
    return list(dct.keys())


def addMethodNamesToDict(classObj, dict, prefix, baseClass=None):
    """
    This goes through C{classObj} (and its bases) and puts method names
    starting with 'prefix' in 'dict' with a value of 1. if baseClass isn't
    None, methods will only be added if classObj is-a baseClass

    If the class in question has the methods 'prefix_methodname' and
    'prefix_methodname2', the resulting dict should look something like:
    {"methodname": 1, "methodname2": 1}.

    @param classObj: A class object from which to collect method names.

    @param dict: A L{dict} which will be updated with the results of the
        accumulation.  Items are added to this dictionary, with method names as
        keys and C{1} as values.
    @type dict: L{dict}

    @param prefix: A native string giving a prefix.  Each method of C{classObj}
        (and base classes of C{classObj}) with a name which begins with this
        prefix will be returned.
    @type prefix: L{str}

    @param baseClass: A class object at which to stop searching upwards for new
        methods.  To collect all method names, do not pass a value for this
        parameter.

    @return: L{None}
    """
    for base in classObj.__bases__:
        addMethodNamesToDict(base, dict, prefix, baseClass)

    if baseClass is None or baseClass in classObj.__bases__:
        for name, method in classObj.__dict__.items():
            optName = name[len(prefix):]
            if ((type(method) is types.FunctionType)
                and (name[:len(prefix)] == prefix)
                and (len(optName))):
                dict[optName] = 1



def prefixedMethods(obj, prefix=''):
    """
    Given an object C{obj}, returns a list of method objects that match the
    string C{prefix}.

    @param obj: An arbitrary object from which to collect methods.

    @param prefix: A native string giving a prefix.  Each method of C{obj} with
        a name which begins with this prefix will be returned.
    @type prefix: L{str}

    @return: A list of the matching method objects.
    @rtype: L{list}
    """
    dct = {}
    accumulateMethods(obj, dct, prefix)
    return list(dct.values())



def accumulateMethods(obj, dict, prefix='', curClass=None):
    """
    Given an object C{obj}, add all methods that begin with C{prefix}.

    @param obj: An arbitrary object to collect methods from.

    @param dict: A L{dict} which will be updated with the results of the
        accumulation.  Items are added to this dictionary, with method names as
        keys and corresponding instance method objects as values.
    @type dict: L{dict}

    @param prefix: A native string giving a prefix.  Each method of C{obj} with
        a name which begins with this prefix will be returned.
    @type prefix: L{str}

    @param curClass: The class in the inheritance hierarchy at which to start
        collecting methods.  Collection proceeds up.  To collect all methods
        from C{obj}, do not pass a value for this parameter.

    @return: L{None}
    """
    if not curClass:
        curClass = obj.__class__
    for base in curClass.__bases__:
        # The implementation of the object class is different on PyPy vs.
        # CPython.  This has the side effect of making accumulateMethods()
        # pick up object methods from all new-style classes -
        # such as __getattribute__, etc.
        # If we ignore 'object' when accumulating methods, we can get
        # consistent behavior on Pypy and CPython.
        if base is not object:
            accumulateMethods(obj, dict, prefix, base)

    for name, method in curClass.__dict__.items():
        optName = name[len(prefix):]
        if ((type(method) is types.FunctionType)
            and (name[:len(prefix)] == prefix)
            and (len(optName))):
            dict[optName] = getattr(obj, name)


def namedModule(name):
    """
    Return a module given its name.
    """
    topLevel = __import__(name)
    packages = name.split(".")[1:]
    m = topLevel
    for p in packages:
        m = getattr(m, p)
    return m


def namedObject(name):
    """
    Get a fully named module-global object.
    """
    classSplit = name.split('.')
    module = namedModule('.'.join(classSplit[:-1]))
    return getattr(module, classSplit[-1])

namedClass = namedObject # backwards compat


def requireModule(name, default=None):
    """
    Try to import a module given its name, returning C{default} value if
    C{ImportError} is raised during import.

    @param name: Module name as it would have been passed to C{import}.
    @type name: C{str}.

    @param default: Value returned in case C{ImportError} is raised while
        importing the module.

    @return: Module or default value.
    """
    try:
        return namedModule(name)
    except ImportError:
        return default



class _NoModuleFound(Exception):
    """
    No module was found because none exists.
    """



class InvalidName(ValueError):
    """
    The given name is not a dot-separated list of Python objects.
    """



class ModuleNotFound(InvalidName):
    """
    The module associated with the given name doesn't exist and it can't be
    imported.
    """



class ObjectNotFound(InvalidName):
    """
    The object associated with the given name doesn't exist and it can't be
    imported.
    """


def qual(clazz):
    """
    Return full import path of a class.
    """
    return clazz.__module__ + '.' + clazz.__name__


def _determineClass(x):
    try:
        return x.__class__
    except:
        return type(x)


def _determineClassName(x):
    c = _determineClass(x)
    try:
        return c.__name__
    except:
        try:
            return str(c)
        except:
            return '<BROKEN CLASS AT 0x%x>' % id(c)


def _safeFormat(formatter, o):
    """
    Helper function for L{safe_repr} and L{safe_str}.

    Called when C{repr} or C{str} fail. Returns a string containing info about
    C{o} and the latest exception.

    @param formatter: C{str} or C{repr}.
    @type formatter: C{type}
    @param o: Any object.

    @rtype: C{str}
    @return: A string containing information about C{o} and the raised
        exception.
    """
    io = NativeStringIO()
    traceback.print_exc(file=io)
    className = _determineClassName(o)
    tbValue = io.getvalue()
    return "<%s instance at 0x%x with %s error:\n %s>" % (
        className, id(o), formatter.__name__, tbValue)


def safe_repr(o):
    """
    Returns a string representation of an object, or a string containing a
    traceback, if that object's __repr__ raised an exception.

    @param o: Any object.

    @rtype: C{str}
    """
    try:
        return repr(o)
    except:
        return _safeFormat(repr, o)


def safe_str(o):
    """
    Returns a string representation of an object, or a string containing a
    traceback, if that object's __str__ raised an exception.
    @param o: Any object.
    @rtype: C{str}
    """
    try:
        return str(o)
    except:
        return _safeFormat(str, o)


class QueueMethod:
    """
    I represent a method that doesn't exist yet.
    """
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls
    def __call__(self, *args):
        self.calls.append((self.name, args))


def fullFuncName(func):
    qualName = (str(pickle.whichmodule(func, func.__name__)) + '.' + func.__name__)
    if namedObject(qualName) is not func:
        raise Exception("Couldn't find %s as %s." % (func, qualName))
    return qualName


def getClass(obj):
    """
    Return the class or type of object 'obj'.
    Returns sensible result for oldstyle and newstyle instances and types.
    """
    if hasattr(obj, '__class__'):
        return obj.__class__
    else:
        return type(obj)


def accumulateClassDict(classObj, attr, adict, baseClass=None):
    """
    Accumulate all attributes of a given name in a class hierarchy into a single dictionary.

    Assuming all class attributes of this name are dictionaries.
    If any of the dictionaries being accumulated have the same key, the
    one highest in the class hierarchy wins.
    (XXX: If \"highest\" means \"closest to the starting class\".)

    Ex::

      class Soy:
        properties = {\"taste\": \"bland\"}

      class Plant:
        properties = {\"colour\": \"green\"}

      class Seaweed(Plant):
        pass

      class Lunch(Soy, Seaweed):
        properties = {\"vegan\": 1 }

      dct = {}

      accumulateClassDict(Lunch, \"properties\", dct)

      print(dct)

    {\"taste\": \"bland\", \"colour\": \"green\", \"vegan\": 1}
    """
    for base in classObj.__bases__:
        accumulateClassDict(base, attr, adict)
    if baseClass is None or baseClass in classObj.__bases__:
        adict.update(classObj.__dict__.get(attr, {}))


def accumulateClassList(classObj, attr, listObj, baseClass=None):
    """
    Accumulate all attributes of a given name in a class hierarchy into a single list.

    Assuming all class attributes of this name are lists.
    """
    for base in classObj.__bases__:
        accumulateClassList(base, attr, listObj)
    if baseClass is None or baseClass in classObj.__bases__:
        listObj.extend(classObj.__dict__.get(attr, []))


def isSame(a, b):
    return (a is b)


def isLike(a, b):
    return (a == b)


__all__ = [
    'InvalidName', 'ModuleNotFound', 'ObjectNotFound',

    'QueueMethod',

    'namedModule', 'namedObject', 'namedClass',  'requireModule',
    'safe_repr', 'prefixedMethodNames', 'addMethodNamesToDict',
    'prefixedMethods', 'accumulateMethods', 'fullFuncName', 'qual', 'getClass',
    'accumulateClassDict', 'accumulateClassList', 'isSame', 'isLike',
    ]
