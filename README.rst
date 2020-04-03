FlyingCircus
============

**FlyingCircus** - Everything you always wanted to have in Python.\*

(\*But were afraid to write)

.. code::

     _____ _       _              ____ _
    |  ___| |_   _(_)_ __   __ _ / ___(_)_ __ ___ _   _ ___
    | |_  | | | | | | '_ \ / _` | |   | | '__/ __| | | / __|
    |  _| | | |_| | | | | | (_| | |___| | | | (__| |_| \__ \
    |_|   |_|\__, |_|_| |_|\__, |\____|_|_|  \___|\__,_|___/
             |___/         |___/


Overview
--------

This software provides a library of miscellaneous utilities / recipes for
generic computations with Python.
It is relatively easy to extend and users are encouraged to tweak with it.

Most of the code is used in a number of projects where it is tested
against real-life scenarios.

All the code is tested againt the examples in the documentation (using `doctest <https://docs.python.org/3/library/doctest.html>`__).

The code has reached a reasonable maturity.
However, until it gets a wider adoption, some of the library components may
undergo some refactoring in the process of improving the code.
Changes will appear in the ``CHANGELOG.rst``.
Please file a bug report if you detect an undocumented refactoring.

Releases information are available through ``NEWS.rst``.

For a more comprehensive list of changes see ``CHANGELOG.rst`` (automatically
generated from the version control system).


Features
--------

The package ``base`` contains a number of generic functions like

-  ``multi_replace()``: performs multiple replacements in a string.
-  ``flatten()``: recursively flattens nested iterables, e.g.
   list of list of tuples to flat list).
-  ``uniques()``: extract unique items from an iterable while
   keeping the order of appearance.

etc.

These are meant to run both in Python 3 and in PyPy.
For this reason, dependencies to external modules are kept to a minimum.
etc.


Installation
------------

The recommended way of installing the software is through
`PyPI <https://pypi.python.org/pypi/flyingcircus>`__:

.. code:: bash

    $ pip install flyingcircus

Alternatively, you can clone the source repository from
`GitHub <https://github.com/norok2/flyingcircus>`__:

.. code:: bash

    $ git clone git@github.com:norok2/flyingcircus.git
    $ cd flyingcircus
    $ pip install -e .

For more details see also ``INSTALL.rst``.


License
-------

This work is licensed through the terms and conditions of the
`GPLv3+ <http://www.gnu.org/licenses/gpl-3.0.html>`__ See the
accompanying ``LICENSE.rst`` for more details.


Acknowledgements
----------------

For a complete list of authors please see ``AUTHORS.rst``.

People who have influenced this work are acknowledged in ``THANKS.rst``.
