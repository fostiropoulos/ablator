Data Type Configuration
=======================

These data type classes are used in configuration classes to specify data type of each config attribute,
which provides ``ablator`` with the flexibility to expand into various configuration formats.

Common data types
-----------------
Ablator supports common structural data types like list, dictionary, etc. You can use these to annotate
configuration attributes. Details of each data type can be found in the following sections:

.. autoclass:: ablator.config.types.List()
   :noindex:

.. autoclass:: ablator.config.types.Tuple()
   :noindex:

.. autoclass:: ablator.config.types.Dict()
   :noindex:

.. autoclass:: ablator.config.types.Optional()
   :noindex:

.. autoclass:: ablator.config.types.Enum()
   :noindex:
   :exclude-members: __eq__, __hash__

Ablator custom data types
-------------------------

The next data classes are specific to ablator framework: ``Derived``, ``Stateless``, and ``Stateful``.
Users have the option to wrap these around the common data types, python primitive type, or custom
classes to further modify their behavior. `Configuration
Basics <./notebooks/Configuration-Basics.ipynb>`_ tutorial also discusses about these data types.

.. autoclass:: ablator.config.types.Stateless
   :noindex:
   :members:

.. autoclass:: ablator.config.types.Derived
   :noindex:
   :members:

.. autoclass:: ablator.config.types.Stateful
   :noindex:
   :members:
