Data Type Configuration
=======================

These data type classes are used in configuration classes to specify data type of each config attribute,
which provides ``ablator`` with the flexibility to expand into various configuration formats.

Common data types
-----------------
Ablator supports common structural data types like list, dictionary, etc. Details of each data type
can be found in the following sections.

.. autoclass:: ablator.config.types.List
   :noindex:
   :members:
   :show-inheritance:

.. autoclass:: ablator.config.types.Tuple
   :noindex:
   :members:
   :show-inheritance:

.. autoclass:: ablator.config.types.Dict
   :noindex:
   :members:
   :show-inheritance:

.. autoclass:: ablator.config.types.Optional
   :noindex:
   :members:
   :show-inheritance:

.. autoclass:: ablator.config.types.Enum
   :noindex:
   :members:
   :show-inheritance:
   :exclude-members: __eq__, __hash__

Ablator custom data types
-------------------------

The next data classes are specific to ablator framework: ``Derived``, ``Stateless``, and ``Stateful``.
Users have the option to wrap these around the common data types, python primitive type, or custom
classes to further modify their behavior. To learn more about these data types, go to `Configuration
Basics <./notebooks/Configuration-Basics.ipynb>`_ tutorial.

.. autoclass:: ablator.config.types.Stateless
   :noindex:
   :members:
   :show-inheritance:

.. autoclass:: ablator.config.types.Derived
   :noindex:
   :members:
   :show-inheritance:

.. autoclass:: ablator.config.types.Stateful
   :noindex:
   :members:
   :show-inheritance:
