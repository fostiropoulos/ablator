Data Type Configuration
=======================

Common data types
-----------------
These data types classes are used to define configuration classes, which gives ``ablator``
the power to extend to different configuration formats.

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

The next data classes are specific to ablator framework: ``Derived``,
``Stateless``, and ``Stateful``. To learn more about these data types, go to :ref:`Configuration Basics <config_basic_tutorial>` tutorial.

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
