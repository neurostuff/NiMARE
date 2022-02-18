:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   :members:  # option to generate document for the members of the target module, class or exception
   :undoc-members:  # option to generate document for the members not having docstrings
   :inherited-members:  # option to include members inherited from base classes
   :private-members:  # option to generate document for the private members
   :special-members:  # option to generate document for the special members (like __special__)
   {% block methods %}
   {% endblock %}

.. include:: {{fullname}}.examples

.. raw:: html

    <div class="clearer"></div>
