:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members:
   :show-inheritance:

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::

   {% for item in methods %}
   {%- if not item.startswith('_') or item in ['__call__'] %}   ~{{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Properties

   .. autosummary::

   {% for item in attributes %}
   {%- if not item.startswith('_') or item in ['__call__'] %}   ~{{ name }}.{{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

.. include:: {{fullname}}.examples

.. raw:: html

    <div class="clearer"></div>
