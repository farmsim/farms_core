<a id="ref-{{doc.class_type.__name__}}"></a>
## {{doc.class_type.__name__}}

{{doc.description}}

{% if children %}**Attributes:**{% endif %}
{% for child in children %}
- `{{child.name}}` (`{{child.class_type}}`): {{child.description}}{% endfor %}

{% if methods %}**Methods:**{% endif %}
{% for method in methods %}
- `{{method.name}}`: {{method.description}}{% endfor %}
