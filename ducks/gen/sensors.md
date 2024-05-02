## {{doc.class_type.__name__}}

{{doc.description}}

```
{{doc.class_type.__name__}}{% for child in children %}
- {{child.name}} [{{child.class_type.__name__}}]: {{child.description}}{% endfor %}
```
