The following images shows a manipulation motion, where the human is manipulating the object "{manipulate object name}".

Please identify the reference object in the image below, which could be an object on which to place {manipulate object name}, or an object that {manipulate object name} is interacting with. Note that there may not necessarily be a reference object, as sometimes human may just playing with the object itself, like throwing it, or spinning it around. You need to first identify whether there is a reference object. If so, you need to output the reference object’s name chosen from the following objects: {a list of task-relevant objects}.

Your output format is:
```json
{
"reference_object_name": "REFERENCE_OBJECT_NAME" or "None",
}
```

Ensure the response can be parsed by Python 'json.loads', e.g.: no trailing commas, no single
quotes, etc.