You need to analyze what the human is doing in the images,  then tell me: 1. All the objects in front of the scene (mostly on a plane similar to a table). You should ignore the background objects. 2. The objects of interest. They should be a subset of your answer to the first question. They are likely the objects manipulated by human or near human. Note that there are irrelevant objects in the scene, such as objects that does not move at all or are not interacted with by the human and/or the other objects. You should ignore the irelevant objects. Your output format is:

The human is xxx.
All objects are xxx.
The objects of interest are:
```json
{
"objects": ["OBJECT1", "OBJECT2", ...],
}
```

Ensure the response can be parsed by Python 'json.loads', e.g.: no trailing commas, no single quotes, etc. You should output the names of objects of interest in a list [“OBJECT1”, “OBJECT2”, ...] that can be easily parsed by Python. The name is a string, e.g., “apple”, “pen”, “keyboard”, etc.