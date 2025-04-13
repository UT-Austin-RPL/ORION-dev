The following images shows a manipulation motion, where the human is manipulating an object. 

Your task is to determine which object is being manipulated in the images below. You need to choose from the following objects: {a list of task-relevant objects}.

Tips: the manipulated object is the object that the human is interacting with, such as picking up, moving, or pressing, and it is in contact with the human or an object (like a utensil) that the human is using.

Your output format is:
```json
{
"manipulate_object_name": "MANIPULATE_OBJECT_NAME",
}
```

Ensure the response can be parsed by Python 'json.loads', e.g.: no trailing commas, no single quotes, etc.