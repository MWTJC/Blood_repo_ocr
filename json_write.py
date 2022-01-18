from collections import defaultdict, OrderedDict
import json

x = OrderedDict()
x["label"] = "haha"
x["data"] = 234
x["score"] = 0.3

test_dict = {
    'version': "1.0",
    'results': x,
    'explain': {
        'used': True,
        'details': "this is for josn test",
  }
}

json_str = json.dumps(test_dict, indent=4)
with open('test_data.json', 'w') as json_file:
    json_file.write(json_str)


