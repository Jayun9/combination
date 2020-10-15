from segment.process import ProcessJSON

sp = ProcessJSON()
sp.jsonFileSplit("./","BAG002-40.json")
sp.jsonFileSplitSave("split_json_bag002")



