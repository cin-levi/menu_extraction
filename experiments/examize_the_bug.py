import json

file_name = "D:\\menu_extraction/data/json/test\\346. Mustang Harry_s.pdf.json"

from utils.convert_menu_to_qa_format import read_data

data = read_data([file_name])

print(data)

with open("debug.json", 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)
