import json

# 변경하려는 파일 경로
file_path = "/Users/semyungpark/Documents/homework/week3/2. DistilBERT.ipynb"

# 파일 읽기
with open(file_path, 'r', encoding='utf-8') as file:
    notebook = json.load(file)

# metadata에서 widgets 삭제
if 'widgets' in notebook.get('metadata', {}):
    del notebook['metadata']['widgets']

# 각 셀의 metadata에서도 widgets 삭제
for cell in notebook['cells']:
    if 'widgets' in cell.get('metadata', {}):
        del cell['metadata']['widgets']

# 변경된 내용을 파일에 저장
with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(notebook, file, indent=2, ensure_ascii=False)