from document_reader import Document

# Creating a document object from a pdf
from processors import prj_path
from glob import glob
from tqdm import tqdm
# train_files = glob(prj_path + '/data\\pdf_test\\*.pdf')
train_files = glob('F:\\wine_menu\\Training wine menus/*.pdf')
# train_files = ['D:\\menu_extraction\\data\\pdf\\346. Mustang Harry_s.pdf']
for file in tqdm(train_files):
    document = Document(file)

    print(document.to_json())