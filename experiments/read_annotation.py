from document_reader import Document

# Creating a document object from a pdf
from processors import prj_path
from glob import glob
from tqdm import tqdm
train_files = glob(prj_path + '/data\\pdf\\training/*.pdf')
for file in tqdm(train_files):
    document = Document(file)

    print(document.to_json())