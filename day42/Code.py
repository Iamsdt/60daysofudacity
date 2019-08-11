from zipfile import ZipFile
from glob import glob

import zipfile
import os

zip_dir = "data/ext"
for subdir, dirs, files in os.walk(zip_dir):
    for zipp in files:
        name = str(zipp).replace('.zip', '')
        filepath = os.path.join(subdir, zipp)
        if filepath.endswith('.zip'):
            zip_ref = zipfile.ZipFile(filepath, 'r')
            zip_ref.extractall('data/ext/'+name)
            zip_ref.close()
        else:
            print("File is not zip", filepath)

print("Done")
