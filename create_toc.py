import os
from pathlib import Path


def create_readme_file(folder, type):
    markdown_string = f"""---
title: {folder}
date: 2022-01-01 00:00:00
description:
---

#### List of all {folder} available on this page  

"""
    to_scan = f'docs/{folder}'

    if type == 'files':
        files = [f.path for f in os.scandir(to_scan)]
        for file in files:
            cur_file = str(file).split('/')[-1]
            if (not cur_file.startswith(".")) and (('README' not in file)):
                print(cur_file)
                link = f"[{cur_file[11:-3]}](https://harsh-maheshwari.github.io/{folder}/{cur_file}/) \n\n"
                markdown_string += link

    elif type == 'dirs':
        subfolders = [f.path for f in os.scandir(to_scan) if f.is_dir()]
        for dir in subfolders:
            cur_dir = str(dir).split('/')[-1]
            if not cur_dir.startswith("."):
                print(cur_dir)
                link = f"[{cur_dir}](https://harsh-maheshwari.github.io/{folder}/{cur_dir}/) \n\n"
                markdown_string += link
    else:
        raise('type should be either files or dirs')

    myFile = open(f"{to_scan}/README.md", "w")
    myFile.write(markdown_string)
    myFile.close()
    return True


def create_readme_files(*args,**kwargs):
    print(f'Blogs : ', create_readme_file('Blogs', type='files'))
    print(f'Projects : ', create_readme_file('Projects', type='dirs'))





