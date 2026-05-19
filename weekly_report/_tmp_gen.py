# -*- coding: utf-8 -*-
import os

PROJECT = r'E:\code\weekly_report'

def write_file(rel_path, content):
    abspath = os.path.join(PROJECT, rel_path)
    os.makedirs(os.path.dirname(abspath), exist_ok=True)
    with open(abspath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'  Created: {rel_path}')

CODE = {}
