#!/usr/bin/env python
TEMPLATE = "ipybn.css.template"
THEME = "desert.theme"

import re

with open(TEMPLATE, "r") as f:
    css_tpl = [line for line in f]

with open(THEME, "r") as f:
    colors = [
            linecheck for linecheck in 
            [(line.rstrip("\n")).split()[:2][::-1] for line in f] 
            if linecheck]
    
tmpl = []
for line in css_tpl :
    for color in colors :
        line = re.sub(r'__'+color[0]+r'__', color[1], line )
    tmpl.append(line)

with open("ipybn.css", 'w') as f:
    for s in tmpl:
        f.write(s)


