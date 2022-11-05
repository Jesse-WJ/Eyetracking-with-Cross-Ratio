'''
Description: draw points and lines in Rhino by pushing '`' on the keyboard
             before execute this .py file, you need install RhinoPython in 
             VScode extensions and config parameter in Rhino.
Author: Jesse
Date: 2022-11-05 20:35:45
LastEditors: Jesse
LastEditTime: 2022-11-05 21:06:11
'''

# coding=utf-8
import Rhino
from rhinoscript.view import CurrentDetail
import scriptcontext as sc
import rhinoscriptsyntax as rs
import System
from System.Drawing import Color
import json


with open("data.json", 'r') as f:
    data = json.load(f)
    for i in range(len(data)):
        rs.AddLayer(str(i))
        for key in data[i]:
            if key == 'Point':
                for names in data[i][key]:
                    name = str(i)+'_'+names.split('_')[0]
                    color = Color.FromArgb(int(names.split('_')[1]), int(
                        names.split('_')[2]), int(names.split('_')[3]))
                    rs.AddLayer(name, parent=str(i))
                    rs.CurrentLayer(name)
                    rs.LayerColor(name, color)
                    for point in data[i][key][names]:
                        rs.AddPoint(point)
            elif key == 'Line':
                for names in data[i][key]:
                    name = str(i)+'_'+names.split('_')[0]
                    color = Color.FromArgb(int(names.split('_')[1]), int(
                        names.split('_')[2]), int(names.split('_')[3]))
                    rs.AddLayer(name, parent=str(i))
                    rs.CurrentLayer(name)
                    rs.LayerColor(name, color)

                    num = len(data[i][key][names][0])
                    for counter in range(num):
                        rs.AddLine(data[i][key][names][0][counter],
                                   data[i][key][names][1][counter])
            elif key == 'Sphere':
                name = str(i)+'_'+key
                rs.AddLayer(name, parent=str(i))
                rs.CurrentLayer(name)
                rs.LayerColor(name, Color.FromArgb(255, 255, 255))
                rs.AddSphere(data[i][key]['center'], data[i][key]['r'])
