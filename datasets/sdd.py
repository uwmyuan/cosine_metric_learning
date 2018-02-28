#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2018-2-15 13:58:51
extract boxes from a video file
@author: yunyuan
"""
import numpy as np
import os
import cv2
import pandas as pd
IMAGE_SHAPE = 128, 128, 3


def read_test_split_to_str(dataset_dir):
	
	return None


def extract_boxes(video_file_name,annotation_file_name):
	ids=[]
	images=[]
	if video_file_name and annotation_file_name:
		df=pd.read_csv(annotation_file_name,sep=" ",header=None,names=["ID","xmin","ymin","xmax","ymax","frame","lost","occluded","generated","label"])
		df.apply(lambda x: pd.to_numeric(x, errors='ignore',downcast='integer'))
		vidcap = cv2.VideoCapture(video_file_name)
		for i in range(len(df)):
			vidcap.set(1,df.at[i,'frame'])
			success,image = vidcap.read()
			if(success):
				#record id
				ids.append(df.at[i, 'ID'])
				#crop image
				image=image[df.at[i,'ymin']:df.at[i,'ymax'],df.at[i,'xmin']:df.at[i,'xmax']]
				#resize image
				image=cv2.resize(image,IMAGE_SHAPE[0:2])
				#record image
				images.append(image)
			else:
				continue
	return images,ids,[0]


def calculate_max_label(annotation_file_name):
	df=pd.read_csv(annotation_file_name,sep=" ",header=None,names=["ID","xmin","ymin","xmax","ymax","frame","lost","occluded","generated","label"])
	df.apply(lambda x: pd.to_numeric(x, errors='ignore',downcast='integer'))
	return len(df['ID'].unique())


def read_train_split_to_str(video_file_name,annotation_file_name):
	images,ids,_=extract_boxes(video_file_name,annotation_file_name)
	np.save("ids_sdd",ids)
	np.save("images_sdd",images)
	return images,ids,[0]


if __name__ == '__main__':
	dir_name="./StanfordDroneDataset"
	annotation_file_name=dir_name+"/annotations/nexus/video0/annotations.txt"
	video_file_name=dir_name+"/videos/nexus/video0/video.mov"
	read_train_split_to_str(video_file_name,annotation_file_name)
	print("Done.")

