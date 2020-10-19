import os 
from segment.segmentation_v2 import Segmentation
from segment.process import ProcessJSON
import copy

import random
from cv2 import cv2 as cv
import matplotlib.pyplot as plt
import json as js
import numpy as np
import copy

def main():
    cb  = Combination()
    cb.run_cb('result', 5500)

class Combination(Segmentation):
    def Combination(self):
        self.mask = None
        self.image_name = None
        self.category_name = None

    def run_cb(self, result_path, sample_count): 
        bag_json_path = "{}/input/bag_json".format(result_path)
        bag_image_path = "{}/input/bag".format(result_path)
        emblem_mask_path = "{}/input/emblem/emblem_mask/gucci".format(result_path)
        emblem_image_path = "{}/input/emblem/emblem_image/gucci".format(result_path)
        emblem_json_path = "{}/input/emblem_json/gucci".format(result_path)
        output_json_path = "{}/output/json".format(result_path) 
        output_image_path = "{}/output/image".format(result_path)

        # 파일 리스트
        bag_json_list = os.listdir(bag_json_path)
        emblem_mask_list_ = os.listdir(emblem_mask_path)
        emblem_image_list_ = os.listdir(emblem_image_path)

        # 샘플 만들기
        for _ in range(0,int(sample_count/2)):                
            # bag image 준비
            # bag_file = random.choice(bag_image_list)
            # bag_image_list.remove(bag_file)
            # bag_image = cv.imread("{}/{}".format(bag_image_path, bag_file))
            bag_json= random.choice(bag_json_list)
            bag_json_list.remove(bag_json)
            bag_file = "{}.{}".format(bag_json.split('.')[0],bag_json.split('.')[1]) 
            bag_image = cv.imread("{}/{}".format(bag_image_path, bag_file))
            with open("./{}/{}".format(bag_json_path, bag_json), "r") as f:
                bag_json_file = js.load(f)
            for _ in range(0,2):
                literal = True
                while literal:
                    emblem_image_list = copy.deepcopy(emblem_image_list_)
                    emblem_mask_list = copy.deepcopy(emblem_mask_list_)
                    # emblem image 준비
                    emblem_file = random.choice(emblem_image_list)
                    sp_emblem_file = emblem_file.split('_')[-1].split('.')[0]
                    ma_emblem_file = "{}.{}".format(emblem_file.split('.')[0], 'png')
                    emblem_image_list.remove(emblem_file)
                    emblem_image = cv.imread("{}/{}".format(emblem_image_path, emblem_file),cv.IMREAD_UNCHANGED)
                    # emblem mask 준비
                    emblem_mask_list.remove(ma_emblem_file)
                    mask_iamge = cv.imread("{}/{}".format(emblem_mask_path, emblem_file), cv.IMREAD_GRAYSCALE)
                    # emblem json
                    with open("./{}/{}_{}.json".format(emblem_json_path, emblem_file, sp_emblem_file), "r") as f:
                        emblem_json = js.load(f)
                    
                    #이미지 합성
                    bbox = self.bbox(emblem_image, emblem_json)
                    bbox_mask = self.bbox(mask_iamge, emblem_json)
                    try:
                        cb_img, new_json = self.combination(bag_image, bbox, bbox_mask, bag_json_file, emblem_json)
                        literal = False
                    except Exception:
                        bag_json= random.choice(bag_json_list)
                        bag_json_list.remove(bag_json)
                        bag_file = "{}.{}".format(bag_json.split('.')[0],bag_json.split('.')[1]) 
                        bag_image = cv.imread("{}/{}".format(bag_image_path, bag_file))
                        with open("./{}/{}".format(bag_json_path, bag_json), "r") as f:
                            bag_json_file = js.load(f)
                        literal = True
                    # image,  json 저장
                    split_name = self.image_name.split('.')[0]
                    save_file_name = '{}_{}.jpg'.format(split_name, self.category_name)

                    # json
                    with open('./{}/{}.json'.format(output_json_path, save_file_name),'w') as f:
                        js.dump(new_json,f)
                    cv.imwrite('./{}/{}'.format(output_image_path, save_file_name), cb_img)

    def run(self,json_load_path, json_list, json_save_path, image_save_path):
        jsonfile = ProcessJSON()
        for json_name in json_list:
            jsonfile.jsonFileLoad(json_load_path,json_name)
            self.json_data = jsonfile.json
            self.myjson = copy.deepcopy(self.json_data)
            self.jsonsplit()
            self.segment(json_load_path, json_save_path, image_save_path) 

    def mask_image(self, img, contours):
        img2 = img.copy()
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2BGRA)
        contours = contours.reshape((contours.shape[0], 1, contours.shape[1]))  # cv.fillPoly 형식에 맞춰서 넣어주기 위해서
        mask = np.zeros(img.shape[:-1], np.uint8)
        cv.fillPoly(mask, [contours], 255, cv.LINE_AA)
        img2[mask == 0] = (0, 0, 0, 0)
        self.mask = mask
        return img2

    def save_json(self, filename, json_save_path, category_id):
        if not os.path.isdir(json_save_path): 
            os.mkdir(json_save_path)
        json_name = "{}_{}.json".format(filename, self.imagenumber)
        file_path = "{}/{}".format(json_save_path,json_name)
        self.myjson["images"][0]["file_name"] = filename
        category = [category for category in self.categories if category["id"] == category_id]
        annotation = [annotation for annotation in self.annotations 
                    if annotation["category_id"] == category_id and annotation['area'] <= 2000] 
        myjson_ = {
            "images" : self.myjson["images"],
            "categories" : category,
            "annotations" : annotation
        }
        with open(file_path, 'w') as outfile:
            js.dump(myjson_, outfile)            

    def save(self, json_save_path,category_id, imagename ,image_save_path, result_img):
        if not os.path.isdir(image_save_path):
            os.mkdir(image_save_path)
        path_list = imagename.split('.')
        category = self.categoryname[category_id]
        filename = "{}_{}_{}.{}".format(path_list[0], category,self.imagenumber,"png")
        filepath = "{}/{}".format(image_save_path,filename)
        mask_path = "{}_mask/{}".format(image_save_path, filename)
        cv.imwrite(filepath,result_img)
        cv.imwrite(mask_path, self.mask)
        self.save_json(filename, json_save_path, category_id) 

    def imshow(self, img):
        plt.imshow(img)
        plt.show()

    def return_annotation(self,img_json):
        annotation = img_json['annotations'][0]
        bbox = annotation['bbox']
        seg = annotation['segmentation']
        return bbox, seg

    def return_bbox(self,img_json):
        bbox, _ = self.return_annotation(img_json)
        x_axis = bbox[0]; y_axis = bbox[1]
        x_width = x_axis + bbox[2]; y_height = y_axis + bbox[3] 
        return y_axis, y_height, x_axis, x_width

    def bbox(self, img, img_json):
        y_axis, y_height, x_axis, x_width = self.return_bbox(img_json)
        bbox_img = img[int(y_axis):int(y_height), int(x_axis):int(x_width)]
        return bbox_img

    # 엠블럼을 놓을 곳
    def roi_setting(self, bag_img, bbox_img):
        img_shape = bag_img.shape
        bbox_shape = bbox_img.shape
        bbox_size = bbox_shape[0] * bbox_shape[1]
        # 임시로 지정한 코드 수정할 것
        center_x = int(img_shape[0]/2-bbox_shape[0]/2); center_y = int(img_shape[1]/2 - bbox_shape[1]/2)
        width = center_x + bbox_shape[0]; height = center_y + bbox_shape[1]
        roi = bag_img[center_x:width, center_y:height]
        roi_gray = cv.cvtColor(roi, cv.COLOR_RGBA2GRAY)
        hist = cv.calcHist([roi_gray],[0],None,[256],[0,256])
        white = hist[250:256].sum()
        white_area = (white / bbox_size) * 100
        if white_area >=50:
            raise Exception('bbox정제가 이상해 넘어갈 것')
        return roi, (center_x, center_y, width, height)

    # 실제 합성
    def combination(self, bag_img, bbox_img, bbox_mask,bag_json_file, emblem_json):
        bag_img = cv.cvtColor(bag_img, cv.COLOR_RGB2RGBA)
        x, widht, y, height = self.return_bbox(bag_json_file)
        bbag_img = self.bbox(bag_img, bag_json_file)

        # 합성
        roi, axis = self.roi_setting(bbag_img, bbox_img)
        bbox_inv = cv.bitwise_not(bbox_mask)
        fg = cv.bitwise_and(bbox_img, bbox_img, mask=bbox_mask)
        bg = cv.bitwise_and(roi, roi, mask=bbox_inv)
        combination_img = cv.add(fg, bg)
        bbag_img[axis[0]:axis[2], axis[1]:axis[3]] = combination_img
        bag_img[int(x):int(widht), int(y):int(height)] = bbag_img
        # json의 더해주어야 할 값 계산
        plus_x = int(x) + axis[0]; plus_y = int(y) + axis[1]
        new_json = self.modify_json(emblem_json, plus_x, plus_y, bag_json_file)
        return bag_img, new_json
    
    def modify_json(self, json, plus_x, plus_y,bag_json_file):
        bbox, seg = self.return_annotation(json)
        #0,0으로 바꿔주는 좌표o
        zero_x = bbox[0]; zero_y = bbox[1]
        new_bbox = [
            bbox[0] - zero_x +plus_y,
            bbox[1] - zero_y +plus_x,
            bbox[2],
            bbox[3]
        ]
        size = len(seg[0])
        seg_x = np.array(seg[0][::2]) - zero_x ;seg_y = np.array(seg[0][1::2]) - zero_y
        new_seg = np.zeros(size)
        new_seg[::2] = seg_x + plus_y; new_seg[1::2] = seg_y + plus_x
        new_seg = np.round_(new_seg,1)
        new_seg = new_seg.tolist()
        # json파일 수정 및 추가
        json['annotations'][0]['bbox'] = new_bbox; json['annotations'][0]['segmentation'][0] = new_seg
        # annotations 추가
        bag_json_file['annotations'].append(json['annotations'][0])
        # categories 추가
        bag_json_file['categories'].append(json['categories'][0])
        # 파일 저장할 위한 변수 저장
        self.category_name = json['categories'][0]['name']
        self.image_name = bag_json_file["images"][0]['file_name']
        image_name_split = self.image_name.split('.')
        bag_json_file['images'][0]["file_name"] = "{}_{}.{}".format(image_name_split[0],self.category_name, image_name_split[1])

        return bag_json_file
        

if __name__ =="__main__":
    main()