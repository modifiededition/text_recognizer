import pandas as pd
import json
from PIL import ImageStat
from PIL import Image

from evidently import ColumnMapping
from evidently.report import Report
from evidently.test_suite import TestSuite

from evidently.metrics import DataDriftTable, TextDescriptorsDriftMetric, ColumnDriftMetric
from evidently.metric_preset import TextOverviewPreset
from evidently.descriptors import TextLength, TriggerWordsPresence, OOV, NonLetterCharacterPercentage, SentenceCount, WordCount, Sentiment
from evidently.tests import *


test_labels_path = "/teamspace/studios/this_studio/handwritting_text_recognizer/data/processed/iam_paragraphs/test/_labels.json"
train_labels_path = "/teamspace/studios/this_studio/handwritting_text_recognizer/data/processed/iam_paragraphs/train/_labels.json"
val_labels_path = "/teamspace/studios/this_studio/handwritting_text_recognizer/data/processed/iam_paragraphs/val/_labels.json"
image_path = "/teamspace/studios/this_studio/handwritting_text_recognizer/data/processed/iam_paragraphs/{}/{}.png"

column_mapping = ColumnMapping(
   numerical_features=['image_mean_intensity', 'image_median','image_area'],
     text_features=['text_label']
)

def prepare_text_label_data(data_path):
    with open(data_path, 'r') as file:
        labels_data = json.load(file)
    
    ids = list(labels_data.keys())
    text_labels = list(labels_data.values())

    return pd.DataFrame(zip(ids,text_labels), columns=["id","text_label"])

def prepate_image_data_stats(data, data_type):

    ids = list(data["id"].unique())
    
    avg_intensity_list = []
    img_median_list = []
    image_extrema_list = []
    image_area_list = []

    for id in ids:
        image = Image.open(image_path.format(data_type,id))
        stats = ImageStat.Stat(image)
        avg_intensity_list.append(stats.mean[0])
        img_median_list.append(stats.median[0])
        image_area_list.append(image.size[0] * image.size[1])

    return pd.DataFrame({"id":ids,
    "image_mean_intensity":avg_intensity_list ,
    "image_median":img_median_list ,
    "image_area":image_area_list })

def merge_data(a,b):
    return a.merge(b,on ="id")

def prepate_text_data():
    train_labels_data = prepare_text_label_data(train_labels_path)
    val_labels_data = prepare_text_label_data(val_labels_path)
    test_labels_data = prepare_text_label_data(test_labels_path)
    return train_labels_data,val_labels_data,test_labels_data

def prepare_image_data(train_labels_data ,val_labels_data ,test_labels_data):
    train_image_data = prepate_image_data_stats(train_labels_data,"train")
    val_image_data = prepate_image_data_stats(val_labels_data,"val")
    test_image_data = prepate_image_data_stats(test_labels_data,"test")
    return train_image_data,val_image_data,test_image_data

def merge_datasets(a,b,c):
    train_labels_data, train_image_data = a
    val_labels_data,val_image_data = b
    test_labels_data, test_image_data = c
    train_data = merge_data(train_labels_data,train_image_data)
    val_data = merge_data(val_labels_data,val_image_data)
    test_data = merge_data(test_labels_data,test_image_data)
    
    return train_data,val_data,test_data


def create_report(ref_data,current_data):
    text_overview_report = Report(metrics=[
    TextOverviewPreset(column_name="text_label", descriptors={
       "Generated text - OOV %" : OOV(),
       "Generated text - Non Letter %" : NonLetterCharacterPercentage(),
       "Generated text - Symbol Length" : TextLength(),
       "Generated text - Sentence Count" : SentenceCount(),
       "Generated text- Word Count" : WordCount(),
       })
       ])

    text_overview_report.run(reference_data=ref_data, current_data=current_data, column_mapping=column_mapping)
    return text_overview_report


if __name__ == "__main__":
    train_labels_data,val_labels_data,test_labels_data = prepate_text_data()
    train_image_data,val_image_data,test_image_data = prepate_image_data(train_labels_data,val_labels_data,test_labels_data)
    train_image_data,val_image_data,test_image_data = merge_datasets(
                                                            (train_labels_data,train_image_data),
                                                            (val_labels_data,val_image_data),
                                                            (test_labels_data,test_image_data)
    )

    return create_report(train_image_data,test_image_data)



