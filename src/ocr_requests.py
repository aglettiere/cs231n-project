import requests
from PIL import Image
import os
from pathlib import Path
import csv
import json
import time

def ocr_space_file(filename, overlay=True, api_key='helloworld', language='eng'):
    """ OCR.space API request with local file.
        Python3.5 - not tested on 2.7
    :param filename: Your file path & name.
    :param overlay: Is OCR.space overlay required in your response.
                    Defaults to False.
    :param api_key: OCR.space API key.
                    Defaults to 'helloworld'.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'en'.
    :return: Result in JSON format.
    """

    payload = {'isOverlayRequired': overlay,
               'apikey': api_key,
               'language': language,
               }
    with open(filename, 'rb') as f:
        r = requests.post('https://api.ocr.space/parse/image',
                          files={filename: f},
                          data=payload,
                          )
    return r.content.decode()


def ocr_space_url(url, overlay=False, api_key='helloworld', language='eng'):
    """ OCR.space API request with remote file.
        Python3.5 - not tested on 2.7
    :param url: Image url.
    :param overlay: Is OCR.space overlay required in your response.
                    Defaults to False.
    :param api_key: OCR.space API key.
                    Defaults to 'helloworld'.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'en'.
    :return: Result in JSON format.
    """

    payload = {'url': url,
               'isOverlayRequired': overlay,
               'apikey': api_key,
               'language': language,
               }
    r = requests.post('https://api.ocr.space/parse/image',
                      data=payload,
                      )
    return r.content.decode()

'''with open('cover_text.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_path", "full_text", "text_bounds"])'''

# Use examples:
directory = '/Users/allisonlettiere/Downloads/National_Geographic_Covers/data/images/'
### SKIPPED /Users/allisonlettiere/Downloads/National_Geographic_Covers/data/images/cover_07_2006.jpg
file_counter = 0
for filename in os.listdir(directory):
  file_counter += 1

  if file_counter > 378:
    if file_counter % 10 == 0 and file_counter!=0:
      time.sleep(600)

    image_path = os.path.join(directory, filename)

    image_ocr_file = ocr_space_file(filename=image_path,language='eng')
    dframe = json.loads(image_ocr_file)
    parsed_text_boxes = dframe['ParsedResults'][0]['TextOverlay']['Lines']

    parsed_text_full = dframe['ParsedResults'][0]['ParsedText']

    with open('cover_text.csv','a') as file:
      writer = csv.writer(file)
      writer.writerow([image_path, parsed_text_full, parsed_text_boxes])


  #test_url = ocr_space_url(url='http://i.imgur.com/31d5L5y.jpg')