import json
import os
import pandas as pd
import time
import glob

def main():
    path_to_folder = 'dop_sbor'

    category_list = os.listdir(path_to_folder)
    content = {'text': [], 'label': []}

    for category in category_list:
        root_folder = path_to_folder + f'/{category}'
        json_files = glob.glob(os.path.join(root_folder, '**', '*.json'), recursive=True)

        for path_json in json_files:
            name_category = category
            if os.path.exists(path_json):
                try:
                    with open(path_json, 'r', encoding='UTF-8') as f:
                        data = json.load(f)

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    return 
                except Exception as e:
                    print(f"An error occurred: {e}")
                    return 
            else:
                print(f"The file '{path_json}' does not exist.")
                return 

            for message in data['messages']:
                text = message['text']
                
                result = ""
                if type(text) == list:
                    for el in text:
                        if type(el) == str and len(el) > 20:
                            result += el.replace('Обсуждение:', '')
                            content['text'].append(result)
                            content['label'].append(name_category)

                if type(text) == str and len(text) > 20:
                    result += text.replace('Обсуждение:', '')
                    content['text'].append(result)
                    content['label'].append(name_category)
    
    timestamp = int(time.time())
    
    df = pd.DataFrame.from_dict(content)
    df.to_csv(f'result {timestamp}.csv', index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    main()