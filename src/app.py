import torch
import pandas as pd
from args import Args
from tqdm import tqdm
from io import BytesIO
from model import Model, load_model
from dataset import CustomDataset
from fastapi import FastAPI, UploadFile, File
from utils import preprocessing_text, id_to_label, remove_duplicates
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn 

app = FastAPI()

@app.post("/predict/")
async def main(file: UploadFile):
    if file.content_type != "text/csv":
        return {"error": "Only CSV files are allowed."}
    
    contents = await file.read()

    data = pd.read_csv(BytesIO(contents))
    data['prepared_text'] = data['text'].apply(preprocessing_text)
    data.drop_duplicates(subset=['text'])
    
    args = Args()
    bert = AutoModel.from_pretrained('model/bert')
    tokenizer = AutoTokenizer.from_pretrained('model/bert')
    
    model = Model(bert=bert, out_dim=args.num_classes)
    checkpoint = torch.load('model/checkpoint.pth', map_location=torch.device(args.device))
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)
    
    predicted_texts = {'text': [], 'channel_id': [], 'category': []}

    for i, row in tqdm(data.iterrows(), desc='Make predictions'):
        with torch.no_grad():
            text = row['prepared_text']
            input_ids = tokenizer.encode(text, padding='max_length',
                                    max_length=args.max_length,
                                    truncation=True,
                                    return_tensors='pt')

            input_ids = input_ids.to(args.device)                       
            channel_id = row['channel_id']

            output = model(input_ids)
            prediction = output.argmax(dim=1).cpu().item()

            predicted_texts['channel_id'].append(channel_id)
            original_text = row['text']
            predicted_texts['text'].append(original_text)
            predicted_texts['category'].append(id_to_label(prediction))

    df_with_labels = pd.DataFrame.from_dict(predicted_texts)
    df_without_duplicates = remove_duplicates(df_with_labels)

    return df_without_duplicates.to_dict()