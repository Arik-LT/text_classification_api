from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
import pickle
import pandas as pd
from functions import cleaning_text, removing_single_letters

app = FastAPI()

class Scoring_item(BaseModel):
        text: str

### Pickle files created for the tfidf vectorizer and final model used which was a liner SVC rapped in a Label Powerset with accruacy over 90%.
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('finalized_model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.post('/')
async def scoring_endpoint(item:Scoring_item):
    df_exp = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    vectors = tfidf.transform(df_exp)
    results = model.predict(vectors)

    return {"item":results}
