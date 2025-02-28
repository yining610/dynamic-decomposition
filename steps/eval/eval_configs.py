import datasets
from typing import List, Dict, Any, Text

def convert_list_to_dict(
        input: List[Dict[Text, Any]]
    ) -> Dict[Text, Any]:

    return {
        key: [item[key] for item in input]
        for key in input[0].keys()
    }

def extract_factscore_claims(dataset: datasets.Dataset):
    claims, subclaims, claims_subclaims = [], [], []
    for topic, data in zip(dataset['topic'], dataset['annotations']):
        if data is None:
            # There are null annotations in the FactScore dataset
            continue 
        for claim in data:
            if not claim['is-relevant']:
                continue
            # remove irrelevant subclaims
            curr_subclaims = [item for item in claim['human-atomic-facts'] if item['label'] != 'IR']
            curr_subclaims = [{'topic': topic, **item} for item in curr_subclaims]
            curr_claim = {'topic': topic, 'text': claim['text'], 'label': 'S' if all([item['label'] == 'S' for item in curr_subclaims]) else 'NS'}
            claims.append(curr_claim)
            subclaims.extend(curr_subclaims)
            claims_subclaims.append({**curr_claim, 'subclaims': curr_subclaims})
    
    all_claims = claims + subclaims

    return convert_list_to_dict(all_claims), claims, subclaims, claims_subclaims

def merge_claims(
        claims: List[Dict[str, Any]], 
        atomicity: int
    ) -> List[Dict[str, Any]]:

    """Recursively merge claims from bottom up to the target atomicity for FactScore dataset.
    """

    if atomicity == 1 or len(claims) == 1:
        return claims, atomicity-1
    else:
        if len(claims) % 2 == 0:
            new_claims = []
            for i in range(0, len(claims), 2):
                new_claims.append({
                    "text": "{claim1} {claim2}".format(
                        claim1=claims[i]['text'],
                        claim2=claims[i+1]['text']
                    ),
                    "topic": claims[i]['topic'],
                    "label": "S" if all([item['label'] == 'S' for item in [claims[i], claims[i+1]]]) else "NS",
                    "subclaims": [claims[i]['text'], claims[i+1]['text']]
                })
        else:
            new_claims = [{
                "text": "{claim1} {claim2} {claim3}".format(
                    claim1=claims[0]['text'],
                    claim2=claims[1]['text'],
                    claim3=claims[2]['text']
                ),
                "topic": claims[0]['topic'],
                "label": "S" if all([item['label'] == 'S' for item in [claims[0], claims[1], claims[2]]]) else "NS",
                "subclaims": [claims[0]['text'], claims[1]['text'], claims[2]['text']]
            }]
            for i in range(3, len(claims), 2):
                new_claims.append({
                    "text": "{claim1} {claim2}".format(
                        claim1=claims[i]['text'],
                        claim2=claims[i+1]['text']
                    ),
                    "topic": claims[i]['topic'],
                    "label": "S" if all([item['label'] == 'S' for item in [claims[i], claims[i+1]]]) else "NS",
                    "subclaims": [claims[i]['text'], claims[i+1]['text']]
                })

        return merge_claims(new_claims, atomicity-1)
