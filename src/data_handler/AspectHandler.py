import json

import requests
from tqdm import tqdm


def chat(user_prompts, url, temperature=0.9, top_p=0.6, top_k=200, max_gen_len=200):
    """输入为batch的提示实施并行处理"""
    headers = {"Content-Type": "application/json"}
    datas = {"model": "openchat_3.5", "prompts": user_prompts, "temperature": temperature, "top_p": top_p, "top_k": top_k, "max_gen_len": max_gen_len}
    json_data = json.dumps(datas)
    response = requests.post(url, headers=headers, data=json_data)
    return response.json()


class AspectHandler(object):
    def __init__(self, args):
        self.prompt = ("Now you are an aspect category and sentiment polarity extractor. "
                       "Your work is to extract aspect category and sentiment polarity pairs from the following sentences:\n {}.\n"
                       "If you could not detect any aspect category and sentiment polarity information from the provided sentences, please just return 'nothing'. "
                       "Remember the polarity should be 'positive', 'negative' and 'neutral'. Note that if the aspect category and sentiment polarity pairs exist, your answer should be a json,"
                       "such as {{'aspect':'price','polarity':'negative'}}. And please removing repeated aspect category.")
        self.args = args

    def extract_aspects(self, sentences):
        sentences = [self.prompt.format(s) for s in sentences]
        # print(sum([len(i.split()) for i in sentences]))
        contents = chat(sentences, self.args.llm_url)
        contents = self.extract_aspect_pairs(contents)
        return contents

    def extract_aspect_pairs(self, contents):
        obj_list = []
        for i in contents:
            json_obj = i.replace('/n', '')
            try:
                obj = eval(json_obj)
                if not isinstance(obj, list):
                    obj = [obj]
            except Exception as e:
                obj = []
            obj_list.append(obj)
        return obj_list

    def process(self, object2reviews):
        object2aspects = {}
        for id in tqdm(object2reviews, ncols=100):
            reviews = object2reviews[id]
            extracted_pairs = self.extract_aspects([v for k, v in reviews.items()])
            object2aspects[id] = extracted_pairs
            if len(extracted_pairs) != len(reviews):
                print('number mismatch between aspects and reviews')
        return object2aspects


if __name__ == '__main__':
    aspect_handler = AspectHandler()
    aspect_handler.extract_aspects(None)
