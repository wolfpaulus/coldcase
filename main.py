"""
    This program demonstrates how to use the OpenAI API to cluster texts, in this casse cold homicide cases summaries. 
    Author: Wolf Paulus (wolfpaulus.com)    
"""
from typing import List, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from openai import OpenAI


def get_content_from_record(cold_case: dict, fields: Tuple[str] = ("gender", "location", "date", "summary")) -> str:
    """ 
        Get a string representation of a record using the specified fields.
        args:
            cold_case: a dictionary with the fields.
            fields: a tuple of strings, the fields to include in the string.
        returns: a string representation of the record.
    """
    return "; ".join([f"{field.capitalize()}: {cold_case[field].strip()}" for field in fields])


def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """
     Get the embedding of a text using the OpenAI API.
     OpenAI embeddings are normalized to length 1, so the cosine similarity between two embeddings is their dot product.
        args:
            text: a string.
            model: the model to use, e.g. "text-embedding-3-small"
        returns: a list of floats, the embedding of the text.        
    """
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def cluster_embeddings(cold_cases: List[dict]) -> KMeans:
    """ Cluster the embeddings of the cold cases. 
    args:
        cold_cases: a list of dictionaries, each with a "embedding" key containing a list of floats.
        returns: a KMeans object. https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.KMeans.html
    """
    embeddings = [case["embedding"] for case in cold_cases]
    matrix = np.vstack(embeddings)
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    kmeans.fit(matrix)
    return kmeans


def plot(cold_cases: List[dict], labels: List[str]):
    """
    Plot the clusters identified in the language 2d using t-SNE.
    args:
        cold_cases: a list of dictionaries, each with a "embedding" key containing a list of floats.
        labels: a list of strings, the labels for the clusters.
    """
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
    embeddings = [case["embedding"] for case in cold_cases]
    matrix = np.vstack(embeddings)
    vis_dims2 = tsne.fit_transform(matrix)
    df = pd.DataFrame(cold_cases)
    x, y = [x for x, y in vis_dims2], [y for x, y in vis_dims2]
    for category, color in enumerate(["purple", "green", "red", "blue"]):
        xs = np.array(x)[df.Cluster == category]
        ys = np.array(y)[df.Cluster == category]
        plt.scatter(xs, ys, color=color, alpha=0.3, label=labels[category])
        avg_x = xs.mean()
        avg_y = ys.mean()
        plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)
    plt.title("Cold Cases Clustered by Summary")
    plt.legend(loc='lower left', fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.show()


def better_labels(cold_cases: List[dict], labels: List[int]) -> List[str]:
    """
    Generate better labels for the clusters.
    args:
        cold_cases: a list of dictionaries, each with a "embedding" key containing a list of floats.
        labels: a list of integers, the cluster labels.
    returns: a list of strings, the better labels for the clusters.
    """
    result = []
    max_per_cluster = 5
    for lbl in sorted(set(labels)):
        x = 0
        content = ""
        for case in cold_cases:
            if case["Cluster"] == lbl:
                x += 1
                content += f"{case['summary']}\n"
            if x >= max_per_cluster:
                break
        messages = [
            {
                "role": "user",
                "content": f'What is the common theme in these unsolved homicide cases?\n\nUnsolved Homicide Cases:\n"""\n{content}\n"""\n\n'
            }
        ]
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)
        result.append(response.choices[0].message.content)
    return result


if __name__ == "__main__":
    with open("./secrets.json", encoding='utf-8') as secrets:
        api_key = json.load(secrets).get("openai_key", None)
        print("API key loaded.")
        client = OpenAI(api_key=api_key)

    with open("./data/madeup_cases.json", encoding='utf-8') as json_file:
        cold_cases = json.load(json_file).get("cold_case_summaries", [])
        print(f"Found {len(cold_cases)} cold cases.")

    if not Path("./data/madeup_cases_with_embedding.json").exists():
        for case in cold_cases:
            case["content"] = get_content_from_record(case)
            case["embedding"] = get_embedding(case["content"])

        with open("./data/madeup_cases_with_embedding.json", "w", encoding='utf-8') as json_file:
            json.dump({"cold_case_summaries": cold_cases}, json_file, indent=4)
            print("Created cold cases file with embeddings.")
    else:
        with open("./data/madeup_cases_with_embedding.json", encoding='utf-8') as json_file:
            cold_cases = json.load(json_file).get("cold_case_summaries", [])
            print(f"Found cold cases with embeddings.")

    x = cluster_embeddings(cold_cases)
    for i, case in enumerate(cold_cases):
        case["Cluster"] = x.labels_[i]
    print("Clustering done.")
    plot(cold_cases, better_labels(cold_cases, x.labels_))
    print("Plotting done.")
