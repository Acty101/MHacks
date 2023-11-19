import os
import requests
import random
from typing import List


class GPlaceFinder:
    def __init__(
        self,
        url="https://places.googleapis.com/v1/places:searchText",
        gplaces_api_key=None,
    ) -> None:
        self.url = url
        self.gplaces_api_key = (
            gplaces_api_key
            if gplaces_api_key is not None
            else os.environ.get("GPLACES_API_KEY")
        )

    def query(self, query: str, num_return = 3) -> List[dict], bool:
        """Search for information about places or restaurants"""
        headers = {
            "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location",
            "X-Goog-Api-Key": self.gplaces_api_key,
            "Content-Type": "application/json",
        }
        payload = {"textQuery": query}
        response = requests.post(self.url, headers=headers, json=payload)
        if response.status_code == 200:
            items = response.json()["places"]
            items = random.sample(items, min(num_return, len(items)))
            items = [
                {
                    "name": item["displayName"]["text"],
                    "address": item["formattedAddress"],
                    "location": item["location"],
                }
                for item in items
            ]
            return items
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return [], False
