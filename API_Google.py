import os
import requests
from typing import Dict, Any, List, Optional, Tuple

from smolagents import Tool, CodeAgent, tool
from smolagents import LiteLLMModel
import litellm

litellm.model_alias = {}
litellm._turn_on_debug()

from dotenv import load_dotenv

load_dotenv()


class GoogleSearchTool(Tool):
    name = "web_search"
    description = """Performs a Google web search for your query then returns a string of the top search results."""
    inputs = {
        "query": {"type": "string", "description": "The search query to perform."},
        "filter_year": {
            "type": "integer",
            "description": "Optionally restrict results to a certain year",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, api_key=None, cx=None):
        super().__init__()
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")  # os.getenv("GOOGLE_API_KEY")
        self.cx = cx or os.environ.get("GOOGLE_CSE_ID")

    def forward(self, query: str, filter_year: Optional[int] = None) -> str:
        """
        Performs a Google web search using the Google Custom Search API.

        Args:
            query: The search query string
            filter_year: Optional year to filter results by

        Returns:
            Formatted string of search results
        """
        if self.api_key is None:
            raise ValueError("Missing Google API key. Make sure you have 'GOOGLE_API_KEY' in your env variables.")
        if self.cx is None:
            raise ValueError("Missing Custom Search Engine ID. Make sure you have 'GOOGLE_CSE_ID' in your env variables.")
        params: Dict[str, Any] = {
            "key": self.api_key,
            "cx": self.cx,
            "q": query,
        }
        if filter_year is not None:
            params["sort"] = "date"
            params["dateRestrict"] = f"y{filter_year}"
        try:
            response = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            results = response.json()
        except requests.RequestException as e:
            raise ValueError(f"Error making request to Google Custom Search API: {str(e)}")
        if "items" not in results or len(results["items"]) == 0:
            year_filter_message = f" with filter year={filter_year}" if filter_year is not None else ""
            return f"No results found for '{query}'{year_filter_message}. Try with a more general query, or remove the year filter."
        web_snippets: List[str] = []
        for idx, item in enumerate(results["items"]):
            title = item.get("title", "No title")
            link = item.get("link", "#")
            formatted_result = f"{idx + 1}. {link}"
            web_snippets.append(formatted_result)
        return "## Search Results\n" + "\n\n".join(web_snippets)

class GooglePlacesTool(Tool):
    name = "places_search"
    description = """Searches for places using Google Places API and returns information about matching locations."""
    inputs = {
        "query": {"type": "string", "description": "The place or business to search for."},
        "location": {
            "type": "string",
            "description": "Optional latitude,longitude coordinates (e.g. '44.8176,20.4633'). If not provided, defaults to Belgrade, Serbia.",
            "nullable": True,
        },
        "radius": {
            "type": "integer",
            "description": "Search radius in meters. Default is 5000 meters.",
            "nullable": True,
        }
    }
    output_type = "string"

    def __init__(self, api_key=None):
        super().__init__()
        self.google_api_key = os.getenv("GOOGLE_API_KEY", api_key)
        self.API_TYPE = "textsearch"  # Alternative: "findplacefromtext"
        self.url = f"https://maps.googleapis.com/maps/api/place/{self.API_TYPE}/json"

        # Default coordinates (Belgrade, Serbia)
        self.default_lat = 44.8176
        self.default_lng = 20.4633
        self.default_radius = 5000

    def forward(self, query: str, location: Optional[str] = None, radius: Optional[int] = None) -> str:
        """
        Searches for places using Google Places API.

        Args:
            query: The search query for a place or business
            location: Optional comma-separated latitude,longitude string
            radius: Optional search radius in meters

        Returns:
            dict: A strings, every sting containg:
                - name
                - adress
                - "place_id" (str): The unique place identifier.
            }
        """
        if self.google_api_key is None:
            raise ValueError("Missing Google API key. Make sure you have 'GOOGLE_API_KEY' in your env variables.")
        lat, lng = self._parse_location(location)
        search_radius = radius if radius is not None else self.default_radius
        places = self._api_request(query, lat, lng, search_radius)
        return self._format_places(query, places, lat, lng, search_radius)

    def _parse_location(self, location: Optional[str]) -> Tuple[float, float]:
        """Parse location string or return defaults."""
        if location and "," in location:
            try:
                lat_str, lng_str = location.split(",")
                lat = float(lat_str.strip())
                lng = float(lng_str.strip())
                return lat, lng
            except (ValueError, TypeError):
                pass
        return self.default_lat, self.default_lng

    def _api_request(self, query: str, lat: float = None, lng: float = None, radius: int = 5000) -> List[Dict[str, Any]]:
        """Make request to Google Places API."""
        if lat is None:
          lat = self.default_lat
        if lng is None:
          lat = self.default_lng
        params: Dict[str, Any] = {
            "key": self.google_api_key
        }
        if self.API_TYPE == "findplacefromtext":
            params.update({
                "input": query,
                "inputtype": "textquery",
                "fields": "formatted_address,name,place_id,geometry,types,rating,user_ratings_total",
                "locationbias": f"circle:{radius}@{lat},{lng}"
            })
        else:  # textsearch
            params.update({
                "query": query,
                "location": f"{lat},{lng}",
                "radius": radius
            })
        try:
            response = requests.get(self.url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            print(data)
            if self.API_TYPE == "findplacefromtext":
                return data.get("candidates", [])
            else:
                return data.get("results", [])
        except requests.RequestException as e:
            raise ValueError(f"Error making request to Google Places API: {str(e)}")

    def _format_places(self, query: str, places: List[Dict[str, Any]], lat: float, lng: float, radius: int) -> str:
        """Format places results as a readable string."""
        if not places:
            return f"No places found for '{query}' within {radius}m of coordinates {lat},{lng}."
        formatted_results = []
        for idx, place in enumerate(places):
            name = place.get("name", "Unnamed location")
            address = place.get("formatted_address", "No address available")
            place_id = place.get("place_id", "No ID")
            rating_info = ""
            if "rating" in place:
                rating = place.get("rating", 0)
                total_ratings = place.get("user_ratings_total", 0)
                rating_info = f"\nRating: {rating}/5 ({total_ratings} reviews)"
            formatted_results.append({'name': name, 'address': address, 'rating_info': rating_info, 'place_id': place_id})
        return formatted_results

@tool
def get_place_working_hours(place_id: str) -> dict:
    """
    Fetches working hours for a place using the Google Places API.
    
    This function retrieves information about opening hours and closing hours and return a dict with fields: 
            - open_time (str): Opening time in 24-hour format (e.g., '0900')
            - close_time (str): Closing time in 24-hour format (e.g., '1700')
    
    Args:
        place_id: The Google Place ID for the location you want information about
    
    Returns:
        dict: A dictionary containing the place's operating hours with the keys:
            - open_time (str): Opening time in 24-hour format (e.g., '0900')
            - close_time (str): Closing time in 24-hour format (e.g., '1700')
            
    Example:
        >>> result = get_place_working_hours(place_id="ChIJj61dQgK6j4AR4GeTYWZsKWw")
        >>> print(result)
        {'open_time': '0900', 'close_time': '1700'}
    """
    google_api_key = os.environ['GOOGLE_API_KEY']
    fields='name,url,opening_hours'
    print(fields)
    params = {
        "place_id": place_id,
        "fields": fields,
        "key": google_api_key
    }
    url = f"https://maps.googleapis.com/maps/api/place/details/json"
    headers = {}
    response = requests.get(url, params=params, headers=headers).json()
    hours = response['result']['opening_hours']['periods'][0]
    res = {'open_time': hours['open']['time'], 'close_time': hours['close']['time']}
    return res

if __name__ == '__main__':
    api_key = os.getenv("GOOGLE_API_KEY")
    print('Finished')
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    openai_model = LiteLLMModel(
        model_id="gpt-3.5-turbo",
        api_key=OPENAI_API_KEY,
        temperature=0.7,
        max_tokens=4096
    )
    adress_tool = GooglePlacesTool(api_key=api_key)
    agent = CodeAgent(
        tools=[adress_tool, get_place_working_hours],
        model=openai_model,
        additional_authorized_imports=[
            "json",
        ],
    )
    result = agent.run(
        """List of Michelin restaurants in Belgrade with lnks and working hours and adresses

        Use adress_tool tool to get a restaurants list

        Once you have a list of restaurants, utilize the get_place_working_hours tool to retrieve working hours by passing the place ID in the tool one by one.
        
        After identifying a restaurant with the latest closing time, print only this specific restaurant.
        """
    )
    with open('output.txt', 'w', encoding='utf8') as f:
        f.write(result)
