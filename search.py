import requests
from bs4 import BeautifulSoup

url = "https://www.google.com/maps/search/restaurants+in+<location>"
response = requests.get(url)
html_content = response.text

soup = BeautifulSoup(html_content, "html.parser")

restaurants = soup.find_all("div", class_="section-result-details-container")

for restaurant in restaurants:
    name = restaurant.find("h3", class_="section-result-title").text.strip()
    address = restaurant.find("span", class_="section-result-location").text.strip()
    
    print("Name:", name)
    print("Address:", address)
    print("-" * 50)