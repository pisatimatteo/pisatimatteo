
import re
from time import sleep
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver


def parse_html(html):
    """Parse content from various tags from OpenTable restaurants listing"""
    data, item = pd.DataFrame(), {}
    soup = BeautifulSoup(html, 'lxml')
    for i, resto in enumerate(soup.find_all('div', class_='rest-row-info')):
        item['name'] = resto.find('span', class_='rest-row-name-text').text

        booking = resto.find('div', class_='booking')
        item['bookings'] = re.search('\d+', booking.text).group() if booking else 'NA'

        rating = resto.find('div', class_='star-rating-score')
        item['rating'] = float(rating['aria-label'].split()[0]) if rating else 'NA'

        reviews = resto.find('span', class_='underline-hover')
        item['reviews'] = int(re.search('\d+', reviews.text).group()) if reviews else 'NA'

        item['price'] = int(resto.find('div', class_='rest-row-pricing').find('i').text.count('$'))
        item['cuisine'] = resto.find('span', class_='rest-row-meta--cuisine rest-row-meta-text sfx1388addContent').text
        item['location'] = resto.find('span', class_='rest-row-meta--location rest-row-meta-text sfx1388addContent').text
        data[i] = pd.Series(item)
    return data.T


# Start selenium and click through pages until reach end
# store results by iteratively appending to csv file
driver = webdriver.Firefox()
