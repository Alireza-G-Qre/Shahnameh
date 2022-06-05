import io
from bs4 import BeautifulSoup

with io.open('./datasets/shahnameh-ferdosi.htm', 'r', encoding='utf-8') as file:
    html = file.read()

soup = BeautifulSoup(html, 'html.parser')


def filter_poems(tag):
    return tag.name == 'span' and tag.has_attr('class') and 'content_text' in tag.get('class') and '****' in tag.get_text()


poems = soup.find_all(filter_poems)

with io.open('./datasets/shahnameh.txt', 'w', encoding='utf-8') as file:
    for item in poems:
        file.writelines(map(lambda x: f'{x}\n', item.get_text().split('****')))
